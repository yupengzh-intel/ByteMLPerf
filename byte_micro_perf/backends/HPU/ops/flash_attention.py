import itertools
import math
import os
import sys
import pathlib
from functools import partial
from time import time
from typing import Any, List, Optional
import torch
import types

import habana_frameworks.torch as ht

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[3])
)

from core.ops.llm_ops import FlashAttentionOp
from core.utils import OpTensorInfo, calc_tensor_size

OP_MAPPING = {}

try:

    from habana_frameworks.torch.hpex.kernels import FusedSDPA

    from vllm_hpu_extension import ops
    from vllm_hpu_extension.utils import Matmul, VLLMKVCache
    import vllm_hpu_extension.environment as environment

    model_config = types.SimpleNamespace()
    setattr(model_config, "model_type", "llama")

    environment.set_model_config(model_config)

    class FusedSDPAOP(FlashAttentionOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            # currently not support c8
            if self.dtype != self.cache_dtype:
                raise NotImplementedError("not support q_dtype != cache_dtype")

            if self.mode == "prefill":

                self.input_tensor_info.update(
                    {
                        "q": OpTensorInfo(
                            shape=[
                                self.batch_size,
                                self.num_tokens // self.batch_size,
                                self.q_head_num,
                                self.head_dim,
                            ],
                            dtype=self.torch_dtype,
                            device=self.backend.get_torch_device_name(),
                        ),
                        "k_cache": OpTensorInfo(
                            shape=[
                                self.batch_size,
                                self.max_kv_len,
                                self.kv_head_num,
                                self.head_dim,
                            ],
                            dtype=self.cache_torch_dtype,
                            device=self.backend.get_torch_device_name(),
                            creator=torch.empty,
                        ),
                        "v_cache": OpTensorInfo(
                            shape=[
                                self.batch_size,
                                self.max_kv_len,
                                self.kv_head_num,
                                self.head_dim,
                            ],
                            dtype=self.cache_torch_dtype,
                            device=self.backend.get_torch_device_name(),
                            creator=torch.empty,
                        ),
                    }
                )
            elif self.mode == "decode" and self.q_seq_len == 1:
                block_size = 128
                block_groups = [-1]
                block_bias = [[0] + [float("-inf")] * (block_size - 1)]

                for i, kv_len in enumerate(self.kv_lens):
                    block_num = math.ceil(kv_len / block_size)
                    block_groups = block_groups + [i] * block_num
                    while kv_len >= block_size:
                        block_bias.append([0] * block_size)
                        kv_len -= block_size
                    if kv_len > 0:
                        block_bias.append(
                            [0] * kv_len + [float("-inf")] * (block_size - kv_len)
                        )

                self.block_groups = torch.tensor(
                    block_groups,
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name()
                )

                self.block_bias = torch.tensor(
                    block_bias,
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                )

                self.block_list = torch.tensor(
                    range(len(block_groups)),
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name(),
                )

                self.block_mapping = torch.nn.functional.one_hot(
                    self.block_groups, num_classes=self.batch_size
                ).to(self.torch_dtype)

                self.matmul_qk = Matmul()
                self.matmul_av = Matmul()
                self.batch2block_matmul = Matmul()
                self.block2batch_matmul = Matmul()
                self.vllm_k_cache = VLLMKVCache()
                self.vllm_v_cache = VLLMKVCache()

                self.input_tensor_info.update(
                    {
                        "q": OpTensorInfo(
                            shape=[self.batch_size, 1, self.q_head_num * self.head_dim],
                            dtype=self.torch_dtype,
                            device=self.backend.get_torch_device_name(),
                        ),
                        "k_cache": OpTensorInfo(
                            shape=[
                                self.batch_size * self.max_kv_len // block_size + 10,
                                block_size,
                                self.kv_head_num,
                                self.head_dim,
                            ],
                            dtype=self.cache_torch_dtype,
                            device=self.backend.get_torch_device_name(),
                            creator=torch.empty,
                        ),
                        "v_cache": OpTensorInfo(
                            shape=[
                                self.batch_size * self.max_kv_len // block_size + 10,
                                block_size,
                                self.kv_head_num,
                                self.head_dim,
                            ],
                            dtype=self.cache_torch_dtype,
                            device=self.backend.get_torch_device_name(),
                            creator=torch.empty,
                        ),
                    }
                )
            else:
                raise NotImplementedError("not support")

        def flash_attention_run(self, tensor_mapping):
            # get pre-allocated tensors
            q = tensor_mapping["q"]
            q_lens = tensor_mapping["q_lens"]
            accum_q_lens = tensor_mapping["accum_q_lens"]
            cache_lens = tensor_mapping["cache_lens"]
            cache_slot_ids = tensor_mapping["cache_slot_ids"]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]
            k_scale = tensor_mapping.get("k_scale", None)
            v_scale = tensor_mapping.get("v_scale", None)

            if self.mode == "prefill" and self.cache_len == 0:
                out = FusedSDPA.apply(q, k_cache, v_cache, None, 0.0, True)
            elif self.mode == "decode":
                out = ops.flat_pa(
                    query=q,
                    key_cache=k_cache,
                    value_cache=v_cache,
                    block_list=self.block_list,
                    block_mapping=self.block_mapping,
                    block_bias=self.block_bias,
                    block_groups=self.block_groups,
                    scale=1.0,
                    matmul_qk_op=self.matmul_qk,
                    matmul_av_op=self.matmul_av,
                    batch2block_matmul_op=self.batch2block_matmul,
                    block2batch_matmul_op=self.block2batch_matmul,
                    keys_fetch_func=self.vllm_k_cache.fetch_from_cache,
                    values_fetch_func=self.vllm_v_cache.fetch_from_cache,
                )

            return out
    OP_MAPPING["fused_sdpa"] = FusedSDPAOP
except:
    pass
