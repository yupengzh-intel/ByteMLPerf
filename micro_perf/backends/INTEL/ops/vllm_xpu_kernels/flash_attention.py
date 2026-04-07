import sys
import math
import pathlib
from functools import partial
import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.llm_ops import FlashAttentionOp


try:
    from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func, FA2_AVAILABLE
    if not FA2_AVAILABLE:
        raise ImportError("vllm_xpu_kernels FA2 not available")

    @ProviderRegistry.register_vendor_impl("flash_attention", "vllm_xpu_kernels")
    class VLLMXPUKernelsFlashAttentionOp(FlashAttentionOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["vllm_xpu_kernels"]

            if self.attn_mode == "prefill":
                self._prefill_init()
            elif self.attn_mode == "decode":
                self._decode_init()

        def _prefill_init(self):
            if not (
                self.dtype == "bfloat16"
                and self.pv_compute_dtype == "bfloat16"
                and self.cache_dtype == "bfloat16"
            ):
                raise ValueError(
                    "VLLMXPUKernelsFlashAttentionOp only supports bfloat16 for prefill"
                )
            if self.cache_type != "linear":
                raise ValueError(
                    "VLLMXPUKernelsFlashAttentionOp only supports linear cache for prefill"
                )
            self._run_func = self._prefill_run

        def _prefill_run(self, tensor_mapping):
            q = tensor_mapping["q"]  # [num_tokens, q_head_num, head_dim]
            accum_q_lens = tensor_mapping["accum_q_lens"]
            accum_kv_lens = tensor_mapping["accum_kv_lens"]

            # For prefill with linear cache, k_cache/v_cache: [B, H, S, D]
            # Reshape to varlen format: [total_kv_tokens, kv_head_num, head_dim]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]

            # Convert from [B, H, S, D] to [total_tokens, H, D]
            B, H, S, D = k_cache.shape
            k = k_cache.permute(0, 2, 1, 3).contiguous().reshape(B * S, H, D)
            v = v_cache.permute(0, 2, 1, 3).contiguous().reshape(B * S, H, D)

            # Use actual sequence lengths for cu_seqlens
            cu_seqlens_q = accum_q_lens.to(torch.int32)
            cu_seqlens_k = accum_kv_lens.to(torch.int32)

            max_seqlen_q = max(self.q_lens)
            max_seqlen_k = max(self.kv_lens)

            # vllm_xpu_kernels API: (q, k, v, max_seqlen_q, cu_seqlens_q,
            #                        max_seqlen_k, cu_seqlens_k, ...)
            out = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q, cu_seqlens_q,
                max_seqlen_k, cu_seqlens_k,
                causal=self.is_causal,
            )
            return out

        def _decode_init(self):
            if not (
                self.dtype == "bfloat16"
                and self.pv_compute_dtype == "bfloat16"
                and self.cache_dtype == "bfloat16"
            ):
                raise ValueError(
                    "VLLMXPUKernelsFlashAttentionOp only supports bfloat16 for decode"
                )
            if self.cache_type != "linear":
                raise ValueError(
                    "VLLMXPUKernelsFlashAttentionOp only supports linear cache for decode"
                )
            self._run_func = self._decode_run

        def _decode_run(self, tensor_mapping):
            q = tensor_mapping["q"]  # [num_tokens, q_head_num, head_dim]
            accum_q_lens = tensor_mapping["accum_q_lens"]
            accum_kv_lens = tensor_mapping["accum_kv_lens"]

            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]

            # Convert from [B, H, S, D] to [total_kv_tokens, H, D]
            B, H, S, D = k_cache.shape
            k = k_cache.permute(0, 2, 1, 3).reshape(B * S, H, D)
            v = v_cache.permute(0, 2, 1, 3).reshape(B * S, H, D)

            cu_seqlens_q = accum_q_lens.to(torch.int32)
            cu_seqlens_k = accum_kv_lens.to(torch.int32)

            max_seqlen_q = max(self.q_lens)
            max_seqlen_k = max(self.kv_lens)

            # vllm_xpu_kernels API: (q, k, v, max_seqlen_q, cu_seqlens_q,
            #                        max_seqlen_k, cu_seqlens_k, ...)
            out = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q, cu_seqlens_q,
                max_seqlen_k, cu_seqlens_k,
                causal=self.is_causal,
            )
            return out

except Exception:
    pass
