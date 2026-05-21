from functools import partial

import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf.micro_perf.core.utils import (
    OpTensorInfo,
    calc_tensor_size,
    ceil_div,
    create_from_list,
    get_torch_dtype,
)


try:
    import flashinfer

    _FP8_TORCH_DTYPE = torch.float8_e4m3fn


    @ProviderRegistry.register_vendor_impl("moe_quant_group_gemm", "flashinfer")
    class FlashInferMoeQuantGroupGemmOp:
        def _resolve_variant(self):
            if (
                self.dtype == "bfloat16"
                and self.w_dtype == "bfloat16"
                and self.dst_dtype == "bfloat16"
                and self.compute_dtype in ["int8", "bfloat16"]
            ):
                return "bf16"

            if (
                self.dtype == "fp8_e4m3"
                and self.w_dtype == "fp8_e4m3"
                and self.compute_dtype == "fp8"
                and self.dst_dtype == "bfloat16"
            ):
                return "fp8_per_tensor"

            if (
                self.dtype == "fp8_e4m3"
                and self.w_dtype == "fp8_e4m3"
                and self.compute_dtype == "fp8_block"
                and self.dst_dtype == "bfloat16"
            ):
                return "fp8_block"

            if (
                self.dtype == "mxfp4"
                and self.w_dtype == "mxfp4"
                and self.compute_dtype == "mxfp4"
                and self.dst_dtype == "bfloat16"
            ):
                return "mxfp4"

            if (
                self.dtype == "nvfp4"
                and self.w_dtype == "nvfp4"
                and self.compute_dtype == "nvfp4"
                and self.dst_dtype == "bfloat16"
            ):
                return "nvfp4"

            raise ValueError(
                f"{type(self).__name__} not support: "
                f"dtype={self.dtype}, w_dtype={self.w_dtype}, "
                f"compute_dtype={self.compute_dtype}, dst_dtype={self.dst_dtype}"
            )

        def _get_segment_offsets_data(self):
            if self.num_experts_per_rank == 0:
                return [0]

            segment_offsets = list(self.expert_dispatch_token_offset)
            segment_offsets.append(
                self.expert_dispatch_token_offset[-1]
                + self.expert_dispatch_token_count[-1]
            )
            return segment_offsets

        def _get_fp4_a_scale_rows(self):
            alignment = 128
            total_rows = 0

            for group_idx, (offset, count) in enumerate(
                zip(
                    self.expert_dispatch_token_offset,
                    self.expert_dispatch_token_count,
                )
            ):
                # FlashInfer's FP4 grouped GEMM pads each group scale matrix
                # independently and places later groups at aligned row offsets.
                padded_offset = (
                    ceil_div(offset + group_idx * (alignment - 1), alignment)
                    * alignment
                )
                padded_count = ceil_div(count, alignment) * alignment
                total_rows = max(total_rows, padded_offset + padded_count)

            return total_rows

        def vendor_parser(self):
            self.variant = self._resolve_variant()

            if self.variant == "fp8_block":
                if self.hidden_size % 128 != 0 or self.new_hidden_size % 128 != 0:
                    raise ValueError(
                        f"{type(self).__name__} fp8_block requires hidden_size and "
                        f"new_hidden_size be multiples of 128, but got "
                        f"hidden_size={self.hidden_size}, "
                        f"new_hidden_size={self.new_hidden_size}"
                    )

            if self.variant in ["mxfp4", "nvfp4"]:
                if self.hidden_size % 128 != 0:
                    raise ValueError(
                        f"{type(self).__name__} {self.variant} requires hidden_size "
                        f"be a multiple of 128, but got hidden_size={self.hidden_size}"
                    )
                if self.new_hidden_size % 8 != 0:
                    raise ValueError(
                        f"{type(self).__name__} {self.variant} requires "
                        f"new_hidden_size be a multiple of 8, but got "
                        f"new_hidden_size={self.new_hidden_size}"
                    )

        def vendor_impl(self):
            self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)
            self.segment_offsets_data = self._get_segment_offsets_data()
            self.extra_providers = ["flashinfer"]

            self.input_tensor_info = {}
            self.output_tensor_info = {}

            device = self.backend.get_torch_device_name()

            if self.variant == "bf16":
                self.torch_dtype = get_torch_dtype("bfloat16")
                self.w_torch_dtype = self.torch_dtype

                self.input_tensor_info["scatter_tokens"] = OpTensorInfo(
                    shape=[self.dispatch_tokens, self.hidden_size],
                    dtype=self.torch_dtype,
                    device=device,
                    creator=torch.zeros,
                )
                self.input_tensor_info["experts_weight"] = OpTensorInfo(
                    shape=[
                        self.num_experts_per_rank,
                        self.new_hidden_size,
                        self.hidden_size,
                    ],
                    dtype=self.w_torch_dtype,
                    device=device,
                    creator=torch.zeros,
                )

            elif self.variant == "fp8_per_tensor":
                self.torch_dtype = _FP8_TORCH_DTYPE
                self.w_torch_dtype = self.torch_dtype

                self.input_tensor_info["scatter_tokens"] = OpTensorInfo(
                    shape=[self.dispatch_tokens, self.hidden_size],
                    dtype=self.torch_dtype,
                    device=device,
                    creator=torch.zeros,
                )
                self.input_tensor_info["experts_weight"] = OpTensorInfo(
                    shape=[
                        self.num_experts_per_rank,
                        self.new_hidden_size,
                        self.hidden_size,
                    ],
                    dtype=self.w_torch_dtype,
                    device=device,
                    creator=torch.zeros,
                )
                self.input_tensor_info["alpha"] = OpTensorInfo(
                    shape=[1],
                    dtype=torch.float32,
                    device=device,
                    creator=torch.ones,
                )

            elif self.variant == "fp8_block":
                self.torch_dtype = _FP8_TORCH_DTYPE
                self.w_torch_dtype = self.torch_dtype

                self.input_tensor_info["scatter_tokens"] = OpTensorInfo(
                    shape=[self.dispatch_tokens, self.hidden_size],
                    dtype=self.torch_dtype,
                    device=device,
                    creator=torch.zeros,
                )
                self.input_tensor_info["experts_weight"] = OpTensorInfo(
                    shape=[
                        self.num_experts_per_rank,
                        self.new_hidden_size,
                        self.hidden_size,
                    ],
                    dtype=self.w_torch_dtype,
                    device=device,
                    creator=torch.zeros,
                )
                self.input_tensor_info["a_scale"] = OpTensorInfo(
                    shape=[self.hidden_size // 128, self.dispatch_tokens],
                    dtype=torch.float32,
                    device=device,
                    creator=torch.ones,
                )
                self.input_tensor_info["b_scale"] = OpTensorInfo(
                    shape=[
                        self.num_experts_per_rank,
                        self.hidden_size // 128,
                        self.new_hidden_size // 128,
                    ],
                    dtype=torch.float32,
                    device=device,
                    creator=torch.ones,
                )

            elif self.variant == "mxfp4":
                self.torch_dtype = _FP8_TORCH_DTYPE
                self.w_torch_dtype = torch.uint8

                new_hidden_size_aligned = ceil_div(self.new_hidden_size, 128) * 128
                a_scale_rows = self._get_fp4_a_scale_rows()

                self.input_tensor_info["scatter_tokens"] = OpTensorInfo(
                    shape=[self.dispatch_tokens, self.hidden_size],
                    dtype=self.torch_dtype,
                    device=device,
                    creator=torch.zeros,
                )
                self.input_tensor_info["experts_weight"] = OpTensorInfo(
                    shape=[
                        self.num_experts_per_rank,
                        self.new_hidden_size,
                        self.hidden_size // 2,
                    ],
                    dtype=self.w_torch_dtype,
                    device=device,
                    creator=torch.zeros,
                )
                self.input_tensor_info["a_scale"] = OpTensorInfo(
                    shape=[a_scale_rows, self.hidden_size // 32],
                    dtype=torch.uint8,
                    device=device,
                    creator=torch.ones,
                )
                self.input_tensor_info["b_scale"] = OpTensorInfo(
                    shape=[
                        self.num_experts_per_rank,
                        new_hidden_size_aligned,
                        self.hidden_size // 32,
                    ],
                    dtype=torch.uint8,
                    device=device,
                    creator=torch.ones,
                )

            elif self.variant == "nvfp4":
                self.torch_dtype = torch.uint8
                self.w_torch_dtype = torch.uint8

                new_hidden_size_aligned = ceil_div(self.new_hidden_size, 128) * 128
                a_scale_rows = self._get_fp4_a_scale_rows()

                self.input_tensor_info["scatter_tokens"] = OpTensorInfo(
                    shape=[self.dispatch_tokens, self.hidden_size // 2],
                    dtype=self.torch_dtype,
                    device=device,
                    creator=torch.zeros,
                )
                self.input_tensor_info["experts_weight"] = OpTensorInfo(
                    shape=[
                        self.num_experts_per_rank,
                        self.new_hidden_size,
                        self.hidden_size // 2,
                    ],
                    dtype=self.w_torch_dtype,
                    device=device,
                    creator=torch.zeros,
                )
                self.input_tensor_info["a_scale"] = OpTensorInfo(
                    shape=[a_scale_rows, self.hidden_size // 16],
                    dtype=torch.uint8,
                    device=device,
                    creator=torch.ones,
                )
                self.input_tensor_info["b_scale"] = OpTensorInfo(
                    shape=[
                        self.num_experts_per_rank,
                        new_hidden_size_aligned,
                        self.hidden_size // 16,
                    ],
                    dtype=torch.uint8,
                    device=device,
                    creator=torch.ones,
                )
            else:
                raise ValueError(f"Unexpected FlashInfer variant: {self.variant}")

            self.input_tensor_info["segment_offsets"] = OpTensorInfo(
                shape=[self.num_experts_per_rank + 1],
                dtype=torch.int32,
                device=device,
                creator=partial(create_from_list, data=self.segment_offsets_data),
            )

            self.output_tensor_info["y"] = OpTensorInfo(
                shape=[self.dispatch_tokens, self.new_hidden_size],
                dtype=self.dst_torch_dtype,
                device=device,
            )

            self.input_tensor_size = sum(
                calc_tensor_size(info) for info in self.input_tensor_info.values()
            )
            self.output_tensor_size = sum(
                calc_tensor_size(info) for info in self.output_tensor_info.values()
            )
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            self.read_bytes = self.input_tensor_size
            self.write_bytes = self.output_tensor_size
            self.io_bytes = self.read_bytes + self.write_bytes

            self.calc_flops = (
                2 * self.dispatch_tokens * self.hidden_size * self.new_hidden_size
            )

            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True,
            )
            self._run_func = self.vendor_impl_run

        def vendor_impl_run(self, tensor_mapping):
            scatter_tokens = tensor_mapping["scatter_tokens"]
            experts_weight = tensor_mapping["experts_weight"]
            segment_offsets = tensor_mapping["segment_offsets"]
            y = tensor_mapping["y"]

            if self.variant == "bf16":
                return flashinfer.grouped_mm_bf16(
                    scatter_tokens,
                    experts_weight,
                    segment_offsets,
                    out=y,
                    out_dtype=self.dst_torch_dtype,
                )

            if self.variant == "fp8_per_tensor":
                return flashinfer.grouped_mm_fp8(
                    scatter_tokens,
                    experts_weight,
                    segment_offsets,
                    alpha=tensor_mapping["alpha"],
                    out=y,
                    out_dtype=self.dst_torch_dtype,
                )

            if self.variant == "fp8_block":
                flashinfer.gemm.group_gemm_fp8_nt_groupwise(
                    scatter_tokens,
                    experts_weight,
                    tensor_mapping["a_scale"],
                    tensor_mapping["b_scale"],
                    segment_offsets,
                    out=y,
                )
                return y

            if self.variant == "mxfp4":
                flashinfer.gemm.group_gemm_mxfp4_nt_groupwise(
                    scatter_tokens,
                    experts_weight,
                    tensor_mapping["a_scale"],
                    tensor_mapping["b_scale"],
                    segment_offsets,
                    out=y,
                )
                return y

            flashinfer.gemm.group_gemm_nvfp4_nt_groupwise(
                scatter_tokens,
                experts_weight,
                tensor_mapping["a_scale"],
                tensor_mapping["b_scale"],
                segment_offsets,
                out=y,
            )
            return y

except Exception:
    pass
