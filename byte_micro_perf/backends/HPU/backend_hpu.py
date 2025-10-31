# Copyright 2023 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company

import os
import json
import random
import logging
import subprocess
import pathlib
import shutil
import time

from datetime import timedelta
import torch.distributed as dist

import torch

from core.backend import Backend

from dataclasses import dataclass

import habana_frameworks.torch as ht


@dataclass
class HPUDeviceProperties:
    total_memory: int


class BackendHPU(Backend):
    def __init__(self):
        super().__init__()

    def get_torch_device_name(self):
        return "hpu"

    def get_device_name(self, index=0):
        return "Gaudi2"

    def get_device_properties(self, index=0):
        # output of torch.hpu.get_device_properties() on Gaudi2:
        # '(sramBaseAddress=1153202979533225984, dramBaseAddress=1153203082662772736, sramSize=50331648, dramSize=102106132480, tpcEnabledMask=16777215, dramEnabled=1, fd=20, device_id=0, device_type=4)'
        dramSize = 102106132480
        return HPUDeviceProperties(dramSize)

    def get_mem_info(self, index=0):
        if hasattr(torch, 'hpu'):
            return torch.hpu.mem_get_info(index)
        else:
            return [self.get_device_properties().total_memory]*2

    def get_device_count(self):
        device_count = int(
            subprocess.check_output(
                "hl-smi -Q module_id -f csv,noheader | wc -l", shell=True, text=True
            )
        )
        return device_count, list(range(device_count))

    def set_device(self, device_index: int):
        try:
            os.environ["HLS_MODULE_ID"] = str(device_index)
        except Exception as e:
            print(str(e), flush=True)

        import habana_frameworks.torch as htorch
        torch.hpu.set_device("hpu:0")

    def get_device(self):
        return 0

    def device_synchronize(self):
        torch.hpu.synchronize()

    def empty_cache(self):
        if hasattr(torch, 'hpu'):
            torch.hpu.synchronize()
        return

    def get_backend_env(self):
        hl_smi_output= subprocess.run(
            ["hl-smi", "-Q", "driver_version", "-f", "csv,noheader"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        __driver_version = hl_smi_output.stdout.splitlines()[0]
        return {
            "torch": torch.__version__,
            "driver": __driver_version,
        }

    def get_dist_module(self):
        return dist

    def get_dist_backend(self):
        return "hccl"

    def _build_profile_filename(self, op_instance, timestamp):
        """根据op实例动态构建有意义的profile文件名"""
        op_name = op_instance.__class__.__name__
        args_dict = op_instance.args_dict
        
        # 清理文件名部分，移除无效字符
        def sanitize_filename_part(text):
            unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            for char in unsafe_chars:
                text = text.replace(char, '_')
            return text
        
        # 参数排序：确保相同的参数组合生成相同的文件名
        sorted_keys = sorted(args_dict.keys())
        
        param_parts = []
        for key in sorted_keys:
            value = args_dict[key]
            
            # 跳过None值
            if value is None:
                continue
                
            # 处理不同类型的值
            if isinstance(value, (str, int, float, bool)):
                safe_key = sanitize_filename_part(str(key))
                safe_value = sanitize_filename_part(str(value))
                param_parts.append(f"{safe_key}_{safe_value}")
            elif isinstance(value, list):
                safe_key = sanitize_filename_part(str(key))
                safe_value = "_".join(sanitize_filename_part(str(v)) for v in value)
                param_parts.append(f"{safe_key}_{safe_value}")
            elif isinstance(value, dict):
                # 对于字典，可以跳过或特殊处理
                continue
        
        # 限制文件名长度，避免过长的文件名
        params_str = "_".join(param_parts)
        if len(params_str) > 200:  # 限制参数部分长度
            params_str = params_str[:200] + "_truncated"
        
        filename = f"{op_name}_{params_str}_{timestamp}.pt.trace.json.gz"
        return filename

    def core_perf(
        self, op_instance, 
        warmup_iterations, prefer_iterations, 
        tensor_list, 
        profiling=False
    ):
        op_group = op_instance.op_group
        group_size = op_instance.group_size

        # if profiling:
        #     raise NotImplementedError("Profiling mode is not supported for HPU")

        if profiling:
            # 创建自定义trace handler
            def custom_trace_handler(prof):
                timestamp = str(time.time_ns())
                filename = self._build_profile_filename(op_instance, timestamp)
                filepath = os.path.join("/workspace/profile", filename)
                prof.export_chrome_trace(filepath)
                logging.info(f"Profile saved to: {filepath}")

            schedule = torch.profiler.schedule(
                wait=0,
                warmup=0,
                active=1,
            )
            prof=torch.profiler.profile(
                schedule=schedule,
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.HPU,
                ],
                on_trace_ready=custom_trace_handler,
                record_shapes=True,
                with_modules=False,
                profile_memory=False,
                with_stack=True,
            )

        ht.core.mark_step()
        for i in range(warmup_iterations):
            index = random.randint(0, len(tensor_list) - 1)
            op_instance.core_run(tensor_list[index])
            ht.core.mark_step()
        start_event = torch.hpu.Event(enable_timing=True)
        end_event = torch.hpu.Event(enable_timing=True)

        if profiling:
            prof.start()

        self.device_synchronize()
        self.op_group_barrier(op_group=op_group, group_size=group_size)
        start_event.record()
        for i in range(prefer_iterations):
            _ = op_instance.core_run(tensor_list[i % len(tensor_list)])
            ht.core.mark_step()
        self.device_synchronize()
        end_event.record()
        end_event.synchronize()

        if profiling:
            prof.stop()

        latency_us = start_event.elapsed_time(end_event) * 1e3 / prefer_iterations
        return latency_us, []
