# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
from collections.abc import Callable
from typing import TypedDict

import torch
import zmq


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


class WorkerExtension:
    """
    The class for vLLM's worker to inherit from.
    By defining an extension class, the code can work no matter what is
    the underlying worker class.

    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """

    def init_weight_update_group(
        self, master_address, master_port, rank_offset, world_size
    ):
        from vllm.distributed.parallel_state import get_world_group

        rank = get_world_group().rank + rank_offset
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

    def update_weight(self, name, dtype_name, shape):
        dtype = getattr(torch, dtype_name)
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(
            weight, src=0, stream=torch.cuda.current_stream()
        )

        try:
            self.model_runner.model.load_weights(weights=[(name, weight)])
        except TypeError as e:
            if "unexpected keyword argument" not in str(e):
                raise
            if not getattr(self, "_fallback_warned", False):
                print(f"[WorkerExtension] load_weights TypeError workaround activated: {e}")
                self._fallback_warned = True
            
            # Prepare state_dict cache
            if getattr(self, "_param_cache", None) is None:
                self._param_cache = dict(self.model_runner.model.named_parameters())
            state_dict = self._param_cache

            mapped_name = name
            if hasattr(self.model_runner.model, "hf_to_vllm_mapper"):
                mapper = self.model_runner.model.hf_to_vllm_mapper
                if hasattr(mapper, "_mappings"):
                    import re
                    for hf_pattern, vllm_pattern in mapper._mappings.items():
                        if isinstance(hf_pattern, re.Pattern):
                            match = hf_pattern.match(name)
                            if match:
                                if callable(vllm_pattern):
                                    res = vllm_pattern(match)
                                    if res:
                                        mapped_name = res
                                else:
                                    mapped_name = hf_pattern.sub(vllm_pattern, name)
                                break
                        elif hf_pattern == name:
                            mapped_name = vllm_pattern
                            break
            
            if mapped_name == name:
                if "gate_up_proj" in name:
                    mapped_name = name.replace("gate_up_proj", "w13_weight")
                elif "down_proj" in name:
                    mapped_name = name.replace("down_proj", "w2_weight")

            mapped_name = str(mapped_name)

            if mapped_name in state_dict:
                state_dict[mapped_name].data.copy_(weight)
            elif mapped_name + ".weight" in state_dict:
                state_dict[mapped_name + ".weight"].data.copy_(weight)
            else:
                matched = False
                for k, param in state_dict.items():
                    if k.endswith(mapped_name) or mapped_name.endswith(k) or k.replace(".weight", "") == mapped_name:
                        param.data.copy_(weight)
                        state_dict[mapped_name] = param
                        matched = True
                        break
                if not matched:
                    raise KeyError(f"Failed to find parameter {mapped_name} (original: {name}) for fallback. Available: {list(state_dict)[:5]}...")

        del weight

    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(p, torch.zeros_like(p))
        return weights_updated


def rebuild_ipc(
    handle: tuple[Callable, tuple], device_id: int | None = None
) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer


class FlattenedTensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    # specify the start offset of this tensor in shared ipc_buffer tensor
    offset: int


class ColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class.

    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """

    def update_weights_from_ipc(self, zmq_handles: dict[str, str]):
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        assert self.device is not None
        if not hasattr(self, "_zmq_ctx") or self._zmq_ctx is None:
            self._zmq_ctx = zmq.Context()
        socket = self._zmq_ctx.socket(zmq.REP)
        socket.connect(zmq_handles[self.report_device_id()])
        buffer: torch.Tensor | None = None
        while True:
            payload: tuple[Callable, tuple] | list[FlattenedTensorMetadata] | None = (
                socket.recv_pyobj()
            )
            if payload is None:
                # means the update is done
                process_weights_after_loading(
                    self.model_runner.model, self.model_config, self.device
                )
                torch.cuda.synchronize()
                socket.send(b"")
                break
            if isinstance(payload, tuple):
                # an ipc handle that vLLM can use `func, args = handle`
                # and `func(*args)` to rebuild GPU tensor.
                buffer = rebuild_ipc(payload, self.device.index)
                assert buffer.dtype == torch.uint8
                socket.send(b"")
                continue
            assert isinstance(payload, list)
            assert buffer is not None
            weights = []
            for item in payload:
                shape = item["shape"]
                if isinstance(shape, (list, tuple)):
                    shape = torch.Size(shape)
                assert isinstance(shape, torch.Size)
                dtype, offset = item["dtype"], item["offset"]
                size = dtype.itemsize * shape.numel()
                tensor = buffer[offset : offset + size].view(dtype=dtype).view(shape)
                weights.append((item["name"], tensor))
            try:
                self.model_runner.model.load_weights(weights=weights)
            except TypeError as e:
                if "unexpected keyword argument" not in str(e):
                    raise
                if not getattr(self, "_fallback_warned", False):
                    print(f"[ColocateWorkerExtension] load_weights TypeError workaround activated for batched weights: {e}")
                    self._fallback_warned = True
                
                if getattr(self, "_param_cache", None) is None:
                    self._param_cache = dict(self.model_runner.model.named_parameters())
                state_dict = self._param_cache

                mapper = getattr(self.model_runner.model, "hf_to_vllm_mapper", None)

                for name, weight_tensor in weights:
                    mapped_name = name
                    if mapper and hasattr(mapper, "_mappings"):
                        import re
                        for hf_pattern, vllm_pattern in mapper._mappings.items():
                            if isinstance(hf_pattern, re.Pattern):
                                match = hf_pattern.match(name)
                                if match:
                                    if callable(vllm_pattern):
                                        res = vllm_pattern(match)
                                        if res:
                                            mapped_name = res
                                    else:
                                        mapped_name = hf_pattern.sub(vllm_pattern, name)
                                    break
                            elif hf_pattern == name:
                                mapped_name = vllm_pattern
                                break
                                
                    if mapped_name == name:
                        if "gate_up_proj" in name:
                            mapped_name = name.replace("gate_up_proj", "w13_weight")
                        elif "down_proj" in name:
                            mapped_name = name.replace("down_proj", "w2_weight")

                    mapped_name = str(mapped_name)

                    if mapped_name in state_dict:
                        state_dict[mapped_name].data.copy_(weight_tensor)
                    elif mapped_name + ".weight" in state_dict:
                        state_dict[mapped_name + ".weight"].data.copy_(weight_tensor)
                    else:
                        matched = False
                        for k, param in state_dict.items():
                            if k.endswith(mapped_name) or mapped_name.endswith(k) or k.replace(".weight", "") == mapped_name:
                                param.data.copy_(weight_tensor)
                                state_dict[mapped_name] = param
                                matched = True
                                break
                        if not matched:
                            raise KeyError(f"Failed to find parameter {mapped_name} (original: {name}) for fallback. Available: {list(state_dict)[:5]}...")

        socket.close()
        del buffer
        gc.collect()
        torch.cuda.empty_cache()

    def report_device_id(self) -> str:
        from vllm.platforms import current_platform

        self.device_uuid = current_platform.get_device_uuid(self.device.index)
        return self.device_uuid

    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(p, torch.zeros_like(p))
        return weights_updated
