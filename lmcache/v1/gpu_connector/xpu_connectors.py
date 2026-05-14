# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 LMCache Authors.
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
# Standard
from typing import List, Optional, Union, cast
import os

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType, _lmcache_nvtx_annotate
from lmcache.v1.gpu_connector.gpu_connectors import (
    GPUConnectorInterface,
    VLLMPagedMemGPUConnectorV2,
)
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    _get_head_size_view,
    _split_token2d_kv,
    get_block_size,
    get_device,
    get_dtype,
    get_elements_per_layer,
    get_group_data_ptrs,
    get_head_size,
    get_hidden_dim_size,
    get_num_blocks,
    get_num_heads,
    get_num_layers,
    get_page_buffer_size,
    get_tokens_per_layer,
    is_mla,
    normalize_kv_and_discover_format,
)
from lmcache.v1.memory_management import (
    GPUMemoryAllocator,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)

ALLOWED_FORMAT_TRANSITIONS = {
    (None, MemoryFormat.KV_MLA_FMT),
    (MemoryFormat.KV_MLA_FMT, MemoryFormat.KV_MLA_FMT),
    (MemoryFormat.KV_T2D, MemoryFormat.KV_MLA_FMT),
}


class VLLMPagedMemXPUConnectorV2(VLLMPagedMemGPUConnectorV2):
    """
    The GPU KV cache should be a nested tuple of K and V tensors.
    More specifically, we have:
    - GPUTensor = Tuple[KVLayer, ...]
    - KVLayer = Tuple[Tensor, Tensor]
    - Tensor: [num_blocks, block_size, num_heads, head_size]

    It will produce / consume memory object with KV_2LTD format
    """

    def __init__(
        self,
        use_gpu: bool = False,
        **kwargs,
    ):
        self._attributes_initialized = False
        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.use_gpu = use_gpu

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMPagedMemXPUConnectorV2":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
            layout_hints: Optional hints about KV cache layout from the
                serving engine.

        Returns:
            A new instance of VLLMPagedMemXPUConnectorV2.
        """
        return cls(
            use_gpu=use_gpu,
        )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)

        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        slices = slot_mapping[start:end]
        self._initialize_attributes(self.kvcaches)
        self._validate_memory_format(memory_obj)

        if self.use_mla:
            tmp = memory_obj.tensor[0].to(slot_mapping.device)
            total_blocks = self.num_blocks * self.block_size
            for i, kvcache in enumerate(self.kvcaches):
                kvcache.view(total_blocks, self.head_size).index_copy_(
                    0, slices, tmp[i]
                )
        else:
            tmp_k = memory_obj.tensor[0].to(slot_mapping.device)
            tmp_v = memory_obj.tensor[1].to(slot_mapping.device)
            total_blocks = self.num_blocks * self.block_size
            d = self.num_heads * self.head_size
            for i, (kcache, vcache) in enumerate(self.kvcaches):
                kcache.view(total_blocks, d).index_copy_(0, slices, tmp_k[i])
                vcache.view(total_blocks, d).index_copy_(0, slices, tmp_v[i])

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_MLA_FMT
        if use_mla is True.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        slices = slot_mapping[start:end]
        self._initialize_attributes(self.kvcaches)
        self._validate_memory_format(memory_obj)

        if self.use_mla:
            total_blocks = self.num_blocks * self.block_size
            tmp = torch.stack(
                [
                    kvcache.view(total_blocks, self.head_size).index_select(0, slices)
                    for kvcache in self.kvcaches
                ]
            )
        else:
            total_blocks = self.num_blocks * self.block_size
            d = self.num_heads * self.head_size
            tmp_k = torch.stack(
                [
                    kvcache[0].view(total_blocks, d).index_select(0, slices)
                    for kvcache in self.kvcaches
                ]
            )
            tmp_v = torch.stack(
                [
                    kvcache[1].view(total_blocks, d).index_select(0, slices)
                    for kvcache in self.kvcaches
                ]
            )
            tmp = torch.stack([tmp_k, tmp_v])
        memory_obj.tensor.copy_(tmp, non_blocking=True)

        if not memory_obj.tensor.is_xpu:
            # Force a synchronize if the target buffer is NOT XPU device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            torch.xpu.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get the shape of the data given the number of tokens.

        Args:
            num_tokens: The number of tokens in the data.

        Returns:
            The shape of the KV cache data.

        Raises:
            RuntimeError: If attributes have not been initialized yet
                (i.e., no kv_caches have been seen).
        """
        if not self._attributes_initialized:
            raise RuntimeError(
                "Cannot determine shape before attributes are initialized. "
                "Call to_gpu or from_gpu first so that _initialize_attributes "
                "can discover the KV cache layout."
            )
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])

    def _validate_memory_format(self, memory_obj: MemoryObj) -> None:
        """Validate that the memory object has the expected format.

        Args:
            memory_obj: The memory object to validate.

        Raises:
            ValueError: If the memory format does not match the expected
                format based on whether MLA is in use.
        """
        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemXPUConnectorV2"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemXPUConnectorV2"
                )

    def _initialize_attributes(self, kv_caches: List[torch.Tensor]):
        """Initialize attributes from the kv_caches using utils functions.

        Uses format discovery and utility functions from utils.py to
        extract all KV cache parameters lazily on first use.

        Args:
            kv_caches: The KV cache tensors from which to discover
                the cache layout and parameters.
        """
        if self._attributes_initialized:
            return

        self.device = kv_caches[0].device
        assert self.device.type == "xpu", "The device should be XPU."

        self.gpu_kv_format, kv_caches = normalize_kv_and_discover_format(
            kv_caches, EngineType.VLLM
        )
        self.num_layers = get_num_layers(kv_caches, self.gpu_kv_format)
        self.num_blocks = get_num_blocks(kv_caches, self.gpu_kv_format)
        self.block_size = get_block_size(kv_caches, self.gpu_kv_format)
        self.page_buffer_size = get_page_buffer_size(kv_caches, self.gpu_kv_format)
        self.hidden_dim_size = get_hidden_dim_size(kv_caches, self.gpu_kv_format)
        self.head_size = get_head_size(kv_caches, self.gpu_kv_format)
        self.use_mla = is_mla(self.gpu_kv_format)
        self.dtype = get_dtype(kv_caches, self.gpu_kv_format)
        self.num_heads = (
            1 if self.use_mla else get_num_heads(kv_caches, self.gpu_kv_format)
        )

        self._attributes_initialized = True
        logger.info(
            "XPU: attributes initialized - format: %s, "
            "num_layers: %d, num_blocks: %d, block_size: %d, "
            "page_buffer_size: %d, hidden_dim_size: %d, head_size: %d, "
            "use_mla: %s, dtype: %s, num_heads: %d",
            self.gpu_kv_format,
            self.num_layers,
            self.num_blocks,
            self.block_size,
            self.page_buffer_size,
            self.hidden_dim_size,
            self.head_size,
            self.use_mla,
            self.dtype,
            self.num_heads,
        )


class VLLMPagedMemLayerwiseXPUConnector(GPUConnectorInterface):
    """
    Layerwise paged KV connector for XPU.

    Implements the *same generator contract* as VLLMPagedMemLayerwiseGPUConnector:
      - batched_to_gpu(...) yields num_layers + 2 times
      - batched_from_gpu(...) yields num_layers + 1 times

    Transfer is implemented with pure torch ops (index_copy_/index_select).
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_xpu: bool = False,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_xpu = use_xpu

        assert "chunk_size" in kwargs, "chunk_size should be provided."
        assert "dtype" in kwargs, "dtype should be provided."
        assert "device" in kwargs, "device should be provided."

        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

        self.kvcaches: Optional[List[torch.Tensor]] = None

        # XPU streams
        self.load_stream = torch.xpu.Stream()
        self.store_stream = torch.xpu.Stream()

        # Optional device staging buffer allocator (same pattern as CUDA class)
        self.gpu_buffer_allocator: Optional[MemoryAllocatorInterface] = None

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_xpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemLayerwiseXPUConnector":
        num_layers = metadata.kv_shape[0]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size
        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_xpu=use_xpu,
            chunk_size=metadata.kv_shape[2],
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
        )

    def _validate_format_transition(self, mem, target_fmt):
        current_fmt = mem.metadata.fmt

        if (current_fmt, target_fmt) not in ALLOWED_FORMAT_TRANSITIONS:
            raise ValueError(
                f"Invalid KV format transition: {current_fmt} -> {target_fmt}"
            )

    def _lazy_initialize_buffer(self, kv_caches: List[torch.Tensor]) -> None:
        # Buffer allocator only needed when use_xpu=True (device staging)
        if self.use_xpu and self.gpu_buffer_allocator is None:
            # First Party
            from lmcache.v1.memory_management import XPUMemoryAllocator

            # Derive size from first layer KV tensor
            layer0 = kv_caches[0]
            derived_bytes = layer0.numel() * layer0.element_size()

            # Allow override via env variable
            staging_bytes = int(
                os.getenv("LMCACHE_GPU_STAGING_BUFFER_BYTES", derived_bytes)
            )

            logger.info(
                "Initializing staging buffer (derived=%d bytes, final=%d bytes)",
                derived_bytes,
                staging_bytes,
            )

            self.gpu_buffer_allocator = XPUMemoryAllocator(
                size=staging_bytes,
                device=self.device,
            )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError("Layerwise uses batched_to_gpu(generator).")

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError("Layerwise uses batched_from_gpu(generator).")

    def _batched_to_gpu_gen(self, starts: List[int], ends: List[int], **kwargs):
        """
        Generator: CPU token2d -> (optional XPU staging) -> XPU paged KV (per layer).
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        def _ensure_xpu(t: torch.Tensor) -> torch.Tensor:
            # Handle both torch.device('xpu:0') and string devices consistently.
            if t is None:
                return t
            if t.device != self.device:
                # non_blocking is fine; will be blocking
                # if underlying memory isn't pinned
                return t.to(self.device, non_blocking=True)
            return t

        # Build a single contiguous mapping in the SAME order we will pack chunks.
        slot_mapping_chunks = [
            slot_mapping[s:e] for s, e in zip(starts, ends, strict=False)
        ]
        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        # Move mapping ONCE to device (fixes multiple small H2D copies).
        slot_mapping_full = _ensure_xpu(slot_mapping_full)

        num_tokens = int(slot_mapping_full.numel())
        if num_tokens <= 0:
            for _ in range(self.num_layers):
                _ = yield
            yield
            if sync:
                torch.xpu.current_stream().wait_stream(self.load_stream)
            yield
            return

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_xpu:
            # First Party
            from lmcache.v1.memory_management import MemoryFormat

            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            requested_bytes = (
                int(buffer_shape.numel())
                * torch.empty((), dtype=self.dtype).element_size()
            )
            allocator_tensor = getattr(self.gpu_buffer_allocator, "tensor", None)
            capacity_bytes: Optional[int] = None
            if isinstance(allocator_tensor, torch.Tensor):
                capacity_bytes = int(
                    allocator_tensor.numel() * allocator_tensor.element_size()
                )
            allocator_backend = getattr(self.gpu_buffer_allocator, "allocator", None)
            allocated_bytes = getattr(allocator_backend, "total_allocated_size", None)
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            if tmp_gpu_buffer_obj is None or tmp_gpu_buffer_obj.tensor is None:
                raise RuntimeError(
                    "Failed to allocate XPU staging buffer for batched_to_gpu: "
                    f"requested_bytes={requested_bytes}, "
                    f"capacity_bytes={capacity_bytes}, "
                    f"allocated_bytes={allocated_bytes}, "
                    f"allocator_type={type(self.gpu_buffer_allocator).__name__}, "
                    f"allocator_tensor_device="
                    f"{getattr(allocator_tensor, 'device', None)}"
                )

        current_stream = torch.xpu.current_stream()

        try:
            for layer_id in range(self.num_layers):
                memory_objs_layer = yield  # List[MemoryObj] for this layer

                if sync:
                    current_stream.wait_stream(self.load_stream)

                with torch.xpu.stream(self.load_stream):
                    dst_layer = self.kvcaches[layer_id]
                    if self.use_mla:
                        dst_flat = cast(
                            torch.Tensor,
                            _get_head_size_view(dst_layer, use_mla=True),
                        )
                    else:
                        dst_k_flat, dst_v_flat = _get_head_size_view(  # type: ignore[misc]
                            dst_layer, use_mla=False
                        )

                    cursor = 0

                    if self.use_xpu:
                        assert tmp_gpu_buffer_obj is not None
                        staged = tmp_gpu_buffer_obj.tensor
                        assert staged is not None

                        for s, e, mem in zip(
                            starts, ends, memory_objs_layer, strict=False
                        ):
                            assert mem.tensor is not None
                            n = int(e - s)
                            if n <= 0:
                                continue

                            src = _ensure_xpu(mem.tensor)

                            staged[cursor : cursor + n].copy_(src, non_blocking=True)
                            cursor += n

                        sl = slot_mapping_full  # already intended to be on device
                        sl = _ensure_xpu(sl)

                        if self.use_mla:
                            staged_xpu = _ensure_xpu(staged)
                            if staged_xpu.dim() == 2:
                                dst_flat.index_copy_(0, sl, staged_xpu)
                            elif staged_xpu.dim() == 3 and staged_xpu.shape[0] == 1:
                                dst_flat.index_copy_(0, sl, staged_xpu[0])
                            else:
                                raise ValueError(
                                    f"Unexpected MLA staged tensor: {staged_xpu.shape}"
                                )
                        else:
                            k_tok, v_tok = _split_token2d_kv(staged)

                            # Make sure k_tok/v_tok are on XPU before index_copy_.
                            k_tok = _ensure_xpu(k_tok)
                            v_tok = _ensure_xpu(v_tok)

                            # Keep your reshape logic as-is (only triggers when needed)
                            if (
                                k_tok.dim() == 2
                                and dst_k_flat.dim() == 3
                                and k_tok.shape[1]
                                == dst_k_flat.shape[1] * dst_k_flat.shape[2]
                            ):
                                k_tok = k_tok.reshape(
                                    k_tok.shape[0],
                                    dst_k_flat.shape[1],
                                    dst_k_flat.shape[2],
                                )
                            if (
                                v_tok.dim() == 2
                                and dst_v_flat.dim() == 3
                                and v_tok.shape[1]
                                == dst_v_flat.shape[1] * dst_v_flat.shape[2]
                            ):
                                v_tok = v_tok.reshape(
                                    v_tok.shape[0],
                                    dst_v_flat.shape[1],
                                    dst_v_flat.shape[2],
                                )

                            dst_k_flat.index_copy_(0, sl, k_tok)
                            dst_v_flat.index_copy_(0, sl, v_tok)

                    else:
                        for s, e, mem in zip(
                            starts, ends, memory_objs_layer, strict=False
                        ):
                            assert mem.tensor is not None
                            n = int(e - s)
                            if n <= 0:
                                continue

                            src = _ensure_xpu(mem.tensor)
                            sl = slot_mapping_full[cursor : cursor + n]
                            sl = _ensure_xpu(sl)
                            cursor += n

                            if self.use_mla:
                                if src.dim() == 2:
                                    dst_flat.index_copy_(0, sl, src)
                                elif src.dim() == 3 and src.shape[0] == 1:
                                    dst_flat.index_copy_(0, sl, src[0])
                                else:
                                    raise ValueError(
                                        f"Unexpected MLA token tensor: {src.shape}"
                                    )
                            else:
                                k_tok, v_tok = _split_token2d_kv(src)
                                k_tok = _ensure_xpu(k_tok)
                                v_tok = _ensure_xpu(v_tok)

                                if (
                                    k_tok.dim() == 2
                                    and dst_k_flat.dim() == 3
                                    and k_tok.shape[1]
                                    == dst_k_flat.shape[1] * dst_k_flat.shape[2]
                                ):
                                    k_tok = k_tok.reshape(
                                        k_tok.shape[0],
                                        dst_k_flat.shape[1],
                                        dst_k_flat.shape[2],
                                    )
                                if (
                                    v_tok.dim() == 2
                                    and dst_v_flat.dim() == 3
                                    and v_tok.shape[1]
                                    == dst_v_flat.shape[1] * dst_v_flat.shape[2]
                                ):
                                    v_tok = v_tok.reshape(
                                        v_tok.shape[0],
                                        dst_v_flat.shape[1],
                                        dst_v_flat.shape[2],
                                    )

                                dst_k_flat.index_copy_(0, sl, k_tok)
                                dst_v_flat.index_copy_(0, sl, v_tok)

            yield

            if sync:
                current_stream.wait_stream(self.load_stream)
        finally:
            if tmp_gpu_buffer_obj is not None:
                tmp_gpu_buffer_obj.ref_count_down()

        yield

    def batched_from_gpu(  # type: ignore[override]
        self,
        memory_objs: List[List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        Generator: XPU paged KV -> (optional XPU staging) -> CPU token2d (per layer).
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        current_stream = torch.xpu.current_stream()

        # ---- helpers (keep local to minimize file-wide changes) ----
        def _flatten_last2_if_needed(
            src: torch.Tensor, dst: torch.Tensor
        ) -> torch.Tensor:
            """
            Make src match dst for the common KV layouts:
            - src: [..., H, HS] -> dst: [..., H*HS]
            - or already matches
            """
            if src.shape == dst.shape:
                return src

            # dst has one less trailing dim: [..., D] where D=H*HS
            if src.dim() == dst.dim() + 1:
                # e.g., src [..., 8, 128] -> dst [..., 1024]
                if dst.shape == (*src.shape[:-2], src.shape[-2] * src.shape[-1]):
                    return src.reshape(*src.shape[:-2], -1)

            # same ndim but dst last dim is flattened (dst ends with D)
            if src.dim() == dst.dim():
                if (
                    dst.shape[:-1] == src.shape[:-2]
                    and dst.shape[-1] == src.shape[-2] * src.shape[-1]
                ):
                    return src.reshape(*src.shape[:-2], -1)

            return src  # caller will error if still incompatible

        def _copy_kv_into_mem(
            mem_tensor: torch.Tensor, k_src: torch.Tensor, v_src: torch.Tensor
        ) -> None:
            """
            Copy K/V into mem.tensor supporting:
            - [2, ..., D]  (K in 0, V in 1)
            - [..., 2, D]  (K in [:,0,:], V in [:,1,:])
            - [2, ..., H, HS] or [..., 2, H, HS] similarly
            """
            if mem_tensor.dim() < 3:
                raise ValueError(
                    f"Unexpected output token2d layout: {mem_tensor.shape}"
                )

            # Case A: mem is [2, ...]
            if mem_tensor.shape[0] == 2:
                k_dst = mem_tensor[0]
                v_dst = mem_tensor[1]
                k_src2 = _flatten_last2_if_needed(k_src, k_dst)
                v_src2 = _flatten_last2_if_needed(v_src, v_dst)
                if k_src2.shape != k_dst.shape or v_src2.shape != v_dst.shape:
                    raise ValueError(
                        f"KV shape mismatch after reshape: "
                        f"k src {k_src.shape}->{k_src2.shape} vs dst {k_dst.shape}; "
                        f"v src {v_src.shape}->{v_src2.shape} vs dst {v_dst.shape}"
                    )
                k_dst.copy_(k_src2.to(k_dst.device), non_blocking=True)
                v_dst.copy_(v_src2.to(v_dst.device), non_blocking=True)
                return

            # Case B: mem is [..., 2, ...]
            if mem_tensor.shape[1] == 2:
                k_dst = mem_tensor[:, 0, ...]
                v_dst = mem_tensor[:, 1, ...]
                k_src2 = _flatten_last2_if_needed(k_src, k_dst)
                v_src2 = _flatten_last2_if_needed(v_src, v_dst)
                if k_src2.shape != k_dst.shape or v_src2.shape != v_dst.shape:
                    raise ValueError(
                        f"KV shape mismatch after reshape: "
                        f"k src {k_src.shape}->{k_src2.shape} vs dst {k_dst.shape}; "
                        f"v src {v_src.shape}->{v_src2.shape} vs dst {v_dst.shape}"
                    )
                k_dst.copy_(k_src2.to(k_dst.device), non_blocking=True)
                v_dst.copy_(v_src2.to(v_dst.device), non_blocking=True)
                return

            raise ValueError(f"Unexpected output token2d layout: {mem_tensor.shape}")

        slot_mapping_on_device = slot_mapping.to(self.device)

        # Precompute “full” mapping for batched gather
        # NOTE: this assumes starts/ends partition slot_mapping contiguously.
        # If not contiguous, concatenation is still correct.
        slot_mapping_full = torch.cat(
            [slot_mapping_on_device[s:e] for s, e in zip(starts, ends, strict=False)],
            dim=0,
        )
        total_tokens = int(slot_mapping_full.numel())

        # Optional staging buffer (will be USED when self.use_xpu=True)
        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_xpu:
            # First Party
            from lmcache.v1.memory_management import MemoryFormat

            # buffer shape uses existing helper; must match how allocator expects KV_T2D
            buffer_shape = self.get_shape(total_tokens)
            assert self.gpu_buffer_allocator is not None
            requested_bytes = (
                int(buffer_shape.numel())
                * torch.empty((), dtype=self.dtype).element_size()
            )
            allocator_tensor = getattr(self.gpu_buffer_allocator, "tensor", None)
            capacity_bytes: Optional[int] = None
            if isinstance(allocator_tensor, torch.Tensor):
                capacity_bytes = int(
                    allocator_tensor.numel() * allocator_tensor.element_size()
                )
            allocator_backend = getattr(self.gpu_buffer_allocator, "allocator", None)
            allocated_bytes = getattr(allocator_backend, "total_allocated_size", None)
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            if tmp_gpu_buffer_obj is None or tmp_gpu_buffer_obj.tensor is None:
                raise RuntimeError(
                    "Failed to allocate XPU staging buffer for batched_from_gpu: "
                    f"requested_bytes={requested_bytes}, "
                    f"capacity_bytes={capacity_bytes}, "
                    f"allocated_bytes={allocated_bytes}, "
                    f"allocator_type={type(self.gpu_buffer_allocator).__name__}, "
                    f"allocator_tensor_device="
                    f"{getattr(allocator_tensor, 'device', None)}"
                )
            tmp = tmp_gpu_buffer_obj.tensor  # staging tensor on device

        try:
            for layer_id in range(self.num_layers):
                mem_layer = memory_objs[layer_id]

                with torch.xpu.stream(self.store_stream):
                    self.store_stream.wait_stream(current_stream)

                    src_layer = self.kvcaches[layer_id]

                    if self.use_mla:
                        src_flat = cast(
                            torch.Tensor,
                            _get_head_size_view(src_layer, use_mla=True),
                        )

                        if self.use_xpu:
                            gathered_full = src_flat.index_select(0, slot_mapping_full)
                            # Write into tmp if possible, else fallback to per-chunk
                            tmp_src = (
                                _flatten_last2_if_needed(gathered_full, tmp)
                                if "tmp" in locals()
                                else gathered_full
                            )
                            if "tmp" in locals() and tmp_src.shape == tmp.shape:
                                tmp.copy_(tmp_src, non_blocking=True)
                                off = 0
                                for s, e, mem in zip(
                                    starts, ends, mem_layer, strict=False
                                ):
                                    assert mem.tensor is not None
                                    n = e - s
                                    chunk = tmp[off : off + n]
                                    off += n
                                    mem.tensor.copy_(
                                        chunk.to(mem.tensor.device), non_blocking=True
                                    )
                            else:
                                for s, e, mem in zip(
                                    starts, ends, mem_layer, strict=False
                                ):
                                    assert mem.tensor is not None
                                    sl = slot_mapping_on_device[s:e]
                                    gathered = src_flat.index_select(0, sl)
                                    mem.tensor.copy_(
                                        gathered.to(mem.tensor.device),
                                        non_blocking=True,
                                    )
                        else:
                            for s, e, mem in zip(starts, ends, mem_layer, strict=False):
                                assert mem.tensor is not None
                                sl = slot_mapping_on_device[s:e]
                                gathered = src_flat.index_select(0, sl)
                                mem.tensor.copy_(
                                    gathered.to(mem.tensor.device), non_blocking=True
                                )

                        # Keep memory format metadata consistent for downstream checks.
                        target_fmt = MemoryFormat.KV_MLA_FMT
                        for mem in mem_layer:
                            self._validate_format_transition(mem, target_fmt)
                            mem.metadata.fmt = target_fmt

                    else:
                        src_k_flat, src_v_flat = _get_head_size_view(
                            src_layer, use_mla=False
                        )

                        if self.use_xpu:
                            k_full = src_k_flat.index_select(0, slot_mapping_full)
                            v_full = src_v_flat.index_select(0, slot_mapping_full)

                            # Slice from staging. If tmp exists and can hold
                            # the layout, use it; otherwise slice k/v directly.
                            off = 0
                            for s, e, mem in zip(starts, ends, mem_layer, strict=False):
                                assert mem.tensor is not None
                                n = e - s

                                k_chunk = k_full[off : off + n]
                                v_chunk = v_full[off : off + n]
                                off += n

                                _copy_kv_into_mem(mem.tensor, k_chunk, v_chunk)

                        else:
                            # per-chunk gather (original behavior);
                            # avoids per-iteration H2D slot_mapping transfers
                            for s, e, mem in zip(starts, ends, mem_layer, strict=False):
                                assert mem.tensor is not None
                                sl = slot_mapping_on_device[s:e]
                                k = src_k_flat.index_select(0, sl)
                                v = src_v_flat.index_select(0, sl)
                                _copy_kv_into_mem(mem.tensor, k, v)

                if sync:
                    self.store_stream.synchronize()
                yield
        finally:
            if tmp_gpu_buffer_obj is not None:
                tmp_gpu_buffer_obj.ref_count_down()

        yield

    def batched_to_gpu(
        self,
        memory_objs: Union[
            List[List[MemoryObj]], List[MemoryObj], List[int], None
        ] = None,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
        **kwargs,
    ):
        return self._batched_to_gpu_gen(starts=starts or [], ends=ends or [], **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        if self.use_mla:
            return torch.Size([num_tokens, self.hidden_dim_size])
        return torch.Size([num_tokens, 2, self.hidden_dim_size])


# SGLang XPU Connectors
class SGLangXPUConnector(GPUConnectorInterface):
    """
    SGLang GPU KV connector for XPU devices.

    Extends the base connector interface with XPU-specific device handling
    and synchronization using torch.xpu instead of torch.cuda.
    """

    def __init__(
        self, hidden_dim_size: int, num_layers: int, use_gpu: bool = False, **kwargs
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

        self.num_kv_cache = num_layers if self.use_mla else num_layers * 2
        self.kv_cache_pointers = torch.empty(
            self.num_kv_cache, dtype=torch.int64, device="cpu"
        )

        if use_gpu:
            assert "chunk_size" in kwargs, (
                "chunk_size should be provided to create a GPU buffer."
            )
            assert "device" in kwargs, (
                "device should be provided to create a GPU buffer."
            )
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(
                shape, dtype=kwargs["dtype"], device=kwargs["device"]
            )
            logger.info(f"GPU buffer: {self.gpu_buffer.shape}")

    def _initialize_pointers(self, kv_caches) -> torch.Tensor:
        self.gpu_kv_format, kv_caches = normalize_kv_and_discover_format(
            kv_caches, EngineType.SGLANG
        )
        num_layers = get_num_layers(kv_caches, self.gpu_kv_format)
        # SGLang registers every layer as one group; pass all indices in order.
        ptrs = get_group_data_ptrs(
            kv_caches, self.gpu_kv_format, list(range(num_layers))
        )
        assert len(ptrs) == self.num_kv_cache, (
            f"Expected {self.num_kv_cache} KV cache pointers, got {len(ptrs)}"
        )
        self.kv_cache_pointers.numpy()[:] = ptrs

        device = get_device(kv_caches)
        assert device.type == "xpu", f"The device should be XPU, got {device.type}."
        idx = device.index
        if idx not in self.kv_cache_pointers_on_gpu:
            self.kv_cache_pointers_on_gpu[idx] = torch.empty(
                self.num_kv_cache, dtype=torch.int64, device=device
            )
        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)

        self.page_buffer_size = get_page_buffer_size(kv_caches, self.gpu_kv_format)
        return self.kv_cache_pointers_on_gpu[idx]

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Store the data in the memory object into a GPU buffer."""
        assert memory_obj.tensor is not None

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    f" order to be processed by {self.__class__.__name__}"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    f" order to be processed by {self.__class__.__name__}"
                )

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        offset = kwargs.get("offset", 0)

        kvcaches = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(kvcaches)
        lmc_ops.multi_layer_kv_transfer_unilateral(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping[start - offset : end - offset],
            get_device(kvcaches),
            self.page_buffer_size,
            lmc_ops.TransferDirection.H2D,
            self.gpu_kv_format,
        )

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Retrieve the data from a GPU buffer into the memory object."""
        assert memory_obj.tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(kvcaches)

        if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
            lmc_ops.multi_layer_kv_transfer_unilateral(
                memory_obj.tensor,
                kv_cache_pointers,
                slot_mapping[start:end],
                get_device(kvcaches),
                self.page_buffer_size,
                lmc_ops.TransferDirection.D2H,
                self.gpu_kv_format,
            )
        else:
            # kvcaches -> gpu_buffer -> memobj
            assert self.gpu_buffer.device == get_device(kvcaches)
            tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
            lmc_ops.multi_layer_kv_transfer_unilateral(
                tmp_gpu_buffer,
                kv_cache_pointers,
                slot_mapping[start:end],
                get_device(kvcaches),
                self.page_buffer_size,
                lmc_ops.TransferDirection.D2H,
                self.gpu_kv_format,
            )
            memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        if not memory_obj.tensor.is_xpu:
            # Force a synchronize if the target buffer is NOT XPU device
            torch.xpu.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, self.num_layers, num_tokens, self.hidden_dim_size])

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)


class SGLangLayerwiseXPUConnector(GPUConnectorInterface):
    """
    SGLang layerwise GPU KV connector for XPU devices.

    Supports layerwise loading/storing with XPU-specific device handling.
    """

    def __init__(
        self, hidden_dim_size: int, num_layers: int, use_gpu: bool = False, **kwargs
    ):
        assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
        self.dtype = kwargs["dtype"]
        assert "device" in kwargs, "device should be provided to create a GPU buffer."
        self.device = kwargs["device"]

        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

        self.num_kv_cache = num_layers if self.use_mla else num_layers * 2
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()
        self.kv_cache_pointers = torch.empty(
            self.num_kv_cache, dtype=torch.int64, device="cpu"
        )
        self.use_gpu = use_gpu
        self.gpu_buffer_allocator: Optional[GPUMemoryAllocator] = None

    def _initialize_pointers(self, kv_caches) -> torch.Tensor:
        self.gpu_kv_format, kv_caches = normalize_kv_and_discover_format(
            kv_caches, EngineType.SGLANG
        )
        num_layers = get_num_layers(kv_caches, self.gpu_kv_format)
        # SGLang registers every layer as one group; pass all indices in order.
        ptrs = get_group_data_ptrs(
            kv_caches, self.gpu_kv_format, list(range(num_layers))
        )
        assert len(ptrs) == self.num_kv_cache, (
            f"Expected {self.num_kv_cache} KV cache pointers, got {len(ptrs)}"
        )
        self.kv_cache_pointers.numpy()[:] = ptrs

        device = get_device(kv_caches)
        assert device.type == "xpu", f"The device should be XPU, got {device.type}."
        idx = device.index
        if idx not in self.kv_cache_pointers_on_gpu:
            self.kv_cache_pointers_on_gpu[idx] = torch.empty(
                self.num_kv_cache, dtype=torch.int64, device=device
            )
        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)

        self.page_buffer_size = get_page_buffer_size(kv_caches, self.gpu_kv_format)
        return self.kv_cache_pointers_on_gpu[idx]

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError(
            "SGLangLayerwiseXPUConnector uses the batched_to_gpu generator."
        )

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError(
            "SGLangLayerwiseXPUConnector uses the batched_from_gpu generator."
        )

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        Generator: CPU token2d -> XPU paged KV (per layer).

        Protocol (called by cache_engine):
          next()               -- initial call, setup and pause at first layer yield
          send(mem_objs_layer) -- per-layer: receive memory objects, do H2D, advance
          next()               -- final call, synchronize

        In total yields num_layers + 1 times.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield

            k_layer = self.kvcaches[0][layer_id]  # [total_slots, head_num, head_size]
            v_layer = self.kvcaches[1][layer_id]

            total_slots = k_layer.shape[0]
            d = k_layer.shape[1] * k_layer.shape[2]

            for s, e, mem in zip(starts, ends, memory_objs_layer, strict=False):
                assert mem.tensor is not None
                sl = slot_mapping[s:e].to(self.device)
                k_tok, v_tok = _split_token2d_kv(mem.tensor.to(self.device))
                k_layer.view(total_slots, d).index_copy_(
                    0, sl, k_tok.reshape(e - s, d)
                )
                v_layer.view(total_slots, d).index_copy_(
                    0, sl, v_tok.reshape(e - s, d)
                )

            logger.debug(f"Finished loading layer {layer_id}")

        yield  # final synchronization yield

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: List[List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        Generator: XPU paged KV -> CPU token2d (per layer).

        Protocol (called by cache_engine):
          next()   -- initial call, setup and pause
          next()   -- per-layer: trigger D2H for that layer, pause
                      (caller then batched_puts memory_objs[layer_id])

        In total yields num_layers + 1 times.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")
        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        yield  # initial yield after first next()

        for layer_id in range(self.num_layers):
            mem_layer = memory_objs[layer_id]

            k_layer = self.kvcaches[0][layer_id]  # [total_slots, head_num, head_size]
            v_layer = self.kvcaches[1][layer_id]

            total_slots = k_layer.shape[0]
            d = k_layer.shape[1] * k_layer.shape[2]

            offset = 0
            for s, e, mem in zip(starts, ends, mem_layer, strict=False):
                assert mem.tensor is not None
                sl = slot_mapping[s:e].to(self.device)
                n = e - s

                k_chunk = k_layer.view(total_slots, d).index_select(0, sl)  # [n, d]
                v_chunk = v_layer.view(total_slots, d).index_select(0, sl)  # [n, d]

                # Write back into the KV_T2D memory object
                if mem.tensor.shape[0] == 2:  # [2, T, D]
                    mem.tensor[0].copy_(
                        k_chunk.reshape(n, -1).to(mem.tensor.device), non_blocking=True
                    )
                    mem.tensor[1].copy_(
                        v_chunk.reshape(n, -1).to(mem.tensor.device), non_blocking=True
                    )
                elif mem.tensor.shape[1] == 2:  # [T, 2, D]
                    mem.tensor[:, 0, :].copy_(
                        k_chunk.reshape(n, -1).to(mem.tensor.device), non_blocking=True
                    )
                    mem.tensor[:, 1, :].copy_(
                        v_chunk.reshape(n, -1).to(mem.tensor.device), non_blocking=True
                    )
                else:
                    raise ValueError(
                        f"Unexpected memory object tensor format: {mem.tensor.shape}"
                    )
                offset += n

            yield  # yield after each layer, caller batched_puts memory_objs[layer_id]
            logger.debug(f"Finished offloading layer {layer_id}")

        yield  # final yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, self.num_layers, num_tokens, self.hidden_dim_size])
