"""
vram_monitor.py
GPU VRAM & KV-cache memory monitor with dynamic batch contraction (_room_for_copy).

Mitigates GPU Out-Of-Memory (OOM) during LLM inference by reading free VRAM
via pynvml / nvidia-ml-py and calculating allocation limits with a conservative
_room_for_copy margin reserved for KV-cache block swapping and activation spikes.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Suppress pynvml deprecation warning if present
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    try:
        import pynvml  # type: ignore
        _HAS_NVML = True
    except ImportError:
        _HAS_NVML = False

logger = logging.getLogger(__name__)


@dataclass
class VramInfo:
    """Snapshot of VRAM state."""
    total_bytes: int
    free_bytes: int
    used_bytes: int
    device_name: str
    is_gpu: bool = True

    @property
    def free_mb(self) -> float:
        return self.free_bytes / (1024 * 1024)

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    @property
    def used_mb(self) -> float:
        return self.used_bytes / (1024 * 1024)


@dataclass
class ModelVramConfig:
    """Model architecture configuration for estimating per-request KV-cache & activations."""
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    dtype_bytes: int = 2  # FP16 / BF16 = 2 bytes
    activation_overhead_bytes: int = 64 * 1024 * 1024  # 64 MB intermediate activations

    def estimate_bytes_per_token(self) -> int:
        """KV-cache bytes per token across all transformer layers."""
        # KV cache stores K and V: 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes

    def estimate_bytes_per_request(self, max_seq_len: int = 512) -> int:
        """Total memory consumed by one request (KV-cache + activation margin)."""
        kv_cache_bytes = max_seq_len * self.estimate_bytes_per_token()
        return kv_cache_bytes + self.activation_overhead_bytes


class VramMonitor:
    """
    Monitors GPU memory and dynamically resizes batch sizes using the _room_for_copy algorithm.
    """

    def __init__(
        self,
        device_index: int = 0,
        min_room_for_copy_bytes: int = 512 * 1024 * 1024,  # 512 MB minimum swap margin
        room_for_copy_ratio: float = 0.20,  # Reserve 20% of free VRAM for copy/swap
        mock_mode: bool = False,
        mock_free_bytes: Optional[int] = None,
        mock_total_bytes: int = 8 * 1024 * 1024 * 1024,  # 8 GB
    ) -> None:
        self.device_index = device_index
        self.min_room_for_copy_bytes = min_room_for_copy_bytes
        self.room_for_copy_ratio = room_for_copy_ratio
        self.mock_mode = mock_mode
        self._mock_free_bytes = mock_free_bytes
        self._mock_total_bytes = mock_total_bytes
        self._nvml_initialized = False

        if not self.mock_mode and _HAS_NVML:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except Exception as e:
                logger.warning(f"NVML initialization failed: {e}. Falling back to CPU / mock.")
                self._nvml_initialized = False

    def __del__(self) -> None:
        if self._nvml_initialized and _HAS_NVML:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def get_vram_info(self) -> VramInfo:
        """Query current VRAM usage from GPU or fallback."""
        if self.mock_mode or not self._nvml_initialized:
            free = self._mock_free_bytes if self._mock_free_bytes is not None else self._get_system_ram_free()
            total = self._mock_total_bytes
            used = total - free
            return VramInfo(
                total_bytes=total,
                free_bytes=max(0, free),
                used_bytes=max(0, used),
                device_name="Simulated / CPU Device",
                is_gpu=False,
            )

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return VramInfo(
                total_bytes=mem.total,
                free_bytes=mem.free,
                used_bytes=mem.used,
                device_name=name,
                is_gpu=True,
            )
        except Exception as e:
            logger.error(f"Failed to query GPU memory: {e}")
            free = self._get_system_ram_free()
            return VramInfo(
                total_bytes=self._mock_total_bytes,
                free_bytes=free,
                used_bytes=self._mock_total_bytes - free,
                device_name="Fallback System Memory",
                is_gpu=False,
            )

    def _get_system_ram_free(self) -> int:
        """Determine free system memory via /proc/meminfo or os.sysconf."""
        try:
            if hasattr(os, "sysconf") and "SC_AVPHYS_PAGES" in os.sysconf_names:
                pages = os.sysconf("SC_AVPHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                return pages * page_size
        except Exception:
            pass
        return 2 * 1024 * 1024 * 1024  # Default 2 GB fallback

    def calculate_room_for_copy(self, free_bytes: int) -> int:
        """
        Calculate the reserved _room_for_copy margin.
        Formula: max(min_room_for_copy_bytes, free_bytes * room_for_copy_ratio).
        """
        ratio_room = int(free_bytes * self.room_for_copy_ratio)
        return max(self.min_room_for_copy_bytes, ratio_room)

    def calculate_adaptive_batch_size(
        self,
        target_batch_size: int,
        seq_len: int = 512,
        model_config: Optional[ModelVramConfig] = None,
    ) -> int:
        """
        Calculates safe batch size preventing OOM.
        Formula:
            room_for_copy = max(min_room, free_vram * ratio)
            effective_vram = free_vram - room_for_copy
            max_safe_batch = effective_vram // bytes_per_req
            adaptive_batch = clamp(max_safe_batch, 1, target_batch_size)
        """
        vram = self.get_vram_info()
        room_for_copy = self.calculate_room_for_copy(vram.free_bytes)
        effective_vram = vram.free_bytes - room_for_copy

        if effective_vram <= 0:
            logger.warning(
                f"VRAM critically constrained! Free: {vram.free_mb:.1f}MB, "
                f"Room for copy: {room_for_copy / (1024*1024):.1f}MB. Throttling batch size to 1."
            )
            return 1

        config = model_config or ModelVramConfig()
        bytes_per_req = config.estimate_bytes_per_request(max_seq_len=seq_len)

        max_safe_batch = int(effective_vram // bytes_per_req)

        # Contraction: shrink target_batch_size if it exceeds safe capacity
        contracted_batch_size = max(1, min(target_batch_size, max_safe_batch))

        if contracted_batch_size < target_batch_size:
            logger.info(
                f"Auto-contracting batch size from {target_batch_size} to {contracted_batch_size} "
                f"to respect _room_for_copy margin ({room_for_copy / (1024*1024):.1f}MB)."
            )

        return contracted_batch_size

    def check_oom_risk(
        self,
        batch_size: int,
        seq_len: int = 512,
        model_config: Optional[ModelVramConfig] = None,
    ) -> bool:
        """Check if executing with batch_size carries high risk of OOM."""
        vram = self.get_vram_info()
        room_for_copy = self.calculate_room_for_copy(vram.free_bytes)
        effective_vram = vram.free_bytes - room_for_copy

        config = model_config or ModelVramConfig()
        required_bytes = batch_size * config.estimate_bytes_per_request(max_seq_len=seq_len)

        return required_bytes > effective_vram

    def split_into_adaptive_batches(
        self,
        items: Sequence[Any],
        max_batch_size: int = 32,
        seq_len: int = 512,
        model_config: Optional[ModelVramConfig] = None,
    ) -> List[List[Any]]:
        """
        Partition items into safe, non-OOM batches dynamically adhering
        to the _room_for_copy margin.
        """
        if not items:
            return []

        adaptive_size = self.calculate_adaptive_batch_size(
            target_batch_size=max_batch_size,
            seq_len=seq_len,
            model_config=model_config,
        )

        batches = []
        for i in range(0, len(items), adaptive_size):
            batches.append(list(items[i : i + adaptive_size]))
        return batches
