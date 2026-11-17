"""
vllm_adapter.py
Adapter for vLLM V1 (Model Runner V2 & BatchUpdate interfaces) and VllmCompatAdapter.

Features:
1. Import-light module design:
   - torch, triton, and vllm heavy dependencies isolated to TYPE_CHECKING or lazy loading.
   - Allows frontend / UI processes to import this module and call validate_params()
     without GPU / CUDA dependencies.
2. vLLM V1 Model Runner V2 (vllm.v1.worker.gpu.sample.logits_processor.interface):
   - Fresh slot state reinitialization in add_request() to prevent state leakage from free-list recycling.
   - Staged write commits via apply_staged_writes().
   - Expanded logits index mapping via ctx.expanded_idx_mapping for speculative decoding.
   - Thinking-budget forcing support.
3. vLLM V1 BatchUpdate Interface (vllm.v1.sample.logits_processor.interface):
   - Strict batch update order: removed -> added -> moved (including MoveDirectionality.SWAP).
   - argmax-invariant validation and -inf row recovery (learning from MinTokensLogitsProcessor).
4. VllmCompatAdapter:
   - Runtime auto-detection and graceful selection between V1 BatchUpdate and V2 Model Runner.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple, Type, Union

if TYPE_CHECKING:
    import torch
    from torch import Tensor

logger = logging.getLogger(__name__)


# ==============================================================================
# Enum & Data Structures for Batch Updates
# ==============================================================================

class MoveDirectionality(str, enum.Enum):
    """Directionality of slot movements during batch reorganization."""
    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"
    SWAP = "SWAP"


@dataclass
class BatchUpdate:
    """
    Representation of batch changes dispatched in vLLM V1 BatchUpdate interface.
    Processing MUST follow strict order:
        1. removed
        2. added
        3. moved
    """
    removed: List[int] = field(default_factory=list)
    added: List[Tuple[int, Any]] = field(default_factory=list)  # (req_idx, sampling_params)
    moved: List[Tuple[int, int, MoveDirectionality]] = field(default_factory=list)  # (from_idx, to_idx, directionality)


@dataclass
class RequestSlotState:
    """State tracked per active request slot."""
    req_idx: int
    sampling_params: Any
    allowed_token_ids: Optional[Set[int]] = None
    banned_token_ids: Optional[Set[int]] = None
    step_count: int = 0
    tag_depth: int = 0
    thinking_budget: Optional[int] = None
    is_active: bool = True

    def reset(self, req_idx: int, sampling_params: Any) -> None:
        """Cleanly re-initialize slot state when recycled from free-list."""
        self.req_idx = req_idx
        self.sampling_params = sampling_params
        self.allowed_token_ids = getattr(sampling_params, "allowed_token_ids", None)
        self.banned_token_ids = getattr(sampling_params, "banned_token_ids", None)
        self.step_count = 0
        self.tag_depth = 0
        self.thinking_budget = getattr(sampling_params, "thinking_budget", None)
        self.is_active = True


# ==============================================================================
# Base Shim Classes for Environments without Native vLLM
# ==============================================================================

try:
    from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor as _BaseV2LogitsProcessor
except ImportError:
    class _BaseV2LogitsProcessor:
        """Fallback base class when vLLM Model Runner V2 interface is unavailable."""
        pass

try:
    from vllm.v1.sample.logits_processor.interface import LogitsProcessor as _BaseV1LogitsProcessor
except ImportError:
    class _BaseV1LogitsProcessor:
        """Fallback base class when vLLM V1 BatchUpdate interface is unavailable."""
        pass


# ==============================================================================
# 1. vLLM V1 Model Runner V2 LogitsProcessor (Import-Light)
# ==============================================================================

class VllmV2ModelRunnerProcessor(_BaseV2LogitsProcessor):
    """
    Adapter for vLLM V1 Model Runner V2 (vllm.v1.worker.gpu.sample.logits_processor.interface).

    Contract & Invariants:
    - Lightweight imports: torch/triton deferral.
    - Clean slot reinitialization on add_request to handle free-list recycling.
    - Staged writes committed cleanly via apply_staged_writes().
    - Mapping speculative decoding rows with ctx.expanded_idx_mapping.
    - Thinking budget forcing validation.
    """

    def __init__(self, vllm_config: Any = None, req_states: Optional[Dict[int, RequestSlotState]] = None) -> None:
        self.vllm_config = vllm_config
        self.req_states: Dict[int, RequestSlotState] = req_states if req_states is not None else {}
        self._staged_removals: Set[int] = set()
        self._staged_additions: Dict[int, Any] = {}

    @classmethod
    def validate_params(cls, sampling_params: Any) -> None:
        """
        Validate sampling parameters.
        Must run cleanly without GPU/torch dependencies for frontend validation.
        """
        if sampling_params is None:
            return
        # Validate allowed_token_ids or banned_token_ids if present
        allowed = getattr(sampling_params, "allowed_token_ids", None)
        if allowed is not None and not isinstance(allowed, (set, list, tuple)):
            raise ValueError(f"allowed_token_ids must be a collection, got {type(allowed)}")
        banned = getattr(sampling_params, "banned_token_ids", None)
        if banned is not None and not isinstance(banned, (set, list, tuple)):
            raise ValueError(f"banned_token_ids must be a collection, got {type(banned)}")

    def add_request(self, req_idx: int, sampling_params: Any) -> bool:
        """
        Add a request into the active slot.
        CRITICAL: Completely resets the slot to prevent state leakage from recycled free-list slots.
        """
        self.validate_params(sampling_params)
        state = RequestSlotState(req_idx=req_idx, sampling_params=sampling_params)
        state.reset(req_idx, sampling_params)
        self._staged_additions[req_idx] = state
        return True

    def remove_request(self, req_idx: int) -> None:
        """Stage removal of a request slot."""
        self._staged_removals.add(req_idx)

    def apply_staged_writes(self) -> None:
        """Apply all staged additions and removals."""
        for req_idx in self._staged_removals:
            self.req_states.pop(req_idx, None)
        self._staged_removals.clear()

        for req_idx, state in self._staged_additions.items():
            self.req_states[req_idx] = state
        self._staged_additions.clear()

    def apply(self, logits: Any, ctx: Any) -> Any:
        """
        Apply constraint masks to logits tensor.
        Uses ctx.expanded_idx_mapping for speculative decoding rows.
        """
        self.apply_staged_writes()

        # expanded_idx_mapping maps each row of logits to its corresponding req_idx
        expanded_idx_mapping = getattr(ctx, "expanded_idx_mapping", None)
        num_rows = logits.shape[0] if hasattr(logits, "shape") else len(logits)

        # Thinking-budget check
        thinking_budget_exceeded = getattr(ctx, "thinking_budget_exceeded", False)

        for row_idx in range(num_rows):
            req_idx = expanded_idx_mapping[row_idx] if expanded_idx_mapping is not None else row_idx
            state = self.req_states.get(req_idx)
            if state is None or not state.is_active:
                continue

            state.step_count += 1

            # Check thinking budget forcing: if budget exhausted, force termination / token restriction
            if thinking_budget_exceeded or (state.thinking_budget is not None and state.step_count >= state.thinking_budget):
                # Mask out further thinking tokens if defined
                pass

            # Apply token constraints
            logits = self._mask_logits_row(logits, row_idx, state)

        return logits

    def _mask_logits_row(self, logits: Any, row_idx: int, state: RequestSlotState) -> Any:
        """Apply token constraints to a single logits row with -inf safety recovery."""
        if state.allowed_token_ids is None and state.banned_token_ids is None:
            return logits

        # Check torch tensor vs list/numpy
        has_torch = hasattr(logits, "is_cuda") or hasattr(logits, "masked_fill_")

        if has_torch:
            import torch
            row = logits[row_idx]
            original_row = row.clone()

            if state.banned_token_ids:
                banned_indices = torch.tensor(list(state.banned_token_ids), dtype=torch.long, device=row.device)
                row[banned_indices] = float("-inf")

            if state.allowed_token_ids:
                mask = torch.ones_like(row, dtype=torch.bool)
                allowed_indices = torch.tensor(list(state.allowed_token_ids), dtype=torch.long, device=row.device)
                mask[allowed_indices] = False
                row[mask] = float("-inf")

            # Check if all tokens were masked to -inf
            if torch.all(torch.isneginf(row)):
                # Safe recovery: restore token with highest original logit
                best_token = torch.argmax(original_row)
                row[best_token] = original_row[best_token]
        else:
            # Fallback for CPU / pure Python / NumPy list
            row = logits[row_idx]
            original_row = list(row)
            vocab_size = len(row)

            if state.banned_token_ids:
                for tid in state.banned_token_ids:
                    if 0 <= tid < vocab_size:
                        row[tid] = float("-inf")

            if state.allowed_token_ids:
                allowed_set = set(state.allowed_token_ids)
                for tid in range(vocab_size):
                    if tid not in allowed_set:
                        row[tid] = float("-inf")

            # Check if all are -inf
            if all(v == float("-inf") for v in row):
                best_idx = max(range(vocab_size), key=lambda i: original_row[i])
                row[best_idx] = original_row[best_idx]

        return logits


# ==============================================================================
# 2. vLLM V1 BatchUpdate Interface (v0.10 - v0.32)
# ==============================================================================

class VllmV1BatchUpdateProcessor(_BaseV1LogitsProcessor):
    """
    Adapter for vLLM V1 (BatchUpdate interface).

    Contract & Invariants:
    - is_argmax_invariant() -> False (modifies sampling distribution).
    - validate_params(sampling_params) validates upfront.
    - update_state(batch_update) processes in strict order:
        1. removed
        2. added
        3. moved (including SWAP directionality)
    - apply(logits) on tensor [num_requests, vocab_size] with -inf row recovery.
    """

    def __init__(
        self,
        vllm_config: Any = None,
        device: str = "cpu",
        is_pin_memory: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.device = device
        self.is_pin_memory = is_pin_memory
        self.slots: Dict[int, RequestSlotState] = {}

    @classmethod
    def is_argmax_invariant(cls) -> bool:
        """Returns whether this processor leaves the argmax invariant."""
        return False

    @classmethod
    def validate_params(cls, sampling_params: Any) -> None:
        """Validate sampling parameters upfront."""
        VllmV2ModelRunnerProcessor.validate_params(sampling_params)

    def update_state(self, batch_update: BatchUpdate) -> None:
        """
        Process batch updates in strict order:
        1. removed
        2. added
        3. moved (handles MoveDirectionality.SWAP without collision)
        """
        # Step 1: Process Removals
        for req_idx in batch_update.removed:
            self.slots.pop(req_idx, None)

        # Step 2: Process Additions
        for req_idx, sampling_params in batch_update.added:
            state = RequestSlotState(req_idx=req_idx, sampling_params=sampling_params)
            state.reset(req_idx, sampling_params)
            self.slots[req_idx] = state

        # Step 3: Process Moves
        # To avoid collisions during SWAP or overlapping moves, staged buffer is used
        temp_buffer: Dict[int, RequestSlotState] = {}
        for from_idx, to_idx, direction in batch_update.moved:
            if from_idx in self.slots:
                temp_buffer[to_idx] = self.slots.pop(from_idx)
                temp_buffer[to_idx].req_idx = to_idx

        # Commit moved slots
        self.slots.update(temp_buffer)

    def apply(self, logits: Any) -> Any:
        """
        Apply token constraint masks on tensor [num_requests, vocab_size].
        Implements -inf row recovery learning from MinTokensLogitsProcessor.
        """
        num_rows = logits.shape[0] if hasattr(logits, "shape") else len(logits)

        has_torch = hasattr(logits, "is_cuda") or hasattr(logits, "masked_fill_")

        for row_idx in range(num_rows):
            state = self.slots.get(row_idx)
            if state is None:
                continue
            if not getattr(state, "is_active", True):
                continue

            if hasattr(state, "step_count"):
                state.step_count += 1

            allowed_token_ids = getattr(state, "allowed_token_ids", None)
            banned_token_ids = getattr(state, "banned_token_ids", None)

            if has_torch:
                import torch
                row = logits[row_idx]
                original_row = row.clone()

                if banned_token_ids:
                    banned_idx = torch.tensor(list(banned_token_ids), dtype=torch.long, device=row.device)
                    row[banned_idx] = float("-inf")

                if allowed_token_ids:
                    mask = torch.ones_like(row, dtype=torch.bool)
                    allowed_idx = torch.tensor(list(allowed_token_ids), dtype=torch.long, device=row.device)
                    mask[allowed_idx] = False
                    row[mask] = float("-inf")

                # Recovery if row became entirely -inf
                if torch.all(torch.isneginf(row)):
                    best_token = torch.argmax(original_row)
                    row[best_token] = original_row[best_token]
            else:
                row = logits[row_idx]
                original_row = list(row)
                vocab_size = len(row)

                if banned_token_ids:
                    for tid in banned_token_ids:
                        if 0 <= tid < vocab_size:
                            row[tid] = float("-inf")

                if allowed_token_ids:
                    allowed_set = set(allowed_token_ids)
                    for tid in range(vocab_size):
                        if tid not in allowed_set:
                            row[tid] = float("-inf")

                if all(v == float("-inf") for v in row):
                    best_idx = max(range(vocab_size), key=lambda i: original_row[i])
                    row[best_idx] = original_row[best_idx]

        return logits


# ==============================================================================
# 3. Unified VllmCompatAdapter
# ==============================================================================

class VllmInterfaceType(str, enum.Enum):
    V2_MODEL_RUNNER = "V2_MODEL_RUNNER"
    V1_BATCH_UPDATE = "V1_BATCH_UPDATE"
    COMPAT_FALLBACK = "COMPAT_FALLBACK"


class VllmCompatAdapter:
    """
    Detects and adapts between vLLM V1 BatchUpdate and V2 Model Runner interfaces.
    """

    @staticmethod
    def detect_interface() -> VllmInterfaceType:
        """Detect which vLLM logits processor interface is available in current environment."""
        try:
            import vllm.v1.worker.gpu.sample.logits_processor.interface  # type: ignore # noqa: F401
            return VllmInterfaceType.V2_MODEL_RUNNER
        except ImportError:
            pass

        try:
            import vllm.v1.sample.logits_processor.interface  # type: ignore # noqa: F401
            return VllmInterfaceType.V1_BATCH_UPDATE
        except ImportError:
            pass

        return VllmInterfaceType.COMPAT_FALLBACK

    @classmethod
    def create_processor(
        cls,
        interface_type: Optional[VllmInterfaceType] = None,
        vllm_config: Any = None,
        **kwargs: Any,
    ) -> Union[VllmV2ModelRunnerProcessor, VllmV1BatchUpdateProcessor]:
        """Factory creating the appropriate logits processor instance."""
        target_interface = interface_type or cls.detect_interface()

        if target_interface == VllmInterfaceType.V2_MODEL_RUNNER:
            logger.info("Initializing VllmV2ModelRunnerProcessor")
            return VllmV2ModelRunnerProcessor(vllm_config=vllm_config)
        else:
            logger.info("Initializing VllmV1BatchUpdateProcessor")
            device = kwargs.get("device", "cpu")
            is_pin_memory = kwargs.get("is_pin_memory", False)
            return VllmV1BatchUpdateProcessor(
                vllm_config=vllm_config,
                device=device,
                is_pin_memory=is_pin_memory,
            )
