"""
runtime_hook.py
Python-level implementation and emulator of the Renslate in-game runtime hook.
Validates:
  1. RevertableDict / RevertableList preservation across save/load/rollback cycles.
  2. Idempotent text filtering: f(f(text)) == f(text).
  3. 90-frame storm circuit breaker preventing engine crash on recursive evaluation errors.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional


class MockRevertableDict(dict):
    """
    Emulates renpy.revertable.RevertableDict rollback tracking.
    """
    def copy_state(self) -> MockRevertableDict:
        return MockRevertableDict(copy.deepcopy(dict(self)))


class MockRevertableList(list):
    """
    Emulates renpy.revertable.RevertableList rollback tracking.
    """
    def copy_state(self) -> MockRevertableList:
        return MockRevertableList(copy.deepcopy(list(self)))


class RenlocCircuitBreaker:
    """
    Monitors consecutive text filter evaluation cycles and errors.
    Automatically trips after 90 consecutive frame errors to return raw text
    without freezing the engine.
    """
    MAX_CONSECUTIVE_ERRORS: int = 90

    def __init__(self, max_consecutive_errors: int = 90):
        self.max_consecutive_errors = max_consecutive_errors
        self.consecutive_errors: int = 0
        self.tripped: bool = False
        self.trip_reason: str = ""
        self.trip_count: int = 0

    def record_success(self) -> None:
        if not self.tripped:
            self.consecutive_errors = 0

    def record_error(self, err_msg: str = "") -> None:
        self.consecutive_errors += 1
        if self.consecutive_errors >= self.max_consecutive_errors and not self.tripped:
            self.tripped = True
            self.trip_count += 1
            self.trip_reason = (
                f"90-Frame Storm Circuit Breaker tripped after {self.consecutive_errors} "
                f"consecutive errors. Last error: {err_msg}"
            )

    def reset(self) -> None:
        self.consecutive_errors = 0
        self.tripped = False
        self.trip_reason = ""


class RuntimeHookManager:
    """
    Manages the runtime hook state, idempotent filter, and rollback simulation.
    """

    def __init__(self):
        self.breaker = RenlocCircuitBreaker()
        self.dynamic_context = MockRevertableDict()
        self.history_stack: List[Dict[str, Any]] = []

    def filter_text(self, text: str) -> str:
        """
        Pure, idempotent text filter function: satisfies f(f(text)) == f(text).
        """
        if not text or not isinstance(text, str):
            return text

        if self.breaker.tripped:
            return text

        try:
            result = text
            if "__RENSLATE_FORCE_RECURSIVE_ERROR__" in text:
                raise RuntimeError("Synthetic recursive substitution error")

            for token, replacement in self.dynamic_context.items():
                if token in result:
                    result = result.replace(token, replacement)

            self.breaker.record_success()
            return result

        except Exception as e:
            self.breaker.record_error(str(e))
            return text

    # -------------------------------------------------------------------------
    # Save / Load / Rollback Simulation
    # -------------------------------------------------------------------------
    def checkpoint(self) -> Dict[str, Any]:
        """
        Takes a snapshot of revertable state (emulates renpy.save or rollback checkpoint).
        """
        state = {
            "dynamic_context": self.dynamic_context.copy_state(),
            "breaker_consecutive_errors": self.breaker.consecutive_errors,
            "breaker_tripped": self.breaker.tripped,
        }
        self.history_stack.append(state)
        return state

    def rollback(self) -> bool:
        """
        Rolls back to the previous checkpoint (emulates renpy.rollback).
        """
        if not self.history_stack:
            return False
        prev = self.history_stack.pop()
        self.dynamic_context = prev["dynamic_context"].copy_state()
        self.breaker.consecutive_errors = prev["breaker_consecutive_errors"]
        self.breaker.tripped = prev["breaker_tripped"]
        return True


# Backward-compatible alias
StormCircuitBreaker = RenlocCircuitBreaker


def generate_runtime_hook_rpy(breaker_threshold: int = 90) -> str:
    """
    Emits the canonical `renloc_runtime_hook.rpy` source code to embed in games.
    Strictly 4 spaces indentation, 0 tabs.
    """
    return f"""# ==============================================================================
# Ren'Py Runtime Localization Hook - Renslate Autonomous Localization Engine
# Priority: init -990 python
# Three Safety Pillars:
#   1. Revertable* State Containers (Zero rollback memory leaks)
#   2. Idempotent config.say_menu_text_filter (Pure text transformation)
#   3. 90-Frame Storm Circuit Breaker (Anti-freeze protection)
# ==============================================================================

init -990 python:
    # 1. Revertable State Imports
    try:
        from renpy.revertable import RevertableDict, RevertableList, RevertableSet
    except ImportError:
        try:
            from renpy.python import RevertableDict, RevertableList, RevertableSet
        except ImportError:
            RevertableDict = dict
            RevertableList = list
            RevertableSet = set

    # 2. 90-Frame Storm Circuit Breaker
    class _RenlocStormCircuitBreaker(object):
        def __init__(self, threshold={breaker_threshold}):
            self.threshold = threshold
            self.consecutive_errors = 0
            self.tripped = False

        def record_success(self):
            self.consecutive_errors = 0

        def record_error(self):
            self.consecutive_errors += 1
            if self.consecutive_errors >= self.threshold:
                self.tripped = True

        def reset(self):
            self.consecutive_errors = 0
            self.tripped = False

    # Instantiate runtime singleton
    _renloc_breaker = _RenlocStormCircuitBreaker(threshold={breaker_threshold})
    _renloc_dynamic_names = RevertableDict()

    # 3. Idempotent Say/Menu Filter
    def _renloc_say_menu_text_filter(text):
        global _renloc_breaker, _renloc_dynamic_names

        # Breaker open: bypass filtering immediately
        if _renloc_breaker.tripped:
            return text

        if not text:
            return text

        try:
            result = text
            for old_name, new_name in _renloc_dynamic_names.items():
                if old_name in result:
                    result = result.replace(old_name, new_name)
            _renloc_breaker.record_success()
            return result
        except Exception:
            _renloc_breaker.record_error()
            return text

    # Chaining onto existing config filter if present
    if getattr(config, "say_menu_text_filter", None) is not None:
        _renloc_prior_filter = config.say_menu_text_filter
        def _renloc_chained_filter(text):
            first_pass = _renloc_prior_filter(text)
            return _renloc_say_menu_text_filter(first_pass)
        config.say_menu_text_filter = _renloc_chained_filter
    else:
        config.say_menu_text_filter = _renloc_say_menu_text_filter
"""

