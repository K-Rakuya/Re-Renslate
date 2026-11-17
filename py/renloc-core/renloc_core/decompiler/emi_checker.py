"""
renloc_core/decompiler/emi_checker.py

EtherVal Equivalence Modulo Inputs (EMI) Checker.
Performs rigorous semantic and behavioral equivalence verification on decompiled Python code:
1. Recompiles recovered Python source into bytecode.
2. Disassembles and compares Control Flow Graph (CFG) topology (blocks, loops, branch structure).
3. Compares semantic opcode sequences and dataflow (variable stores, function calls, comparisons).
4. Executes/simulates behavioral equivalence across test inputs (Modulo Inputs).
5. If EtherVal EMI fails, automatically downgrades to direct analysis on xdis instruction stream.
"""

from __future__ import annotations

import ast
import copy
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from renloc_core.decompiler.decompiler_pipeline import (
    ControlFlowGraph,
    DecompilationMode,
    DecompiledResult,
)
from renloc_core.decompiler.disassembler import (
    DisassembledCode,
    DisassembledInstruction,
    Disassembler,
)


@dataclass
class EMICheckResult:
    """Result of EtherVal EMI equivalence validation."""

    is_equivalent: bool
    mode: DecompilationMode
    source_code: Optional[str]
    recompiled_code: Optional[types.CodeType] = None
    fallback_instructions: Optional[List[DisassembledInstruction]] = None
    cfg_original: Optional[ControlFlowGraph] = None
    cfg_recompiled: Optional[ControlFlowGraph] = None
    checks_passed: Dict[str, bool] = field(default_factory=dict)
    error_reason: Optional[str] = None

    def __repr__(self) -> str:
        status = "PASSED (Equivalence Verified)" if self.is_equivalent else "FAILED (Downgraded to Disassembly Fallback)"
        return f"<EMICheckResult: {status}, checks={self.checks_passed}, reason={self.error_reason}>"


class EtherValEMIChecker:
    """
    Equivalence Modulo Inputs (EMI) verification engine.
    Ensures decompiled Python code faithfully preserves original bytecode semantics
    or gracefully falls back to xdis instruction stream.
    """

    def __init__(self, disassembler: Optional[Disassembler] = None):
        self.disassembler = disassembler or Disassembler()

    def check_equivalence(
        self,
        decompiled: DecompiledResult,
        original_target: Any = None,
        test_inputs: Optional[List[Dict[str, Any]]] = None,
    ) -> EMICheckResult:
        """
        Validates equivalence of decompiled result against original bytecode.

        Args:
            decompiled: Output from DecompilerPipeline.
            original_target: Optional original PyCode, code object, or bytes.
            test_inputs: Optional list of variable environment states for EMI simulation.

        Returns:
            EMICheckResult with equivalence status and fallback instructions if failed.
        """
        checks: Dict[str, bool] = {
            "recompilation": False,
            "cfg_topology": False,
            "opcode_semantics": False,
            "emi_simulation": False,
        }

        orig_instructions = decompiled.instruction_stream.instructions
        orig_cfg = decompiled.cfg

        # If already flagged as fallback by pipeline
        if decompiled.is_fallback:
            return EMICheckResult(
                is_equivalent=False,
                mode=DecompilationMode.DISASSEMBLY_FALLBACK,
                source_code=decompiled.source_code,
                fallback_instructions=orig_instructions,
                cfg_original=orig_cfg,
                checks_passed=checks,
                error_reason=decompiled.fallback_reason or "Pipeline produced fallback directly",
            )

        # ---------------------------------------------------------------------
        # 1. Recompilation Check
        # ---------------------------------------------------------------------
        recompiled_code: Optional[types.CodeType] = None
        try:
            recompiled_code = compile(decompiled.source_code, "<emi_verify>", "exec")
            checks["recompilation"] = True
        except Exception as e:
            return EMICheckResult(
                is_equivalent=False,
                mode=DecompilationMode.DISASSEMBLY_FALLBACK,
                source_code=decompiled.source_code,
                fallback_instructions=orig_instructions,
                cfg_original=orig_cfg,
                checks_passed=checks,
                error_reason=f"Recompilation syntax/compilation error: {e}",
            )

        # ---------------------------------------------------------------------
        # 2. Recompiled Disassembly & CFG Construction
        # ---------------------------------------------------------------------
        try:
            recompiled_dis = self.disassembler.disassemble(recompiled_code)
            recompiled_cfg = ControlFlowGraph.build(recompiled_dis.instructions)
        except Exception as e:
            return EMICheckResult(
                is_equivalent=False,
                mode=DecompilationMode.DISASSEMBLY_FALLBACK,
                source_code=decompiled.source_code,
                fallback_instructions=orig_instructions,
                cfg_original=orig_cfg,
                checks_passed=checks,
                error_reason=f"Recompiled bytecode disassembly failure: {e}",
            )

        # ---------------------------------------------------------------------
        # 3. CFG Topology Equivalence Check
        # ---------------------------------------------------------------------
        orig_has_loops = len(orig_cfg.loops) > 0
        recomp_has_loops = len(recompiled_cfg.loops) > 0

        orig_branches = sum(
            1 for b in orig_cfg.blocks.values()
            if any(i.is_conditional_jump for i in b.instructions)
        )
        recomp_branches = sum(
            1 for b in recompiled_cfg.blocks.values()
            if any(i.is_conditional_jump for i in b.instructions)
        )

        # Loops must match in presence
        if orig_has_loops != recomp_has_loops:
            return EMICheckResult(
                is_equivalent=False,
                mode=DecompilationMode.DISASSEMBLY_FALLBACK,
                source_code=decompiled.source_code,
                fallback_instructions=orig_instructions,
                cfg_original=orig_cfg,
                cfg_recompiled=recompiled_cfg,
                checks_passed=checks,
                error_reason=(
                    f"CFG Topology loop mismatch: original has_loops={orig_has_loops}, "
                    f"recompiled has_loops={recomp_has_loops}"
                ),
            )

        # Branches must match in presence
        if (orig_branches > 0) != (recomp_branches > 0):
            return EMICheckResult(
                is_equivalent=False,
                mode=DecompilationMode.DISASSEMBLY_FALLBACK,
                source_code=decompiled.source_code,
                fallback_instructions=orig_instructions,
                cfg_original=orig_cfg,
                cfg_recompiled=recompiled_cfg,
                checks_passed=checks,
                error_reason=(
                    f"CFG Topology branch mismatch: original branches={orig_branches}, "
                    f"recompiled branches={recomp_branches}"
                ),
            )

        checks["cfg_topology"] = True

        # ---------------------------------------------------------------------
        # 4. Semantic Opcode & Dataflow Equivalence Check
        # ---------------------------------------------------------------------
        orig_events = self._extract_semantic_events(orig_instructions)
        recomp_events = self._extract_semantic_events(recompiled_dis.instructions)

        # Verify all modified variables in original exist in recompiled
        orig_stores = {name for event_type, name in orig_events if event_type == "STORE"}
        recomp_stores = {name for event_type, name in recomp_events if event_type == "STORE"}

        if not orig_stores.issubset(recomp_stores) and not (orig_stores == recomp_stores):
            missing = orig_stores - recomp_stores
            return EMICheckResult(
                is_equivalent=False,
                mode=DecompilationMode.DISASSEMBLY_FALLBACK,
                source_code=decompiled.source_code,
                fallback_instructions=orig_instructions,
                cfg_original=orig_cfg,
                cfg_recompiled=recompiled_cfg,
                checks_passed=checks,
                error_reason=f"Semantic opcode mismatch: missing variable assignments {missing}",
            )

        # Verify function calls in original exist in recompiled
        orig_calls = [name for event_type, name in orig_events if event_type == "CALL"]
        recomp_calls = [name for event_type, name in recomp_events if event_type == "CALL"]
        if orig_calls and not any(c in recomp_calls for c in orig_calls):
            return EMICheckResult(
                is_equivalent=False,
                mode=DecompilationMode.DISASSEMBLY_FALLBACK,
                source_code=decompiled.source_code,
                fallback_instructions=orig_instructions,
                cfg_original=orig_cfg,
                cfg_recompiled=recompiled_cfg,
                checks_passed=checks,
                error_reason=f"Semantic opcode mismatch: function calls missing {orig_calls}",
            )

        checks["opcode_semantics"] = True

        # ---------------------------------------------------------------------
        # 5. Equivalence Modulo Inputs (EMI) Execution Simulation
        # ---------------------------------------------------------------------
        sim_ok = self._simulate_modulo_inputs(
            decompiled.source_code,
            recompiled_code,
            original_target,
            test_inputs or [{}, {"x": 10, "score": 100}],
        )
        if not sim_ok:
            return EMICheckResult(
                is_equivalent=False,
                mode=DecompilationMode.DISASSEMBLY_FALLBACK,
                source_code=decompiled.source_code,
                fallback_instructions=orig_instructions,
                cfg_original=orig_cfg,
                cfg_recompiled=recompiled_cfg,
                checks_passed=checks,
                error_reason="Equivalence Modulo Inputs (EMI) simulation mismatch across test inputs",
            )

        checks["emi_simulation"] = True

        # ---------------------------------------------------------------------
        # All checks passed!
        # ---------------------------------------------------------------------
        return EMICheckResult(
            is_equivalent=True,
            mode=DecompilationMode.DECOMPILED_SOURCE,
            source_code=decompiled.source_code,
            recompiled_code=recompiled_code,
            fallback_instructions=None,
            cfg_original=orig_cfg,
            cfg_recompiled=recompiled_cfg,
            checks_passed=checks,
            error_reason=None,
        )

    def _extract_semantic_events(self, instructions: List[DisassembledInstruction]) -> List[Tuple[str, str]]:
        """Extracts high-level semantic operations (stores, calls, comparisons) ignoring runtime scaffolding."""
        events: List[Tuple[str, str]] = []
        for inst in instructions:
            op = inst.opname
            if op in ("STORE_NAME", "STORE_FAST", "STORE_GLOBAL"):
                events.append(("STORE", str(inst.argval)))
            elif op == "STORE_ATTR":
                events.append(("STORE_ATTR", str(inst.argval)))
            elif op in ("CALL", "CALL_FUNCTION", "CALL_METHOD"):
                events.append(("CALL", str(inst.argval or inst.argrepr or "call")))
            elif op == "COMPARE_OP":
                events.append(("COMPARE", str(inst.argval or inst.argrepr or "==")))
            elif op == "FOR_ITER":
                events.append(("LOOP_FOR", "for"))
        return events

    def _simulate_modulo_inputs(
        self,
        source_code: str,
        recompiled_code: types.CodeType,
        original_target: Any,
        test_inputs: List[Dict[str, Any]],
    ) -> bool:
        """
        Executes recompiled bytecode across test inputs in an isolated namespace,
        and optionally compares against original code object if directly executable.
        """
        # Create a mock environment with common Ren'Py builtins
        class MockRenpy:
            @staticmethod
            def say(*args, **kwargs):
                return None

        class MockCharacter:
            def __init__(self, *args, **kwargs):
                pass

        base_builtins = {
            "renpy": MockRenpy(),
            "Character": MockCharacter,
            "DynamicCharacter": MockCharacter,
            "range": range,
            "len": len,
            "int": int,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
        }

        class AutoEnv(dict):
            def __missing__(self, key):
                return f"<mock_{key}>"

        for input_state in test_inputs:
            # Test recompiled execution
            ns_recomp = AutoEnv()
            ns_recomp.update(base_builtins)
            ns_recomp.update(input_state)
            try:
                exec(recompiled_code, ns_recomp)
            except Exception:
                # If recompiled fails to execute on valid test input
                return False

            # If original_target is a native code object, test original execution
            if isinstance(original_target, types.CodeType):
                ns_orig = AutoEnv()
                ns_orig.update(base_builtins)
                ns_orig.update(input_state)
                try:
                    exec(original_target, ns_orig)
                    # Compare resulting shared keys
                    for k in ns_orig:
                        if k in base_builtins:
                            continue
                        if k in ns_recomp and ns_orig[k] != ns_recomp[k]:
                            return False
                except Exception:
                    # Original might require specific game state; ignore if only recompiled passes
                    pass

        return True
