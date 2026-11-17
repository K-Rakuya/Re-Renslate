"""
tests/test_decompiler.py

Comprehensive unit test suite for renloc_core.decompiler:
- disassembler.py: multi-version xdis disassembly (Python 2.7 through 3.12+)
- decompiler_pipeline.py: SAILR CFG, PyChD structuring, PyLingual expressions
- emi_checker.py: EtherVal EMI validation and automatic xdis disassembly fallback.
"""

import ast
import marshal
import os
import sys
import types
from pathlib import Path
import pytest
import xdis

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "py" / "renloc-core"))

from renloc_core.ast_reader.ast_nodes import PyCode
from renloc_core.decompiler import (
    Disassembler,
    DisassembledCode,
    DisassembledInstruction,
    DecompilerPipeline,
    DecompiledResult,
    DecompilationMode,
    BasicBlock,
    ControlFlowGraph,
    NaturalLoop,
    EtherValEMIChecker,
    EMICheckResult,
)


# =========================================================================
# 1. Multi-Version Disassembler Tests (Python 2.7 to 3.12+)
# =========================================================================

def test_disassembler_python_312_native():
    """Verify disassembly of native Python 3.12 code objects and instructions."""
    dis = Disassembler()
    code = compile("x = 42\ny = x + 1\nprint(y)", "<test>", "exec")
    result = dis.disassemble(code)

    assert isinstance(result, DisassembledCode)
    assert result.version[0] == 3
    assert len(result.instructions) > 0
    assert any(inst.opname == "STORE_NAME" and inst.argval == "x" for inst in result.instructions)
    assert any(inst.opname == "STORE_NAME" and inst.argval == "y" for inst in result.instructions)
    assert any(inst.opname in ("CALL", "CALL_FUNCTION") for inst in result.instructions)

    # Check to_text output
    text = result.to_text()
    assert "Disassembly for <module>" in text
    assert "STORE_NAME" in text


def test_disassembler_python_27_code2():
    """Verify cross-version disassembly of Python 2.7 bytecode via Code2."""
    dis = Disassembler()
    # Python 2.7 bytecode:
    # LOAD_CONST 0 (100), STORE_NAME 0 ('score'), LOAD_CONST 1 (None), RETURN_VALUE
    code2 = xdis.Code2(
        co_argcount=0,
        co_nlocals=0,
        co_stacksize=2,
        co_flags=64,
        co_code=bytes([100, 0, 0, 90, 0, 0, 100, 1, 0, 83]),
        co_consts=(100, None),
        co_names=("score",),
        co_varnames=(),
        co_freevars=(),
        co_cellvars=(),
        co_filename="<script27>",
        co_name="<module>",
        co_firstlineno=1,
        co_lnotab=b"",
    )

    result = dis.disassemble(code2, python_version=(2, 7))
    assert result.version == (2, 7)
    assert result.name == "<module>"
    assert len(result.instructions) == 4

    inst_names = [i.opname for i in result.instructions]
    assert inst_names == ["LOAD_CONST", "STORE_NAME", "LOAD_CONST", "RETURN_VALUE"]
    assert result.instructions[1].argval == "score"


def test_disassembler_magic_and_version_detection():
    """Verify version detection from magic numbers and version strings."""
    dis = Disassembler()

    # Normalization tests
    assert dis.normalize_version("2.7") == (2, 7)
    assert dis.normalize_version(3.12) == (3, 12)
    assert dis.normalize_version((3, 10)) == (3, 10)

    # Magic byte detection
    magic_27 = xdis.magics.magics["2.7"]
    detected_27 = dis.detect_version(magic_27 + b"\x00" * 4)
    assert detected_27 == (2, 7)

    magic_312 = xdis.magics.magics["3.12"]
    detected_312 = dis.detect_version(magic_312 + b"\x00" * 12)
    assert detected_312 == (3, 12)


def test_disassembler_pycode_handling():
    """Verify disassembly from inert PyCode objects with both source and bytecode."""
    dis = Disassembler()

    # PyCode with source
    p_src = PyCode(source="greeting = 'Hello Renpy'\ncount = 1", location=("story.rpy", 10))
    res_src = dis.disassemble(p_src)
    assert res_src.name == "<module>"
    assert any(i.argval == "greeting" for i in res_src.instructions)

    # PyCode with marshalled bytecode
    compiled = compile("health = 100", "<renpy>", "exec")
    p_bc = PyCode(bytecode=marshal.dumps(compiled), location=("story.rpy", 20))
    res_bc = dis.disassemble(p_bc)
    assert any(i.argval == "health" for i in res_bc.instructions)


# =========================================================================
# 2. CFG Construction & SAILR Dominators
# =========================================================================

def test_cfg_basic_blocks_and_dominators():
    """Verify CFG partition, predecessor/successor edges, and dominator computation."""
    dis = Disassembler()
    code = compile(
        "x = 10\nif x > 5:\n    y = 1\nelse:\n    y = 2\nz = x + y",
        "<cfg_test>",
        "exec",
    )
    dis_code = dis.disassemble(code)
    cfg = ControlFlowGraph.build(dis_code.instructions)

    assert len(cfg.blocks) >= 3
    assert cfg.entry_id == 0

    # Entry block must dominate itself
    assert 0 in cfg.dominators[0]

    # Every block must be dominated by entry block 0
    for node, doms in cfg.dominators.items():
        assert 0 in doms

    # Dominator tree check
    assert cfg.entry_id in cfg.dominator_tree


def test_cfg_natural_loop_detection():
    """Verify loop detection in CFG with natural back-edges."""
    dis = Disassembler()
    code = compile(
        "total = 0\nfor i in [1, 2, 3]:\n    total += i",
        "<loop_test>",
        "exec",
    )
    dis_code = dis.disassemble(code)
    cfg = ControlFlowGraph.build(dis_code.instructions)

    assert len(cfg.loops) > 0
    loop = cfg.loops[0]
    assert loop.loop_type == "for"
    assert loop.header_id in cfg.blocks
    assert loop.latch_id in cfg.blocks
    assert loop.body_blocks


# =========================================================================
# 3. Decompiler Pipeline (PyChD + PyLingual + SAILR)
# =========================================================================

def test_decompile_expressions_and_inplace():
    """Verify decompilation of variables, arithmetic, and augmented assignments."""
    pipeline = DecompilerPipeline()
    src = "a = 10\nb = 20\na += b\nc = (a * 2)"
    code = compile(src, "<test>", "exec")

    decomp = pipeline.decompile(code)
    assert decomp.mode == DecompilationMode.DECOMPILED_SOURCE
    assert not decomp.is_fallback

    # Validate AST parsing
    ast_tree = ast.parse(decomp.source_code)
    assert isinstance(ast_tree, ast.Module)

    # Check key reconstructed statements
    assert "a = 10" in decomp.source_code
    assert "b = 20" in decomp.source_code
    assert "a += b" in decomp.source_code


def test_decompile_conditionals_if_else():
    """Verify decompilation of if-else branch structuring."""
    pipeline = DecompilerPipeline()
    src = (
        "score = 75\n"
        "if score >= 50:\n"
        "    result = 'Passed'\n"
        "else:\n"
        "    result = 'Failed'"
    )
    code = compile(src, "<test>", "exec")
    decomp = pipeline.decompile(code)

    assert decomp.mode == DecompilationMode.DECOMPILED_SOURCE
    assert "if" in decomp.source_code
    assert "else:" in decomp.source_code
    assert "Passed" in decomp.source_code
    assert "Failed" in decomp.source_code


def test_decompile_renpy_patterns():
    """Verify decompilation of Ren'Py specific idiom: renpy.say(s, msg)."""
    pipeline = DecompilerPipeline()
    src = "renpy.say(s, 'A mysterious journey awaits.')"
    code = compile(src, "<test>", "exec")
    decomp = pipeline.decompile(code)

    assert "renpy.say" in decomp.source_code
    assert "A mysterious journey awaits." in decomp.source_code


def test_decompile_for_loop():
    """Verify decompilation of for loops with iteration variable recovery."""
    pipeline = DecompilerPipeline()
    src = (
        "items = ['sword', 'shield']\n"
        "for item in items:\n"
        "    renpy.say(narrator, item)"
    )
    code = compile(src, "<test>", "exec")
    decomp = pipeline.decompile(code)

    assert "for item in" in decomp.source_code
    assert "renpy.say" in decomp.source_code


# =========================================================================
# 4. EtherVal EMI (Equivalence Modulo Inputs) Checker & Fallback
# =========================================================================

def test_emi_checker_success_on_valid_decompilation():
    """Verify EtherVal EMI succeeds and validates equivalence on accurate decompilation."""
    pipeline = DecompilerPipeline()
    emi = EtherValEMIChecker()

    src = (
        "points = 10\n"
        "if points > 5:\n"
        "    rank = 'A'\n"
        "else:\n"
        "    rank = 'B'"
    )
    code = compile(src, "<emi_test>", "exec")
    decomp = pipeline.decompile(code)

    emi_res = emi.check_equivalence(decomp, original_target=code)
    assert emi_res.is_equivalent is True
    assert emi_res.mode == DecompilationMode.DECOMPILED_SOURCE
    assert emi_res.checks_passed["recompilation"] is True
    assert emi_res.checks_passed["cfg_topology"] is True
    assert emi_res.checks_passed["opcode_semantics"] is True
    assert emi_res.checks_passed["emi_simulation"] is True
    assert emi_res.fallback_instructions is None


def test_emi_checker_fallback_on_syntax_error():
    """Verify EtherVal EMI automatically downgrades to xdis instruction stream on compile failure."""
    pipeline = DecompilerPipeline()
    emi = EtherValEMIChecker()

    code = compile("x = 100", "<test>", "exec")
    decomp = pipeline.decompile(code)

    # Invalidate source code to simulate decompiler bug / corrupt grammar
    decomp.source_code = "def syntax_error_here( %%% = invalid"

    emi_res = emi.check_equivalence(decomp, original_target=code)
    assert emi_res.is_equivalent is False
    assert emi_res.mode == DecompilationMode.DISASSEMBLY_FALLBACK
    assert emi_res.fallback_instructions is not None
    assert len(emi_res.fallback_instructions) > 0
    assert "syntax" in emi_res.error_reason.lower()


def test_emi_checker_fallback_on_semantic_divergence():
    """Verify EtherVal EMI downgrades to xdis instruction stream when variable stores are lost."""
    pipeline = DecompilerPipeline()
    emi = EtherValEMIChecker()

    # Original code sets x, y, z
    src = "x = 1\ny = 2\nz = 3"
    code = compile(src, "<test>", "exec")
    decomp = pipeline.decompile(code)

    # Simulate incomplete decompilation where 'z = 3' is omitted
    decomp.source_code = "x = 1\ny = 2\n"

    emi_res = emi.check_equivalence(decomp, original_target=code)
    assert emi_res.is_equivalent is False
    assert emi_res.mode == DecompilationMode.DISASSEMBLY_FALLBACK
    assert emi_res.fallback_instructions is not None
    assert "missing variable assignments" in emi_res.error_reason


def test_emi_checker_fallback_on_branch_divergence():
    """Verify EtherVal EMI downgrades to xdis instruction stream if conditional branch is flattened."""
    pipeline = DecompilerPipeline()
    emi = EtherValEMIChecker()

    # Original code has an if-condition
    src = "flag = True\nif flag:\n    state = 'active'"
    code = compile(src, "<test>", "exec")
    decomp = pipeline.decompile(code)

    # Corrupt source by flattening the condition into unconditional statement
    decomp.source_code = "flag = True\nstate = 'active'"

    emi_res = emi.check_equivalence(decomp, original_target=code)
    assert emi_res.is_equivalent is False
    assert emi_res.mode == DecompilationMode.DISASSEMBLY_FALLBACK
    assert "branch mismatch" in emi_res.error_reason
