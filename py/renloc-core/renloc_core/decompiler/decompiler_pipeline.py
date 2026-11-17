"""
renloc_core/decompiler/decompiler_pipeline.py

Control-flow recovery and Python source decompilation framework integrating:
- SAILR: Structured AST and Intermediate Language Recovery (CFG without distortion, dominators, loops)
- PyChD: Python Control-flow Decompilation (loop and conditional structuring)
- PyLingual: Symbolic expression stack evaluation and high-level Python source reconstruction.
"""

from __future__ import annotations

import ast
import enum
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from renloc_core.decompiler.disassembler import (
    DisassembledCode,
    DisassembledInstruction,
    Disassembler,
)


class DecompilationMode(enum.Enum):
    """Execution mode of the decompiler pipeline."""

    DECOMPILED_SOURCE = "decompiled_source"
    DISASSEMBLY_FALLBACK = "disassembly_fallback"


@dataclass
class BasicBlock:
    """A maximal sequence of contiguous instructions with single entry and exit."""

    block_id: int
    start_offset: int
    end_offset: int
    instructions: List[DisassembledInstruction] = field(default_factory=list)
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    is_loop_header: bool = False
    is_loop_latch: bool = False

    def __repr__(self) -> str:
        inst_count = len(self.instructions)
        return (
            f"Block_{self.block_id}[{self.start_offset}:{self.end_offset}] "
            f"({inst_count} instrs, preds={self.predecessors}, succs={self.successors})"
        )


@dataclass
class NaturalLoop:
    """Represents an identified natural loop in the CFG."""

    header_id: int
    latch_id: int
    body_blocks: Set[int]
    loop_type: str = "while"  # "for", "while", or "infinite"
    target_var: Optional[str] = None
    iter_expr: Optional[str] = None
    condition_expr: Optional[str] = None


@dataclass
class ControlFlowGraph:
    """Control Flow Graph (CFG) representation with dominators and loop hierarchy."""

    blocks: Dict[int, BasicBlock] = field(default_factory=dict)
    entry_id: int = 0
    exit_ids: Set[int] = field(default_factory=set)
    dominators: Dict[int, Set[int]] = field(default_factory=dict)
    idom: Dict[int, Optional[int]] = field(default_factory=dict)
    dominator_tree: Dict[int, List[int]] = field(default_factory=dict)
    loops: List[NaturalLoop] = field(default_factory=list)

    @classmethod
    def build(cls, instructions: List[DisassembledInstruction]) -> ControlFlowGraph:
        """Constructs CFG from a sequence of disassembled instructions."""
        cfg = cls()
        if not instructions:
            return cfg

        # 1. Identify block leaders
        leaders: Set[int] = {instructions[0].offset}
        offset_to_idx = {inst.offset: i for i, inst in enumerate(instructions)}

        for i, inst in enumerate(instructions):
            # Target of jump is a leader
            if inst.is_jump and inst.jump_target is not None:
                if inst.jump_target in offset_to_idx:
                    leaders.add(inst.jump_target)

            # Instruction following a branch/jump/terminal is a leader
            if inst.is_jump or inst.is_terminal:
                if i + 1 < len(instructions):
                    leaders.add(instructions[i + 1].offset)

        # 2. Partition into BasicBlocks
        sorted_leaders = sorted(leaders)
        leader_to_block_id: Dict[int, int] = {}
        for block_id, leader_offset in enumerate(sorted_leaders):
            leader_to_block_id[leader_offset] = block_id

        current_block: Optional[BasicBlock] = None
        for inst in instructions:
            if inst.offset in leader_to_block_id:
                if current_block is not None and current_block.instructions:
                    current_block.end_offset = current_block.instructions[-1].offset
                    cfg.blocks[current_block.block_id] = current_block

                block_id = leader_to_block_id[inst.offset]
                current_block = BasicBlock(
                    block_id=block_id,
                    start_offset=inst.offset,
                    end_offset=inst.offset,
                    instructions=[inst],
                )
            else:
                if current_block is not None:
                    current_block.instructions.append(inst)

        if current_block is not None and current_block.instructions:
            current_block.end_offset = current_block.instructions[-1].offset
            cfg.blocks[current_block.block_id] = current_block

        # 3. Connect CFG edges
        for block_id, block in cfg.blocks.items():
            if not block.instructions:
                continue

            last_inst = block.instructions[-1]

            if last_inst.is_return or (last_inst.opname in ("RAISE_VARARGS", "RERAISE")):
                cfg.exit_ids.add(block_id)
                continue

            if last_inst.is_unconditional_jump:
                target_offset = last_inst.jump_target
                if target_offset in leader_to_block_id:
                    succ_id = leader_to_block_id[target_offset]
                    cfg._add_edge(block_id, succ_id)
                continue

            if last_inst.is_conditional_jump:
                # Branch taken edge
                target_offset = last_inst.jump_target
                if target_offset in leader_to_block_id:
                    succ_id = leader_to_block_id[target_offset]
                    cfg._add_edge(block_id, succ_id)

                # Fallthrough edge
                next_block_id = block_id + 1
                if next_block_id in cfg.blocks:
                    cfg._add_edge(block_id, next_block_id)
                continue

            # Sequential fallthrough
            next_block_id = block_id + 1
            if next_block_id in cfg.blocks:
                cfg._add_edge(block_id, next_block_id)
            else:
                cfg.exit_ids.add(block_id)

        # 4. Compute Dominators and Natural Loops
        cfg._compute_dominators()
        cfg._detect_natural_loops()

        return cfg

    def _add_edge(self, u: int, v: int) -> None:
        """Adds a directed edge from u to v."""
        if v not in self.blocks[u].successors:
            self.blocks[u].successors.append(v)
        if u not in self.blocks[v].predecessors:
            self.blocks[v].predecessors.append(u)

    def _compute_dominators(self) -> None:
        """Computes dominator sets and immediate dominator tree using fixed-point iteration."""
        all_nodes = set(self.blocks.keys())
        if not all_nodes:
            return

        self.dominators = {node: set(all_nodes) for node in all_nodes}
        self.dominators[self.entry_id] = {self.entry_id}

        changed = True
        while changed:
            changed = False
            for node in all_nodes:
                if node == self.entry_id:
                    continue
                preds = self.blocks[node].predecessors
                if not preds:
                    new_dom = {node}
                else:
                    new_dom = set.intersection(*(self.dominators[p] for p in preds))
                    new_dom.add(node)

                if new_dom != self.dominators[node]:
                    self.dominators[node] = new_dom
                    changed = True

        # Immediate dominators
        self.idom = {node: None for node in all_nodes}
        self.dominator_tree = {node: [] for node in all_nodes}

        for node in all_nodes:
            if node == self.entry_id:
                continue
            strict_doms = self.dominators[node] - {node}
            for d in strict_doms:
                if all(d not in (self.dominators[other] - {other}) for other in strict_doms if other != d):
                    self.idom[node] = d
                    self.dominator_tree[d].append(node)
                    break

    def _detect_natural_loops(self) -> None:
        """Finds back-edges and constructs NaturalLoop records."""
        self.loops.clear()
        for u, block in self.blocks.items():
            for v in block.successors:
                # Back-edge if v dominates u
                if v in self.dominators.get(u, set()):
                    self.blocks[v].is_loop_header = True
                    self.blocks[u].is_loop_latch = True

                    loop_body = {v, u}
                    stack = [u]
                    while stack:
                        curr = stack.pop()
                        for p in self.blocks[curr].predecessors:
                            if p not in loop_body and p != v:
                                loop_body.add(p)
                                stack.append(p)

                    loop_type = "while"
                    header_block = self.blocks[v]
                    if any(inst.opname == "FOR_ITER" for inst in header_block.instructions):
                        loop_type = "for"

                    self.loops.append(
                        NaturalLoop(
                            header_id=v,
                            latch_id=u,
                            body_blocks=loop_body,
                            loop_type=loop_type,
                        )
                    )


@dataclass
class DecompiledResult:
    """Output of the decompiler pipeline containing source code, CFG, and mode."""

    source_code: str
    cfg: ControlFlowGraph
    instruction_stream: DisassembledCode
    version: Tuple[int, int]
    mode: DecompilationMode = DecompilationMode.DECOMPILED_SOURCE
    ast_tree: Optional[ast.AST] = None
    is_fallback: bool = False
    fallback_reason: Optional[str] = None


class PyLingualStackSimulator:
    """Symbolic stack execution for Python bytecode basic blocks."""

    BINARY_OPS_312 = {
        0: "+", 1: "&", 2: "//", 3: "<<", 4: "@",
        5: "*", 6: "%", 7: "|", 8: "**", 9: ">>",
        10: "-", 11: "/", 12: "^",
        13: "+=", 14: "&=", 15: "//=", 16: "<<=",
        17: "@=", 18: "*=", 19: "%=", 20: "|=",
        21: "**=", 22: ">>=", 23: "-=", 24: "/=", 25: "^=",
    }

    LEGACY_BINARY_OPS = {
        "BINARY_ADD": "+",
        "BINARY_SUBTRACT": "-",
        "BINARY_MULTIPLY": "*",
        "BINARY_TRUE_DIVIDE": "/",
        "BINARY_FLOOR_DIVIDE": "//",
        "BINARY_MODULO": "%",
        "BINARY_POWER": "**",
        "BINARY_LSHIFT": "<<",
        "BINARY_RSHIFT": ">>",
        "BINARY_AND": "&",
        "BINARY_OR": "|",
        "BINARY_XOR": "^",
        "INPLACE_ADD": "+=",
        "INPLACE_SUBTRACT": "-=",
        "INPLACE_MULTIPLY": "*=",
        "INPLACE_TRUE_DIVIDE": "/=",
        "INPLACE_FLOOR_DIVIDE": "//=",
        "INPLACE_MODULO": "%=",
        "INPLACE_POWER": "**=",
        "INPLACE_LSHIFT": "<<=",
        "INPLACE_RSHIFT": ">>=",
        "INPLACE_AND": "&=",
        "INPLACE_OR": "|=",
        "INPLACE_XOR": "^=",
    }

    def __init__(self, version: Tuple[int, int]):
        self.version = version
        self.stack: List[str] = []
        self.statements: List[str] = []

    def push(self, val: str) -> None:
        self.stack.append(val)

    def pop(self) -> str:
        if self.stack:
            return self.stack.pop()
        return "None"

    def execute_instruction(self, inst: DisassembledInstruction) -> Optional[str]:
        """
        Symbolically executes a single instruction.
        Returns a statement if one was produced directly, or None.
        """
        op = inst.opname

        if op in ("CACHE", "RESUME", "PRECALL", "NOP", "END_FOR"):
            return None

        if op == "PUSH_NULL":
            self.push("__NULL__")
            return None

        if op == "LOAD_CONST":
            self.push(repr(inst.argval))
            return None

        if op in ("LOAD_NAME", "LOAD_FAST", "LOAD_GLOBAL", "LOAD_DEREF"):
            name = str(inst.argval)
            self.push(name)
            return None

        if op in ("STORE_NAME", "STORE_FAST", "STORE_GLOBAL", "STORE_DEREF"):
            name = str(inst.argval)
            val = self.pop()
            # Handle inplace augmented assignment match (e.g. total += i)
            if val.startswith(f"__INPLACE__{name} "):
                parts = val.split(" ", 2)
                op_sym = parts[1]
                right_val = parts[2]
                stmt = f"{name} {op_sym} {right_val}"
            elif val == name:
                # No-op store of same variable
                return None
            else:
                stmt = f"{name} = {val}"
            self.statements.append(stmt)
            return stmt

        if op == "LOAD_ATTR":
            attr = str(inst.argval)
            obj = self.pop()
            if obj == "__NULL__":
                obj = self.pop()
            self.push(f"{obj}.{attr}")
            return None

        if op == "STORE_ATTR":
            attr = str(inst.argval)
            obj = self.pop()
            val = self.pop()
            stmt = f"{obj}.{attr} = {val}"
            self.statements.append(stmt)
            return stmt

        if op == "BINARY_SUBSCR":
            idx = self.pop()
            container = self.pop()
            self.push(f"{container}[{idx}]")
            return None

        if op == "STORE_SUBSCR":
            idx = self.pop()
            container = self.pop()
            val = self.pop()
            stmt = f"{container}[{idx}] = {val}"
            self.statements.append(stmt)
            return stmt

        if op == "BINARY_OP":
            oparg = inst.arg if inst.arg is not None else 0
            symbol = self.BINARY_OPS_312.get(oparg, "+")
            right = self.pop()
            left = self.pop()
            if symbol.endswith("="):
                # Tag as inplace to be consumed by immediate STORE_*
                self.push(f"__INPLACE__{left} {symbol} {right}")
                return None
            else:
                self.push(f"({left} {symbol} {right})")
                return None

        if op in self.LEGACY_BINARY_OPS:
            symbol = self.LEGACY_BINARY_OPS[op]
            right = self.pop()
            left = self.pop()
            if symbol.endswith("="):
                self.push(f"__INPLACE__{left} {symbol} {right}")
                return None
            else:
                self.push(f"({left} {symbol} {right})")
                return None

        if op == "COMPARE_OP":
            cmp_symbol = inst.argrepr or str(inst.argval or "==")
            cmp_symbol = cmp_symbol.replace("to ", "").strip()
            if not cmp_symbol or cmp_symbol.isdigit():
                cmp_symbol = "=="
            right = self.pop()
            left = self.pop()
            self.push(f"({left} {cmp_symbol} {right})")
            return None

        if op == "IS_OP":
            right = self.pop()
            left = self.pop()
            symbol = "is not" if inst.arg == 1 else "is"
            self.push(f"({left} {symbol} {right})")
            return None

        if op == "CONTAINS_OP":
            right = self.pop()
            left = self.pop()
            symbol = "not in" if inst.arg == 1 else "in"
            self.push(f"({left} {symbol} {right})")
            return None

        if op in ("UNARY_NEGATIVE", "UNARY_NOT", "UNARY_INVERT"):
            val = self.pop()
            prefix = {"UNARY_NEGATIVE": "-", "UNARY_NOT": "not ", "UNARY_INVERT": "~"}[op]
            self.push(f"({prefix}{val})")
            return None

        if op in ("CALL", "CALL_FUNCTION", "CALL_METHOD"):
            n_args = inst.arg if inst.arg is not None else 0
            args = []
            for _ in range(n_args):
                args.append(self.pop())
            args.reverse()

            func = self.pop()
            if self.stack and self.stack[-1] == "__NULL__":
                self.stack.pop()

            call_str = f"{func}({', '.join(args)})"
            self.push(call_str)
            return None

        if op == "BUILD_LIST":
            n = inst.arg or 0
            items = [self.pop() for _ in range(n)][::-1]
            self.push(f"[{', '.join(items)}]")
            return None

        if op == "BUILD_TUPLE":
            n = inst.arg or 0
            items = [self.pop() for _ in range(n)][::-1]
            if len(items) == 1:
                self.push(f"({items[0]},)")
            else:
                self.push(f"({', '.join(items)})")
            return None

        if op == "BUILD_SET":
            n = inst.arg or 0
            items = [self.pop() for _ in range(n)][::-1]
            self.push(f"{{{', '.join(items)}}}")
            return None

        if op == "BUILD_MAP":
            n = inst.arg or 0
            entries = []
            for _ in range(n):
                val = self.pop()
                key = self.pop()
                entries.append(f"{key}: {val}")
            entries.reverse()
            self.push(f"{{{', '.join(entries)}}}")
            return None

        if op == "BUILD_CONST_KEY_MAP":
            keys_repr = self.pop()
            n = inst.arg or 0
            vals = [self.pop() for _ in range(n)][::-1]
            try:
                keys = ast.literal_eval(keys_repr)
                entries = [f"{repr(k)}: {v}" for k, v in zip(keys, vals)]
                self.push(f"{{{', '.join(entries)}}}")
            except Exception:
                self.push(f"dict(zip({keys_repr}, [{', '.join(vals)}]))")
            return None

        if op == "POP_TOP":
            if self.stack:
                val = self.pop()
                if "(" in val and not val.startswith("(") and val.endswith(")"):
                    self.statements.append(val)
                    return val
                elif not (val.startswith("(") and val.endswith(")")) and not val.isdigit():
                    if val != "__NULL__" and val != "None":
                        self.statements.append(val)
                        return val
            return None

        if op == "RETURN_VALUE":
            val = self.pop()
            if val != "None":
                stmt = f"return {val}"
                self.statements.append(stmt)
                return stmt
            return None

        if op == "RETURN_CONST":
            if inst.argval is not None:
                stmt = f"return {repr(inst.argval)}"
                self.statements.append(stmt)
                return stmt
            return None

        if op == "LIST_APPEND":
            val = self.pop()
            target = self.stack[-1] if self.stack else "lst"
            stmt = f"{target}.append({val})"
            self.statements.append(stmt)
            return stmt

        return None


class DecompilerPipeline:
    """Control-flow recovery pipeline translating bytecode to structured Python source."""

    def __init__(self, default_version: Optional[Tuple[int, int]] = None):
        self.disassembler = Disassembler(default_version=default_version)

    def decompile(
        self,
        target: Any,
        python_version: Optional[Any] = None,
    ) -> DecompiledResult:
        """
        Decompiles PyCode, code object, or bytecode into Python source code.

        Returns DecompiledResult containing reconstructed source code and CFG.
        """
        dis_code = self.disassembler.disassemble(target, python_version=python_version)
        cfg = ControlFlowGraph.build(dis_code.instructions)

        try:
            source = self._structure_and_emit(dis_code, cfg)
            parsed_ast = ast.parse(source)
            return DecompiledResult(
                source_code=source,
                cfg=cfg,
                instruction_stream=dis_code,
                version=dis_code.version,
                mode=DecompilationMode.DECOMPILED_SOURCE,
                ast_tree=parsed_ast,
                is_fallback=False,
            )
        except Exception as e:
            fallback_source = self._emit_disassembly_fallback(dis_code)
            return DecompiledResult(
                source_code=fallback_source,
                cfg=cfg,
                instruction_stream=dis_code,
                version=dis_code.version,
                mode=DecompilationMode.DISASSEMBLY_FALLBACK,
                is_fallback=True,
                fallback_reason=f"Pipeline structuring exception: {e}",
            )

    def _structure_and_emit(self, dis_code: DisassembledCode, cfg: ControlFlowGraph) -> str:
        """Transforms CFG and instructions into high-level structured Python source."""
        lines: List[str] = []

        # Emit nested functions first if any
        for nc in dis_code.nested_codes:
            sub_cfg = ControlFlowGraph.build(nc.instructions)
            sub_body = self._structure_and_emit(nc, sub_cfg)
            args_str = ", ".join(nc.varnames[: nc.argcount])
            lines.append(f"def {nc.name}({args_str}):")
            for line in sub_body.splitlines():
                lines.append(f"    {line}")
            lines.append("")

        sim = PyLingualStackSimulator(dis_code.version)
        visited_blocks: Set[int] = set()

        def emit_block(block_id: int, indent_level: int = 0, stop_block: Optional[int] = None) -> List[str]:
            if block_id not in cfg.blocks or block_id in visited_blocks or block_id == stop_block:
                return []
            visited_blocks.add(block_id)

            block = cfg.blocks[block_id]
            block_lines: List[str] = []
            indent = "    " * indent_level

            # Process block instructions sequentially
            last_inst: Optional[DisassembledInstruction] = None
            for inst in block.instructions:
                if inst.is_conditional_jump or inst.is_unconditional_jump:
                    last_inst = inst
                    break
                stmt = sim.execute_instruction(inst)
                if stmt:
                    block_lines.append(f"{indent}{stmt}")

            # Check if block is a loop header
            loop_match = next((lp for lp in cfg.loops if lp.header_id == block_id), None)
            if loop_match:
                if loop_match.loop_type == "for":
                    target_var = "item"
                    for b_id in sorted(loop_match.body_blocks - {block_id}):
                        b = cfg.blocks.get(b_id)
                        if b:
                            for inst in b.instructions:
                                if inst.opname in ("STORE_NAME", "STORE_FAST", "STORE_GLOBAL"):
                                    target_var = str(inst.argval)
                                    break
                            if target_var != "item":
                                break

                    iter_expr = sim.pop()
                    block_lines.append(f"{indent}for {target_var} in {iter_expr}:")

                    for b_id in sorted(loop_match.body_blocks - {block_id}):
                        b = cfg.blocks.get(b_id)
                        if b:
                            visited_blocks.add(b_id)
                            for inst in b.instructions:
                                if inst.opname in ("STORE_NAME", "STORE_FAST", "STORE_GLOBAL") and str(inst.argval) == target_var:
                                    continue
                                if inst.is_jump or inst.is_terminal:
                                    continue
                                stmt = sim.execute_instruction(inst)
                                if stmt:
                                    block_lines.append(f"{indent}    {stmt}")

                    exit_blocks = [s for s in block.successors if s not in loop_match.body_blocks]
                    for ex in exit_blocks:
                        for line in emit_block(ex, indent_level, stop_block=stop_block):
                            block_lines.append(line)
                    return block_lines

                else:
                    cond = sim.pop()
                    if cond in ("None", "__NULL__", ""):
                        cond = "True"
                    block_lines.append(f"{indent}while {cond}:")

                    for b_id in sorted(loop_match.body_blocks - {block_id}):
                        for line in emit_block(b_id, indent_level + 1, stop_block=stop_block):
                            block_lines.append(line)

                    exit_blocks = [s for s in block.successors if s not in loop_match.body_blocks]
                    for ex in exit_blocks:
                        for line in emit_block(ex, indent_level, stop_block=stop_block):
                            block_lines.append(line)
                    return block_lines

            # Handle conditional branches (if / else)
            if last_inst and last_inst.is_conditional_jump:
                cond = sim.pop()
                if not cond or cond in ("None", "__NULL__"):
                    cond = "True"

                block_lines.append(f"{indent}if {cond}:")

                true_block_id = block_id + 1
                false_offset = last_inst.jump_target
                false_block = next((b for b in cfg.blocks.values() if b.start_offset == false_offset), None)
                false_block_id = false_block.block_id if false_block else None

                # Check if false_block is a merge block (reachable from true_block)
                is_merge_block = False
                if false_block_id is not None and true_block_id in cfg.blocks:
                    q = [true_block_id]
                    seen = set()
                    while q:
                        curr = q.pop()
                        if curr == false_block_id:
                            is_merge_block = True
                            break
                        seen.add(curr)
                        if curr in cfg.blocks:
                            for s in cfg.blocks[curr].successors:
                                if s not in seen and s != block_id:
                                    q.append(s)

                if is_merge_block:
                    # if-without-else: true branch stops at false_block_id
                    true_lines = emit_block(true_block_id, indent_level + 1, stop_block=false_block_id)
                    if true_lines:
                        block_lines.extend(true_lines)
                    else:
                        block_lines.append(f"{indent}    pass")

                    if false_block_id is not None and false_block_id not in visited_blocks:
                        block_lines.extend(emit_block(false_block_id, indent_level, stop_block=stop_block))
                    return block_lines
                else:
                    # if-else: detect common merge successor if any
                    merge_id = None
                    if true_block_id in cfg.blocks and false_block_id in cfg.blocks:
                        true_succs = set(cfg.blocks[true_block_id].successors)
                        false_succs = set(cfg.blocks[false_block_id].successors)
                        common = true_succs.intersection(false_succs)
                        if common:
                            merge_id = next(iter(common))

                    true_lines = emit_block(true_block_id, indent_level + 1, stop_block=merge_id)
                    if true_lines:
                        block_lines.extend(true_lines)
                    else:
                        block_lines.append(f"{indent}    pass")

                    if false_block_id is not None and false_block_id not in visited_blocks:
                        block_lines.append(f"{indent}else:")
                        false_lines = emit_block(false_block_id, indent_level + 1, stop_block=merge_id)
                        if false_lines:
                            block_lines.extend(false_lines)
                        else:
                            block_lines.append(f"{indent}    pass")

                    if merge_id is not None and merge_id not in visited_blocks:
                        block_lines.extend(emit_block(merge_id, indent_level, stop_block=stop_block))

                    return block_lines

            for succ_id in block.successors:
                if succ_id not in visited_blocks and succ_id != stop_block:
                    block_lines.extend(emit_block(succ_id, indent_level, stop_block=stop_block))

            return block_lines

        main_lines = emit_block(cfg.entry_id, indent_level=0)
        lines.extend(main_lines)

        source_out = "\n".join(lines).strip()
        if not source_out:
            source_out = "pass"
        return source_out

    def _emit_disassembly_fallback(self, dis_code: DisassembledCode) -> str:
        """Emits structured disassembly fallback representation."""
        comment_header = "# [FALLBACK] Direct xdis instruction stream representation\n"
        instr_comments = []
        for inst in dis_code.instructions:
            instr_comments.append(f"# {inst.offset:>4d}: {inst.opname:<20s} {inst.argrepr or inst.argval or ''}")
        return comment_header + "\n".join(instr_comments) + "\npass\n"
