"""
renloc_core/static_graph/lifecycle.py
Simulates Ren'Py's 5-phase initialization lifecycle:
  Phase 1: python early (runs before custom statement compilation)
  Phase 2: init [offset] blocks (sorted ascending by integer priority, e.g. -999 to +999)
  Phase 3: default statements (initializes gameplay state variables)
  Phase 4: entry point labels (splashscreen, main_menu, etc.)
  Phase 5: story flow starting from label start
Maintains WrapNode location tracking (filename, linenumber, block_depth, phase).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from renloc_core.ast_reader.ast_nodes import (
    EarlyPython,
    Init,
    Default,
    Define,
    Label,
    Python,
    Say,
    Screen,
    Node,
)


class LifecyclePhase(Enum):
    EARLY_PYTHON = 1
    INIT = 2
    DEFAULT = 3
    ENTRY_POINT = 4
    START_STORY = 5


@dataclass
class WrapNode:
    """
    Metadata container that wraps AST nodes, maintaining precise source file,
    line number, nesting depth, and execution lifecycle phase.
    """
    node: Any
    filename: str = ""
    linenumber: int = 0
    block_depth: int = 0
    phase: LifecyclePhase = LifecyclePhase.START_STORY
    priority: int = 0
    ast_id: str = ""

    def __repr__(self) -> str:
        node_name = type(self.node).__name__
        return (
            f"WrapNode({node_name}, phase={self.phase.name}, priority={self.priority}, "
            f"loc={self.filename}:{self.linenumber}, depth={self.block_depth})"
        )


class LifecycleSimulator:
    """
    Simulates the strict 5-phase initialization and execution order of the Ren'Py engine.
    Ensures character definitions, default state variables, and entry points are resolved
    in the exact deterministic order matching the Ren'Py SDK runtime.
    """

    KNOWN_ENTRY_POINTS = {
        "splashscreen",
        "before_main_menu",
        "main_menu",
        "after_load",
    }

    def __init__(self):
        self.wrapped_nodes: List[WrapNode] = []
        self.phase_map: Dict[LifecyclePhase, List[WrapNode]] = {phase: [] for phase in LifecyclePhase}
        self.defined_characters: Dict[str, Any] = {}
        self.default_variables: Dict[str, Any] = {}
        self.early_flags: Set[str] = set()

    def wrap_ast(
        self,
        nodes: List[Any],
        filename: str = "script.rpy",
        current_depth: int = 0,
    ) -> List[WrapNode]:
        """
        Recursively wraps a sequence of AST nodes, determining phase, location,
        and nesting depth for each node.
        """
        results: List[WrapNode] = []
        node_counter = 0

        for item in nodes:
            node_counter += 1
            file_loc = getattr(item, "filename", "") or filename
            line_loc = getattr(item, "linenumber", 0)

            # 1. EarlyPython (Phase 1)
            if isinstance(item, EarlyPython):
                w = WrapNode(
                    node=item,
                    filename=file_loc,
                    linenumber=line_loc,
                    block_depth=current_depth,
                    phase=LifecyclePhase.EARLY_PYTHON,
                    priority=-99999,
                    ast_id=f"{file_loc}_{line_loc}_{node_counter}",
                )
                results.append(w)
                self.phase_map[LifecyclePhase.EARLY_PYTHON].append(w)
                continue

            # 2. Init Block or Define (Phase 2)
            if isinstance(item, Init):
                prio = getattr(item, "priority", 0)
                w = WrapNode(
                    node=item,
                    filename=file_loc,
                    linenumber=line_loc,
                    block_depth=current_depth,
                    phase=LifecyclePhase.INIT,
                    priority=prio,
                    ast_id=f"{file_loc}_{line_loc}_{node_counter}",
                )
                results.append(w)
                self.phase_map[LifecyclePhase.INIT].append(w)

                # Recursively wrap inner block
                inner_block = getattr(item, "block", [])
                if inner_block:
                    inner_wrapped = self.wrap_ast(inner_block, filename=file_loc, current_depth=current_depth + 1)
                    for iw in inner_wrapped:
                        iw.phase = LifecyclePhase.INIT
                        iw.priority = prio
                    results.extend(inner_wrapped)
                continue

            if isinstance(item, (Define, Screen)):
                # Define and Screen statements in Ren'Py run in init phase (default priority 0)
                prio = getattr(item, "priority", 0)
                w = WrapNode(
                    node=item,
                    filename=file_loc,
                    linenumber=line_loc,
                    block_depth=current_depth,
                    phase=LifecyclePhase.INIT,
                    priority=prio,
                    ast_id=f"{file_loc}_{line_loc}_{node_counter}",
                )
                results.append(w)
                self.phase_map[LifecyclePhase.INIT].append(w)
                continue

            # 3. Default statements (Phase 3)
            if isinstance(item, Default):
                w = WrapNode(
                    node=item,
                    filename=file_loc,
                    linenumber=line_loc,
                    block_depth=current_depth,
                    phase=LifecyclePhase.DEFAULT,
                    priority=0,
                    ast_id=f"{file_loc}_{line_loc}_{node_counter}",
                )
                results.append(w)
                self.phase_map[LifecyclePhase.DEFAULT].append(w)
                continue

            # 4. Labels: Entry points vs Main Story (Phase 4 vs Phase 5)
            if isinstance(item, Label):
                lbl_name = getattr(item, "name", "")
                if lbl_name in self.KNOWN_ENTRY_POINTS:
                    phase = LifecyclePhase.ENTRY_POINT
                else:
                    phase = LifecyclePhase.START_STORY

                w = WrapNode(
                    node=item,
                    filename=file_loc,
                    linenumber=line_loc,
                    block_depth=current_depth,
                    phase=phase,
                    priority=0,
                    ast_id=f"{file_loc}_{line_loc}_{node_counter}",
                )
                results.append(w)
                self.phase_map[phase].append(w)

                # Recurse inner block if present
                inner_block = getattr(item, "block", [])
                if inner_block:
                    inner_wrapped = self.wrap_ast(inner_block, filename=file_loc, current_depth=current_depth + 1)
                    for iw in inner_wrapped:
                        iw.phase = phase
                    results.extend(inner_wrapped)
                continue

            # 5. General statements (Say, Python, Show, Menu, etc.)
            w = WrapNode(
                node=item,
                filename=file_loc,
                linenumber=line_loc,
                block_depth=current_depth,
                phase=LifecyclePhase.START_STORY,
                priority=0,
                ast_id=f"{file_loc}_{line_loc}_{node_counter}",
            )
            results.append(w)
            self.phase_map[LifecyclePhase.START_STORY].append(w)

        self.wrapped_nodes.extend(results)
        return results

    def get_execution_order(self) -> List[WrapNode]:
        """
        Returns all wrapped nodes sorted strictly by Ren'Py's 5-phase initialization order:
          1. Phase 1: python early
          2. Phase 2: init [offset] sorted ascending by priority (-999, -1, 0, 1, 999)
          3. Phase 3: default statements
          4. Phase 4: entry points (splashscreen, main_menu)
          5. Phase 5: main narrative flow (start)
        """
        ordered: List[WrapNode] = []

        # 1. Phase 1: Early Python
        ordered.extend(self.phase_map[LifecyclePhase.EARLY_PYTHON])

        # 2. Phase 2: Init blocks sorted by priority offset
        init_nodes = sorted(self.phase_map[LifecyclePhase.INIT], key=lambda w: w.priority)
        ordered.extend(init_nodes)

        # 3. Phase 3: Default statements
        ordered.extend(self.phase_map[LifecyclePhase.DEFAULT])

        # 4. Phase 4: Entry points
        ordered.extend(self.phase_map[LifecyclePhase.ENTRY_POINT])

        # 5. Phase 5: Main story flow
        ordered.extend(self.phase_map[LifecyclePhase.START_STORY])

        return ordered

    def simulate_state(self) -> Dict[str, Any]:
        """
        Simulates environmental state after Phase 1, Phase 2, and Phase 3:
          - Records defined characters and global definitions
          - Records default gameplay variables
        """
        for w in self.get_execution_order():
            node = w.node
            if isinstance(node, Define):
                varname = getattr(node, "varname", "")
                code = getattr(node, "code", None)
                if varname:
                    self.defined_characters[varname] = code

            elif isinstance(node, Default):
                varname = getattr(node, "varname", "")
                code = getattr(node, "code", None)
                if varname:
                    self.default_variables[varname] = code

            elif isinstance(node, EarlyPython):
                self.early_flags.add(w.ast_id)

        return {
            "defined_characters": self.defined_characters,
            "default_variables": self.default_variables,
            "early_flags": self.early_flags,
        }
