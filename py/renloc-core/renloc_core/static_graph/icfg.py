"""
renloc_core/static_graph/icfg.py
Interprocedural Control Flow Graph (ICFG) with k-CFA (k=2 call-string context sensitivity).
Each label is modeled as a procedure:
  - Direct jump: direct control-flow edge preserving current call string context.
  - Jump expression: dynamic jump edge with conservative target resolution.
  - Call label: pushes call site onto call stack (capped at length k=2).
  - Return: pops call stack and routes control back to the call site's return continuation.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from renloc_core.ast_reader.ast_nodes import (
    Call,
    Jump,
    Label,
    Return,
    Say,
    Node,
)
from .lifecycle import WrapNode


class EdgeType(str, Enum):
    SEQUENTIAL = "SEQUENTIAL"
    JUMP = "JUMP"
    DYNAMIC_JUMP = "DYNAMIC_JUMP"
    CALL = "CALL"
    RETURN = "RETURN"
    BRANCH = "BRANCH"
    FALLTHROUGH = "FALLTHROUGH"


CallContext = Tuple[str, ...]  # Call sites tuple of length <= k


@dataclass(frozen=True)
class ICFGNode:
    """
    Context-sensitive node in the Interprocedural Control Flow Graph.
    Identified uniquely by (label, statement_idx, call_context, sub_id).
    """
    node_id: str
    label: str
    index: int
    context: CallContext = ()
    ast_node: Any = None
    is_entry: bool = False
    is_exit: bool = False
    description: str = ""

    def __repr__(self) -> str:
        ctx_str = "::".join(self.context) if self.context else "root"
        return f"ICFGNode({self.node_id}, lbl={self.label}, idx={self.index}, ctx=[{ctx_str}])"


@dataclass
class ICFGEdge:
    """Directed edge between two context-sensitive ICFG nodes."""
    source: str
    target: str
    edge_type: EdgeType
    condition: Optional[str] = None
    call_site: Optional[str] = None

    def __repr__(self) -> str:
        cond_str = f" if '{self.condition}'" if self.condition else ""
        return f"ICFGEdge({self.source} -> {self.target}, type={self.edge_type.value}{cond_str})"


class ICFG:
    """
    Interprocedural Control Flow Graph representation with query methods.
    """
    def __init__(self, k: int = 2):
        self.k: int = k
        self.nodes: Dict[str, ICFGNode] = {}
        self.edges: List[ICFGEdge] = []
        self.adj: Dict[str, Set[str]] = {}
        self.pred: Dict[str, Set[str]] = {}
        self.edge_map: Dict[Tuple[str, str], List[ICFGEdge]] = {}
        self.label_entries: Dict[Tuple[str, CallContext], str] = {}
        self.label_exits: Dict[Tuple[str, CallContext], Set[str]] = {}

    def add_node(self, node: ICFGNode) -> None:
        self.nodes[node.node_id] = node
        if node.node_id not in self.adj:
            self.adj[node.node_id] = set()
        if node.node_id not in self.pred:
            self.pred[node.node_id] = set()

        if node.is_entry:
            self.label_entries[(node.label, node.context)] = node.node_id
        if node.is_exit:
            key = (node.label, node.context)
            if key not in self.label_exits:
                self.label_exits[key] = set()
            self.label_exits[key].add(node.node_id)

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType = EdgeType.SEQUENTIAL,
        condition: Optional[str] = None,
        call_site: Optional[str] = None,
    ) -> ICFGEdge:
        if source not in self.adj:
            self.adj[source] = set()
        if target not in self.pred:
            self.pred[target] = set()

        self.adj[source].add(target)
        self.pred[target].add(source)

        edge = ICFGEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            condition=condition,
            call_site=call_site,
        )
        self.edges.append(edge)
        key = (source, target)
        if key not in self.edge_map:
            self.edge_map[key] = []
        self.edge_map[key].append(edge)
        return edge

    def get_successors(self, node_id: str) -> List[str]:
        return sorted(list(self.adj.get(node_id, set())))

    def get_predecessors(self, node_id: str) -> List[str]:
        return sorted(list(self.pred.get(node_id, set())))

    def get_node(self, node_id: str) -> Optional[ICFGNode]:
        return self.nodes.get(node_id)

    def get_entry_node(self, label: str, context: CallContext = ()) -> Optional[str]:
        return self.label_entries.get((label, context))

    def get_labels(self) -> List[str]:
        labels = set(node.label for node in self.nodes.values() if node.label)
        return sorted(list(labels))

    def get_nodes_for_label(self, label: str) -> List[ICFGNode]:
        return [node for node in self.nodes.values() if node.label == label]

    def all_node_ids(self) -> Set[str]:
        return set(self.nodes.keys())


class ICFGBuilder:
    """
    Builds a k-CFA Context-Sensitive Interprocedural Control Flow Graph from Ren'Py AST.
    """
    def __init__(self, k: int = 2):
        self.k: int = k
        self.labels_stmts: Dict[str, List[Any]] = {}

    def extract_labels(self, statements: List[Any]) -> Dict[str, List[Any]]:
        """
        Groups AST / WrapNode statements into a mapping of {label_name: [statements]}.
        Handles both Label nodes with explicit .block attribute and flat sequences of statements.
        """
        labels_map: Dict[str, List[Any]] = {}
        current_label = "start"
        current_stmts: List[Any] = []

        for item in statements:
            raw = item.node if isinstance(item, WrapNode) else item
            if isinstance(raw, Label):
                if current_stmts:
                    if current_label not in labels_map:
                        labels_map[current_label] = []
                    labels_map[current_label].extend(current_stmts)
                    current_stmts = []

                current_label = getattr(raw, "name", "") or "unnamed_label"
                if current_label not in labels_map:
                    labels_map[current_label] = []

                # If Label node has inner block attached (Ren'Py 8.x AST structure)
                inner_block = getattr(raw, "block", None)
                if inner_block and isinstance(inner_block, (list, tuple)):
                    labels_map[current_label].extend(inner_block)
            else:
                current_stmts.append(item)

        if current_stmts:
            if current_label not in labels_map:
                labels_map[current_label] = []
            labels_map[current_label].extend(current_stmts)

        self.labels_stmts = labels_map
        return labels_map

    def build_from_labels(
        self,
        labels_map: Dict[str, List[Any]],
        entry_labels: Optional[List[str]] = None,
        dynamic_jump_targets: Optional[Dict[str, List[str]]] = None,
    ) -> ICFG:
        """
        Builds the k-CFA ICFG by traversing from entry labels under bounded call contexts.
        k=2: Call stack records up to 2 most recent call sites.
        """
        self.labels_stmts = labels_map
        icfg = ICFG(k=self.k)
        dynamic_targets = dynamic_jump_targets or {}

        if not entry_labels:
            if "start" in labels_map:
                entry_labels = ["start"]
            elif labels_map:
                entry_labels = [next(iter(labels_map.keys()))]
            else:
                return icfg

        # Worklist of (label, call_context) to expand
        worklist: List[Tuple[str, CallContext]] = [(lbl, ()) for lbl in entry_labels if lbl in labels_map]
        visited_procedures: Set[Tuple[str, CallContext]] = set()

        # Track call sites needing return wiring: (callee_label, callee_ctx) -> list of (return_continuation_node_id, caller_ctx)
        call_return_wiring: Dict[Tuple[str, CallContext], List[Tuple[str, CallContext]]] = {}

        while worklist:
            lbl, ctx = worklist.pop(0)
            if (lbl, ctx) in visited_procedures:
                continue
            visited_procedures.add((lbl, ctx))

            stmts = labels_map.get(lbl, [])
            if not stmts:
                # Empty procedure: create single entry/exit node
                node_id = self._node_id(lbl, 0, ctx)
                node = ICFGNode(
                    node_id=node_id,
                    label=lbl,
                    index=0,
                    context=ctx,
                    ast_node=None,
                    is_entry=True,
                    is_exit=True,
                    description=f"empty label {lbl}",
                )
                icfg.add_node(node)
                continue

            prev_nodes: List[str] = []

            for idx, stmt in enumerate(stmts):
                raw = stmt.node if isinstance(stmt, WrapNode) else stmt
                is_entry = (idx == 0)
                is_last = (idx == len(stmts) - 1)

                curr_id = self._node_id(lbl, idx, ctx)
                curr_node = ICFGNode(
                    node_id=curr_id,
                    label=lbl,
                    index=idx,
                    context=ctx,
                    ast_node=stmt,
                    is_entry=is_entry,
                    is_exit=is_last and not isinstance(raw, (Jump, Call)),
                    description=f"{type(raw).__name__} at {lbl}:{idx}",
                )
                icfg.add_node(curr_node)

                # Connect from predecessors within procedure
                for p_id in prev_nodes:
                    icfg.add_edge(p_id, curr_id, edge_type=EdgeType.SEQUENTIAL)
                prev_nodes = [curr_id]

                # Branch handling for Ren'Py menu / if statements
                menu_items = getattr(raw, "items", None)
                if menu_items and isinstance(menu_items, (list, tuple)):
                    # Ren'Py Menu: list of tuples (caption, condition, block)
                    choice_ends: List[str] = []
                    for c_idx, item in enumerate(menu_items):
                        if not isinstance(item, (list, tuple)) or len(item) < 3:
                            continue
                        caption, condition, choice_block = item[0], item[1], item[2]
                        if not choice_block:
                            choice_ends.append(curr_id)
                            continue

                        # Process choice sub-block
                        sub_prev = curr_id
                        for sub_i, sub_stmt in enumerate(choice_block):
                            sub_raw = sub_stmt.node if isinstance(sub_stmt, WrapNode) else sub_stmt
                            sub_id = f"{curr_id}_choice{c_idx}_{sub_i}"
                            sub_node = ICFGNode(
                                node_id=sub_id,
                                label=lbl,
                                index=idx,
                                context=ctx,
                                ast_node=sub_stmt,
                                description=f"MenuChoice '{caption}' step {sub_i}",
                            )
                            icfg.add_node(sub_node)
                            edge_type = EdgeType.BRANCH if sub_prev == curr_id else EdgeType.SEQUENTIAL
                            icfg.add_edge(
                                sub_prev,
                                sub_id,
                                edge_type=edge_type,
                                condition=caption if sub_prev == curr_id else None,
                            )
                            sub_prev = sub_id

                            # Check jump/call inside choice block
                            if isinstance(sub_raw, Jump):
                                target = getattr(sub_raw, "target", "")
                                self._wire_jump(icfg, sub_id, target, ctx, worklist, dynamic_targets, sub_raw)
                                sub_prev = None
                                break
                            elif isinstance(sub_raw, Call):
                                target = getattr(sub_raw, "label", "")
                                ret_id = f"{sub_id}_after_call"
                                self._wire_call(
                                    icfg, sub_id, target, ret_id, ctx, worklist, call_return_wiring
                                )
                                sub_prev = ret_id

                        if sub_prev is not None:
                            choice_ends.append(sub_prev)

                    if choice_ends:
                        prev_nodes = choice_ends
                    continue

                if_entries = getattr(raw, "entries", None)
                if if_entries and isinstance(if_entries, (list, tuple)):
                    # Ren'Py If: list of tuples (condition, block)
                    branch_ends: List[str] = []
                    for b_idx, (cond, b_block) in enumerate(if_entries):
                        if not b_block:
                            branch_ends.append(curr_id)
                            continue
                        sub_prev = curr_id
                        for sub_i, sub_stmt in enumerate(b_block):
                            sub_raw = sub_stmt.node if isinstance(sub_stmt, WrapNode) else sub_stmt
                            sub_id = f"{curr_id}_if{b_idx}_{sub_i}"
                            sub_node = ICFGNode(
                                node_id=sub_id,
                                label=lbl,
                                index=idx,
                                context=ctx,
                                ast_node=sub_stmt,
                                description=f"IfBranch '{cond}' step {sub_i}",
                            )
                            icfg.add_node(sub_node)
                            edge_type = EdgeType.BRANCH if sub_prev == curr_id else EdgeType.SEQUENTIAL
                            icfg.add_edge(
                                sub_prev,
                                sub_id,
                                edge_type=edge_type,
                                condition=str(cond) if sub_prev == curr_id else None,
                            )
                            sub_prev = sub_id

                            if isinstance(sub_raw, Jump):
                                target = getattr(sub_raw, "target", "")
                                self._wire_jump(icfg, sub_id, target, ctx, worklist, dynamic_targets, sub_raw)
                                sub_prev = None
                                break
                            elif isinstance(sub_raw, Call):
                                target = getattr(sub_raw, "label", "")
                                ret_id = f"{sub_id}_after_call"
                                self._wire_call(
                                    icfg, sub_id, target, ret_id, ctx, worklist, call_return_wiring
                                )
                                sub_prev = ret_id

                        if sub_prev is not None:
                            branch_ends.append(sub_prev)

                    if branch_ends:
                        prev_nodes = branch_ends
                    continue

                # Statement is Jump
                if isinstance(raw, Jump):
                    target = getattr(raw, "target", "")
                    self._wire_jump(icfg, curr_id, target, ctx, worklist, dynamic_targets, raw)
                    prev_nodes = []  # Unreachable after jump

                # Statement is Call
                elif isinstance(raw, Call):
                    target = getattr(raw, "label", "")
                    # Return continuation point in caller
                    ret_node_id = self._node_id(lbl, idx + 1, ctx) if idx + 1 < len(stmts) else f"{curr_id}_ret"
                    self._wire_call(
                        icfg, curr_id, target, ret_node_id, ctx, worklist, call_return_wiring
                    )
                    # Next statement in this procedure continues from ret_node_id
                    prev_nodes = [curr_id]

                # Statement is Return
                elif isinstance(raw, Return):
                    curr_node = ICFGNode(
                        node_id=curr_id,
                        label=lbl,
                        index=idx,
                        context=ctx,
                        ast_node=stmt,
                        is_exit=True,
                        description=f"Return in {lbl}",
                    )
                    icfg.add_node(curr_node)
                    # Wire return edges back to callers
                    if (lbl, ctx) in call_return_wiring:
                        for ret_target, caller_ctx in call_return_wiring[(lbl, ctx)]:
                            icfg.add_edge(
                                curr_id,
                                ret_target,
                                edge_type=EdgeType.RETURN,
                                call_site=ctx[-1] if ctx else None,
                            )
                    prev_nodes = []

            # Procedure exit nodes
            for p_id in prev_nodes:
                p_node = icfg.get_node(p_id)
                if p_node:
                    # Mark as exit
                    exit_node = ICFGNode(
                        node_id=p_node.node_id,
                        label=p_node.label,
                        index=p_node.index,
                        context=p_node.context,
                        ast_node=p_node.ast_node,
                        is_entry=p_node.is_entry,
                        is_exit=True,
                        description=p_node.description,
                    )
                    icfg.add_node(exit_node)

        # Wire any remaining call returns that were registered out-of-order
        for (callee_lbl, callee_ctx), ret_list in call_return_wiring.items():
            exits = icfg.label_exits.get((callee_lbl, callee_ctx), set())
            for exit_id in exits:
                for ret_target, caller_ctx in ret_list:
                    icfg.add_edge(
                        exit_id,
                        ret_target,
                        edge_type=EdgeType.RETURN,
                        call_site=callee_ctx[-1] if callee_ctx else None,
                    )

        return icfg

    def _node_id(self, label: str, index: int, context: CallContext) -> str:
        ctx_suffix = "::".join(context) if context else "0"
        return f"{label}#{index}@{ctx_suffix}"

    def _wire_jump(
        self,
        icfg: ICFG,
        curr_id: str,
        target: str,
        ctx: CallContext,
        worklist: List[Tuple[str, CallContext]],
        dynamic_targets: Dict[str, List[str]],
        raw_node: Any,
    ) -> None:
        is_expression = getattr(raw_node, "expression", False)
        if is_expression or not target or target.startswith("expression "):
            # Dynamic jump expression
            resolved = dynamic_targets.get(target, [])
            if not resolved:
                # Add a synthetic dynamic jump target node
                dyn_id = f"DYNAMIC_JUMP_{curr_id}"
                icfg.add_node(
                    ICFGNode(
                        node_id=dyn_id,
                        label="*DYNAMIC*",
                        index=0,
                        context=ctx,
                        description=f"Dynamic jump target for {target}",
                    )
                )
                icfg.add_edge(curr_id, dyn_id, edge_type=EdgeType.DYNAMIC_JUMP)
            else:
                for t in resolved:
                    t_entry = self._node_id(t, 0, ctx)
                    icfg.add_edge(curr_id, t_entry, edge_type=EdgeType.DYNAMIC_JUMP)
                    worklist.append((t, ctx))
        else:
            # Direct jump: same context
            target_entry_id = self._node_id(target, 0, ctx)
            icfg.add_edge(curr_id, target_entry_id, edge_type=EdgeType.JUMP)
            worklist.append((target, ctx))

    def _wire_call(
        self,
        icfg: ICFG,
        curr_id: str,
        target: str,
        ret_target_id: str,
        ctx: CallContext,
        worklist: List[Tuple[str, CallContext]],
        call_return_wiring: Dict[Tuple[str, CallContext], List[Tuple[str, CallContext]]],
    ) -> None:
        call_site_id = curr_id
        # k-CFA context push (capped at length k=2)
        new_ctx: CallContext = (ctx + (call_site_id,))[-self.k:]

        callee_entry_id = self._node_id(target, 0, new_ctx)
        icfg.add_edge(
            curr_id,
            callee_entry_id,
            edge_type=EdgeType.CALL,
            call_site=call_site_id,
        )

        wiring_key = (target, new_ctx)
        if wiring_key not in call_return_wiring:
            call_return_wiring[wiring_key] = []
        call_return_wiring[wiring_key].append((ret_target_id, ctx))

        worklist.append((target, new_ctx))
