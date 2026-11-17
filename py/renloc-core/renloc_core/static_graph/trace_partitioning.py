"""
renloc_core/static_graph/trace_partitioning.py
Path-Sensitive Trace Partitioning for relational gameplay variables.
Tracks abstract states of affection, route_id, player_name, menu choices, and condition flags
along parallel execution branches, keeping context versions strictly separated when a shared label
is reached under contrasting relational states.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from renloc_core.ast_reader.ast_nodes import Python, Say, UserStatement
from .icfg import EdgeType, ICFG, ICFGNode
from .lifecycle import WrapNode
from .tarjan_scc import CondensationDAG


@dataclass(frozen=True)
class PartitionKey:
    """Hashable token that defines trace partitioning boundaries."""
    route_id: str
    affection_bracket: str
    condition_signature: Tuple[str, ...]

    def __repr__(self) -> str:
        cond_str = ",".join(self.condition_signature[-2:]) if self.condition_signature else "none"
        return f"PKey(route='{self.route_id}', aff='{self.affection_bracket}', conds=[{cond_str}])"


@dataclass
class AbstractState:
    """
    Abstract state representing variable values and branch constraints along an execution trace.
    """
    variables: Dict[str, Any] = field(default_factory=dict)
    path_conditions: List[str] = field(default_factory=list)
    active_route: str = "common"
    menu_choices: List[str] = field(default_factory=list)
    loop_counters: Dict[str, int] = field(default_factory=dict)

    def clone(self) -> "AbstractState":
        return AbstractState(
            variables=deepcopy(self.variables),
            path_conditions=list(self.path_conditions),
            active_route=self.active_route,
            menu_choices=list(self.menu_choices),
            loop_counters=dict(self.loop_counters),
        )

    def get_var(self, name: str, default: Any = None) -> Any:
        return self.variables.get(name, default)

    def set_var(self, name: str, value: Any) -> None:
        self.variables[name] = value
        if name in ("route", "route_id", "current_route"):
            self.active_route = str(value)

    def update_affection(self, character: str, delta: int) -> None:
        key = f"affection_{character}" if not character.startswith("affection") else character
        curr = self.variables.get(key, 0)
        if isinstance(curr, (int, float)):
            self.variables[key] = curr + delta
        else:
            self.variables[key] = delta

    def get_affection_bracket(self, character: str = "main") -> str:
        """Discretizes numerical affection into symbolic brackets."""
        # Check specific character affection or general 'affection'
        val = self.variables.get(f"affection_{character}", self.variables.get("affection", 0))
        if not isinstance(val, (int, float)):
            return "UNKNOWN"
        if val < 5:
            return "LOW"
        elif val < 15:
            return "MEDIUM"
        else:
            return "HIGH"

    def compute_partition_key(self) -> PartitionKey:
        """Generates hashable partitioning token."""
        aff_bracket = self.get_affection_bracket()
        # Keep the last 3 path conditions to avoid state explosion while retaining path sensitivity
        cond_sig = tuple(self.path_conditions[-3:]) if self.path_conditions else ()
        return PartitionKey(
            route_id=self.active_route,
            affection_bracket=aff_bracket,
            condition_signature=cond_sig,
        )


@dataclass
class PartitionedTraceNode:
    """A node instantiated in a specific trace partition."""
    trace_id: str
    icfg_node_id: str
    state: AbstractState
    partition_key: PartitionKey


class TracePartitioningEngine:
    """
    Executes path-sensitive abstract interpretation over an ICFG with trace partitioning.
    Maintains separate context versions of labels reached with contrasting relational variables.
    """
    def __init__(
        self,
        icfg: ICFG,
        dag: Optional[CondensationDAG] = None,
        max_loop_iterations: int = 2,
    ):
        self.icfg: ICFG = icfg
        self.dag: Optional[CondensationDAG] = dag
        self.max_loop_iterations: int = max_loop_iterations
        # node_id -> {PartitionKey: AbstractState}
        self.partition_table: Dict[str, Dict[PartitionKey, AbstractState]] = {}
        # label -> List of distinct PartitionedTraceNode
        self.label_versions: Dict[str, List[PartitionedTraceNode]] = {}

    def run(
        self,
        entry_node_id: Optional[str] = None,
        initial_variables: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[PartitionKey, AbstractState]]:
        """
        Runs path-sensitive trace partitioning from entry node.
        """
        if entry_node_id is None:
            entry_node_id = self.icfg.get_entry_node("start", context=())
            if not entry_node_id and self.icfg.all_node_ids():
                entry_node_id = sorted(list(self.icfg.all_node_ids()))[0]

        if not entry_node_id:
            return {}

        start_state = AbstractState()
        if initial_variables:
            for k, v in initial_variables.items():
                start_state.set_var(k, v)

        # Worklist: (node_id, AbstractState)
        worklist: List[Tuple[str, AbstractState]] = [(entry_node_id, start_state)]
        visited_traces: Set[Tuple[str, PartitionKey]] = set()

        while worklist:
            curr_id, curr_state = worklist.pop(0)
            node = self.icfg.get_node(curr_id)
            if not node:
                continue

            # Check loop iteration bound if node is inside a loop SCC
            is_loop = self.dag.is_node_in_loop(curr_id) if self.dag else False
            if is_loop:
                count = curr_state.loop_counters.get(curr_id, 0)
                if count >= self.max_loop_iterations:
                    # Stabilize/widen: stop re-entering loop along this trace
                    continue
                curr_state.loop_counters[curr_id] = count + 1

            # Execute statement transfer function
            next_state = self._apply_transfer_function(node, curr_state)
            pkey = next_state.compute_partition_key()

            trace_marker = (curr_id, pkey)
            if trace_marker in visited_traces:
                # Compare state equality or join
                existing = self.partition_table.get(curr_id, {}).get(pkey)
                if existing and existing.variables == next_state.variables:
                    continue

            visited_traces.add(trace_marker)

            if curr_id not in self.partition_table:
                self.partition_table[curr_id] = {}
            self.partition_table[curr_id][pkey] = next_state

            # Track label version
            if node.label:
                if node.label not in self.label_versions:
                    self.label_versions[node.label] = []
                pnode = PartitionedTraceNode(
                    trace_id=f"{curr_id}@{pkey}",
                    icfg_node_id=curr_id,
                    state=next_state,
                    partition_key=pkey,
                )
                self.label_versions[node.label].append(pnode)

            # Traverse outgoing edges
            edges = self.icfg.edges
            succ_edges = [e for e in edges if e.source == curr_id]

            for edge in succ_edges:
                branch_state = next_state.clone()

                # Handle branch conditions
                if edge.condition:
                    branch_state.path_conditions.append(edge.condition)
                    # Check if condition modifies choices or route
                    self._interpret_condition(edge.condition, branch_state)

                worklist.append((edge.target, branch_state))

        return self.partition_table

    def _apply_transfer_function(self, node: ICFGNode, state: AbstractState) -> AbstractState:
        """
        Abstract transfer function interpreting Ren'Py statements modifying variables.
        """
        new_state = state.clone()
        ast_node = node.ast_node
        raw = ast_node.node if isinstance(ast_node, WrapNode) else ast_node

        if isinstance(raw, Python):
            code_obj = getattr(raw, "code", None)
            code_source = getattr(code_obj, "source", "") if code_obj else ""
            if not code_source and isinstance(code_obj, str):
                code_source = code_obj
            self._interpret_python_code(code_source, new_state)

        elif isinstance(raw, UserStatement):
            line = getattr(raw, "line", "")
            if line.startswith("$"):
                self._interpret_python_code(line[1:].strip(), new_state)

        return new_state

    def _interpret_python_code(self, code_str: str, state: AbstractState) -> None:
        """Simple pattern interpreter for variable assignments and affection updates."""
        if not code_str:
            return

        lines = code_str.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Affection += delta (e.g. affection += 5 or affection_alice += 2)
            m_inc = re.match(r"([a-zA-Z_]\w*)\s*\+=\s*(-?\d+)", line)
            if m_inc:
                var, delta = m_inc.group(1), int(m_inc.group(2))
                curr = state.get_var(var, 0)
                state.set_var(var, (curr if isinstance(curr, int) else 0) + delta)
                continue

            # Variable assignment (e.g. route = "alice" or affection = 10)
            m_assign = re.match(r"([a-zA-Z_]\w*)\s*=\s*(.+)", line)
            if m_assign:
                var, val_str = m_assign.group(1), m_assign.group(2).strip()
                # Parse literals
                if val_str.isdigit():
                    state.set_var(var, int(val_str))
                elif val_str.startswith('"') and val_str.endswith('"'):
                    state.set_var(var, val_str[1:-1])
                elif val_str.startswith("'") and val_str.endswith("'"):
                    state.set_var(var, val_str[1:-1])
                elif val_str in ("True", "False"):
                    state.set_var(var, val_str == "True")
                else:
                    state.set_var(var, val_str)

    def _interpret_condition(self, condition: str, state: AbstractState) -> None:
        """Extracts route selection or menu choices from branch conditions."""
        state.menu_choices.append(condition)

        # Route switch pattern in menu captions or condition: "Choose Alice" or "route == 'alice'"
        m_route = re.search(r"route\s*==\s*['\"](\w+)['\"]", condition, re.IGNORECASE)
        if m_route:
            state.active_route = m_route.group(1)
        elif "alice" in condition.lower():
            state.active_route = "alice"
        elif "bob" in condition.lower():
            state.active_route = "bob"
        elif "yuri" in condition.lower():
            state.active_route = "yuri"

    def get_context_versions(self, label: str) -> List[PartitionedTraceNode]:
        """Returns all distinct context versions for a given label."""
        return self.label_versions.get(label, [])

    def is_context_separated(self, label: str, var_name: str) -> bool:
        """
        True if the label is entered under >= 2 distinct context partitions with different values
        for the given variable (e.g. route_id or affection).
        """
        versions = self.get_context_versions(label)
        if len(versions) < 2:
            return False

        vals = set()
        for v in versions:
            val = v.state.get_var(var_name)
            vals.add(str(val))

        return len(vals) > 1
