"""
renloc_core/static_graph/tarjan_scc.py
Tarjan's Strongly Connected Components (SCC) algorithm and Inter-Region DAG Condensation.
Detects gameplay loops (day/night cycles, free-roam map loops, repeating menus) and
condenses the ICFG into an acyclic inter-region Directed Acyclic Graph (DAG) to prevent
infinite recursion during static analysis while tagging loop states.
"""
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .icfg import ICFG, ICFGNode


class LoopKind(str, Enum):
    NONE = "NONE"
    DAY_NIGHT_LOOP = "DAY_NIGHT_LOOP"
    FREE_ROAM_LOOP = "FREE_ROAM_LOOP"
    MENU_LOOP = "MENU_LOOP"
    GENERAL_LOOP = "GENERAL_LOOP"


@dataclass
class SCCComponent:
    """
    Condensed meta-node representing a Strongly Connected Component.
    """
    scc_id: str
    nodes: Set[str] = field(default_factory=set)
    is_loop: bool = False
    loop_kind: LoopKind = LoopKind.NONE
    labels: Set[str] = field(default_factory=set)
    entry_nodes: Set[str] = field(default_factory=set)
    exit_nodes: Set[str] = field(default_factory=set)

    def __repr__(self) -> str:
        loop_str = f", loop={self.loop_kind.value}" if self.is_loop else ""
        return f"SCCComponent({self.scc_id}, size={len(self.nodes)}{loop_str}, labels={sorted(list(self.labels))})"


class CondensationDAG:
    """
    Acyclic condensation of the ICFG where each node is an SCCComponent.
    Guarantees no cycles exist across components.
    """
    def __init__(self):
        self.components: Dict[str, SCCComponent] = {}
        self.adj: Dict[str, Set[str]] = {}
        self.pred: Dict[str, Set[str]] = {}
        self.node_to_scc: Dict[str, str] = {}

    def add_component(self, comp: SCCComponent) -> None:
        self.components[comp.scc_id] = comp
        if comp.scc_id not in self.adj:
            self.adj[comp.scc_id] = set()
        if comp.scc_id not in self.pred:
            self.pred[comp.scc_id] = set()
        for nid in comp.nodes:
            self.node_to_scc[nid] = comp.scc_id

    def add_edge(self, src_scc: str, dst_scc: str) -> None:
        if src_scc == dst_scc:
            return  # Internal cycles are condensed away
        if src_scc not in self.adj:
            self.adj[src_scc] = set()
        if dst_scc not in self.pred:
            self.pred[dst_scc] = set()
        self.adj[src_scc].add(dst_scc)
        self.pred[dst_scc].add(src_scc)

    def get_successors(self, scc_id: str) -> List[str]:
        return sorted(list(self.adj.get(scc_id, set())))

    def get_predecessors(self, scc_id: str) -> List[str]:
        return sorted(list(self.pred.get(scc_id, set())))

    def get_component_for_node(self, node_id: str) -> Optional[SCCComponent]:
        scc_id = self.node_to_scc.get(node_id)
        return self.components.get(scc_id) if scc_id else None

    def is_node_in_loop(self, node_id: str) -> bool:
        comp = self.get_component_for_node(node_id)
        return comp.is_loop if comp else False

    def get_loop_components(self) -> List[SCCComponent]:
        return [c for c in self.components.values() if c.is_loop]

    def topological_sort(self) -> List[str]:
        """
        Kahn's algorithm to compute topological ordering of SCC components.
        Guaranteed to succeed because condensation eliminates all cycles.
        """
        in_degrees: Dict[str, int] = {cid: len(self.pred.get(cid, set())) for cid in self.components}
        queue = [cid for cid, deg in in_degrees.items() if deg == 0]
        topo_order: List[str] = []

        while queue:
            queue.sort()  # Deterministic order
            curr = queue.pop(0)
            topo_order.append(curr)

            for nxt in self.adj.get(curr, set()):
                in_degrees[nxt] -= 1
                if in_degrees[nxt] == 0:
                    queue.append(nxt)

        if len(topo_order) < len(self.components):
            # Fallback if any component had cycles (which cannot happen mathematically in an SCC condensation)
            remaining = [cid for cid in self.components if cid not in topo_order]
            topo_order.extend(sorted(remaining))

        return topo_order


class TarjanSCC:
    """
    Executes Tarjan's Strongly Connected Components algorithm on an ICFG
    and produces the condensation DAG with loop state tagging.
    """
    def __init__(self, icfg: ICFG):
        self.icfg: ICFG = icfg
        self._index: int = 0
        self._indices: Dict[str, int] = {}
        self._lowlink: Dict[str, int] = {}
        self._on_stack: Set[str] = {}
        self._stack: List[str] = []
        self._sccs: List[List[str]] = []

    def compute_sccs(self) -> List[List[str]]:
        """Finds all SCCs in the ICFG."""
        self._index = 0
        self._indices = {}
        self._lowlink = {}
        self._on_stack = set()
        self._stack = []
        self._sccs = []

        # Deterministic node traversal
        for node_id in sorted(self.icfg.all_node_ids()):
            if node_id not in self._indices:
                self._strongconnect(node_id)

        return self._sccs

    def _strongconnect(self, v: str) -> None:
        self._indices[v] = self._index
        self._lowlink[v] = self._index
        self._index += 1
        self._stack.append(v)
        self._on_stack.add(v)

        for w in self.icfg.get_successors(v):
            if w not in self._indices:
                self._strongconnect(w)
                self._lowlink[v] = min(self._lowlink[v], self._lowlink[w])
            elif w in self._on_stack:
                self._lowlink[v] = min(self._lowlink[v], self._indices[w])

        if self._lowlink[v] == self._indices[v]:
            scc: List[str] = []
            while True:
                w = self._stack.pop()
                self._on_stack.remove(w)
                scc.append(w)
                if w == v:
                    break
            self._sccs.append(scc)

    def condense(self) -> CondensationDAG:
        """
        Condenses the ICFG into an acyclic inter-region DAG of SCCComponents,
        tagging loop states and classifying loop kinds.
        """
        raw_sccs = self.compute_sccs()
        dag = CondensationDAG()

        # Build SCCComponent for each SCC
        for idx, scc_nodes in enumerate(raw_sccs):
            scc_id = f"SCC_{idx:03d}"
            node_set = set(scc_nodes)

            # Check if this SCC represents a loop:
            # size > 1, or size == 1 with a self-loop edge
            is_loop = len(node_set) > 1
            if not is_loop and len(node_set) == 1:
                single_node = next(iter(node_set))
                if single_node in self.icfg.get_successors(single_node):
                    is_loop = True

            labels = set()
            for nid in node_set:
                node = self.icfg.get_node(nid)
                if node and node.label:
                    labels.add(node.label)

            loop_kind = self._classify_loop(is_loop, labels, node_set)

            comp = SCCComponent(
                scc_id=scc_id,
                nodes=node_set,
                is_loop=is_loop,
                loop_kind=loop_kind,
                labels=labels,
            )
            dag.add_component(comp)

        # Wire condensation edges and determine component entry/exit nodes
        for u in self.icfg.all_node_ids():
            src_scc = dag.node_to_scc.get(u)
            if not src_scc:
                continue

            for v in self.icfg.get_successors(u):
                dst_scc = dag.node_to_scc.get(v)
                if not dst_scc:
                    continue

                if src_scc != dst_scc:
                    dag.add_edge(src_scc, dst_scc)
                    dag.components[src_scc].exit_nodes.add(u)
                    dag.components[dst_scc].entry_nodes.add(v)

        return dag

    def _classify_loop(self, is_loop: bool, labels: Set[str], node_ids: Set[str]) -> LoopKind:
        if not is_loop:
            return LoopKind.NONE

        raw_text = " ".join(list(labels) + list(node_ids))
        text_corpus = re.sub(r"[_\W]+", " ", raw_text).lower()

        # Day/Night loop keywords
        if re.search(r"\b(day|night|morning|evening|afternoon|noon|ngay|dem|sang|toi)\b", text_corpus):
            return LoopKind.DAY_NIGHT_LOOP

        # Free-roam map loop keywords
        if re.search(r"\b(map|roam|explore|hub|room|courtyard|travel|move|location|ban_do|thu_vien|library)\b", text_corpus):
            return LoopKind.FREE_ROAM_LOOP

        # Menu repeat loop keywords
        if re.search(r"\b(menu|choice|ask|retry|repeat|select|again|chon)\b", text_corpus):
            return LoopKind.MENU_LOOP

        return LoopKind.GENERAL_LOOP
