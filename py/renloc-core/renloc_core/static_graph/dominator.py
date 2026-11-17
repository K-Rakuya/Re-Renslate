"""
renloc_core/static_graph/dominator.py
Lengauer-Tarjan Dominator Tree algorithm on condensed DAGs and ICFGs.
Computes semidominators, immediate dominators (idom), and mandatory dominator sets Dom(v)
from root label 'start' (and secondary entry points via a virtual super-root).
Separates global story milestones from character branch divergences.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .icfg import ICFG
from .tarjan_scc import CondensationDAG, SCCComponent


class DominatorRole(str, Enum):
    SUPER_ROOT = "SUPER_ROOT"
    GLOBAL_MILESTONE = "GLOBAL_MILESTONE"     # Mandatory milestone traversed by all story paths
    BRANCH_DIVERGENCE = "BRANCH_DIVERGENCE"   # Character-specific or optional route divergence
    CONFLUENCE_JOIN = "CONFLUENCE_JOIN"       # Convergence point where branches reunite
    TERMINAL_ENDING = "TERMINAL_ENDING"       # Story termination / epilogue leaf node


@dataclass
class DominatorInfo:
    """Dominator properties of a single graph node."""
    node_id: str
    idom: Optional[str]                       # Immediate dominator
    dom_set: Set[str] = field(default_factory=set)  # Full mandatory dominator set
    dominated_children: List[str] = field(default_factory=list) # Children in dominator tree
    role: DominatorRole = DominatorRole.BRANCH_DIVERGENCE
    depth_in_dom_tree: int = 0


class LengauerTarjanDominator:
    """
    Computes immediate dominators and mandatory dominator sets using the Lengauer-Tarjan algorithm.
    Can operate on a CondensationDAG or directly on an ICFG / generic directed graph.
    """
    SUPER_ROOT = "__SUPER_ROOT__"

    def __init__(
        self,
        adj: Dict[str, Set[str]],
        pred: Dict[str, Set[str]],
        entry_nodes: List[str],
        all_nodes: Optional[Set[str]] = None,
    ):
        self.raw_adj: Dict[str, Set[str]] = {k: set(v) for k, v in adj.items()}
        self.raw_pred: Dict[str, Set[str]] = {k: set(v) for k, v in pred.items()}
        self.entry_nodes: List[str] = [e for e in entry_nodes if e]
        self.all_nodes: Set[str] = set(all_nodes or adj.keys()).union(pred.keys())

        # If multiple entries exist, attach virtual super-root
        self.use_super_root = len(self.entry_nodes) > 1 or (
            len(self.entry_nodes) == 1 and self.entry_nodes[0] != self.SUPER_ROOT
        )
        self._setup_graph_with_root()

        # Dominator computation state
        self.idom: Dict[str, Optional[str]] = {}
        self.dom_tree_children: Dict[str, List[str]] = {}
        self.dom_sets: Dict[str, Set[str]] = {}
        self.dominator_info: Dict[str, DominatorInfo] = {}

    def _setup_graph_with_root(self) -> None:
        self.adj = {k: set(v) for k, v in self.raw_adj.items()}
        self.pred = {k: set(v) for k, v in self.raw_pred.items()}

        if self.use_super_root:
            self.adj[self.SUPER_ROOT] = set(self.entry_nodes)
            self.pred[self.SUPER_ROOT] = set()
            for e in self.entry_nodes:
                if e not in self.pred:
                    self.pred[e] = set()
                self.pred[e].add(self.SUPER_ROOT)
            self.root = self.SUPER_ROOT
        else:
            self.root = self.entry_nodes[0] if self.entry_nodes else self.SUPER_ROOT

    @classmethod
    def from_condensation_dag(
        cls,
        dag: CondensationDAG,
        preferred_root_label: str = "start",
        secondary_labels: Optional[List[str]] = None,
    ) -> "LengauerTarjanDominator":
        """
        Builds dominator engine from an acyclic CondensationDAG, finding entry SCCs
        that contain 'start' and secondary entry points (e.g. 'splashscreen', 'main_menu').
        """
        secondary = set(secondary_labels or ["splashscreen", "main_menu", "before_main_menu"])
        primary_entries: List[str] = []
        secondary_entries: List[str] = []
        fallback_entries: List[str] = []

        for cid, comp in dag.components.items():
            if preferred_root_label in comp.labels:
                primary_entries.append(cid)
            elif comp.labels.intersection(secondary):
                secondary_entries.append(cid)
            elif len(dag.pred.get(cid, set())) == 0:
                fallback_entries.append(cid)

        entry_sccs = primary_entries or fallback_entries
        if secondary_entries:
            entry_sccs.extend([s for s in secondary_entries if s not in entry_sccs])

        return cls(
            adj=dag.adj,
            pred=dag.pred,
            entry_nodes=entry_sccs,
            all_nodes=set(dag.components.keys()),
        )

    @classmethod
    def from_icfg(
        cls,
        icfg: ICFG,
        preferred_root_label: str = "start",
        secondary_labels: Optional[List[str]] = None,
    ) -> "LengauerTarjanDominator":
        """
        Builds dominator engine directly from an ICFG.
        """
        secondary = set(secondary_labels or ["splashscreen", "main_menu", "before_main_menu"])
        entries: List[str] = []

        start_entry = icfg.get_entry_node(preferred_root_label, context=())
        if start_entry:
            entries.append(start_entry)

        for sec in secondary:
            sec_entry = icfg.get_entry_node(sec, context=())
            if sec_entry and sec_entry not in entries:
                entries.append(sec_entry)

        if not entries:
            # Fallback: find any node with 0 in-degree
            for nid in sorted(icfg.all_node_ids()):
                if len(icfg.get_predecessors(nid)) == 0:
                    entries.append(nid)
            if not entries and icfg.all_node_ids():
                entries = [sorted(icfg.all_node_ids())[0]]

        return cls(
            adj=icfg.adj,
            pred=icfg.pred,
            entry_nodes=entries,
            all_nodes=icfg.all_node_ids(),
        )

    def compute_dominators(self) -> Dict[str, Optional[str]]:
        """
        Executes the Lengauer-Tarjan algorithm:
          1. DFS preorder numbering
          2. Semi-dominator computation in reverse preorder
          3. Immediate dominator resolution
        """
        # Step 1: DFS preorder numbering
        dfnum: Dict[str, int] = {}
        vertex: List[str] = [self.SUPER_ROOT]  # 1-indexed placeholder
        parent: Dict[str, Optional[str]] = {self.root: None}
        semi: Dict[str, int] = {}
        ancestor: Dict[str, Optional[str]] = {}
        label_map: Dict[str, str] = {}

        def dfs(v: str) -> None:
            n = len(vertex)
            dfnum[v] = n
            vertex.append(v)
            label_map[v] = v
            ancestor[v] = None
            semi[v] = n

            # Deterministic traversal order
            for w in sorted(self.adj.get(v, set())):
                if w not in dfnum:
                    parent[w] = v
                    dfs(w)

        dfs(self.root)
        num_vertices = len(vertex) - 1

        def link(v: str, w: str) -> None:
            ancestor[w] = v

        def eval_node(v: str) -> str:
            # Path compression
            if ancestor[v] is None:
                return label_map[v]
            compress(v)
            return label_map[v]

        def compress(v: str) -> None:
            anc = ancestor[v]
            if anc is not None and ancestor[anc] is not None:
                compress(anc)
                if semi[label_map[anc]] < semi[label_map[v]]:
                    label_map[v] = label_map[anc]
                ancestor[v] = ancestor[anc]

        bucket: Dict[str, Set[str]] = {vertex[i]: set() for i in range(1, num_vertices + 1)}
        idom: Dict[str, Optional[str]] = {vertex[i]: None for i in range(1, num_vertices + 1)}

        # Step 2: Compute semidominators in reverse preorder
        for i in range(num_vertices, 1, -1):
            w = vertex[i]
            for v in self.pred.get(w, set()):
                if v not in dfnum:
                    continue  # Unreachable node from root
                u = eval_node(v)
                if semi[u] < semi[w]:
                    semi[w] = semi[u]

            semi_v = vertex[semi[w]]
            bucket[semi_v].add(w)
            p = parent[w]
            if p is not None:
                link(p, w)
                # Process vertices in parent's bucket
                if p in bucket:
                    for v in list(bucket[p]):
                        u = eval_node(v)
                        idom[v] = p if semi[u] == semi[v] else u
                    bucket[p].clear()

        # Step 3: Explicitly adjust idom in forward preorder
        for i in range(2, num_vertices + 1):
            w = vertex[i]
            if idom[w] != vertex[semi[w]]:
                idom[w] = idom[idom[w]] if idom[w] in idom else idom[w]

        idom[self.root] = None
        self.idom = idom

        # Build dominator tree children mapping
        self.dom_tree_children = {v: [] for v in vertex[1:]}
        for v in vertex[2:]:
            p = idom.get(v)
            if p and p in self.dom_tree_children:
                self.dom_tree_children[p].append(v)

        # Step 4: Compute full mandatory dominator sets Dom(v)
        self._compute_dominator_sets(vertex[1:])

        # Step 5: Classify global milestones vs branch divergences
        self._classify_roles(vertex[1:])

        return self.idom

    def _compute_dominator_sets(self, reachable_nodes: List[str]) -> None:
        self.dom_sets = {}
        for node in reachable_nodes:
            curr: Optional[str] = node
            doms: Set[str] = set()
            while curr is not None:
                doms.add(curr)
                curr = self.idom.get(curr)
            self.dom_sets[node] = doms

    def _classify_roles(self, reachable_nodes: List[str]) -> None:
        """
        Classifies each node as:
          - GLOBAL_MILESTONE: mandatory story milestone (dominates all reachable endings or main narrative confluence)
          - BRANCH_DIVERGENCE: divergent branch node on character routes
          - CONFLUENCE_JOIN: join point of branches
          - TERMINAL_ENDING: exit/leaf ending
        """
        # Find terminal leaf nodes (nodes with 0 successors in reachable set)
        terminals = [n for n in reachable_nodes if n != self.SUPER_ROOT and len(self.adj.get(n, set())) == 0]
        if not terminals:
            # If graph is cyclic or ends in loop, take nodes with minimal out-degrees
            terminals = reachable_nodes[-1:]

        for node in reachable_nodes:
            if node == self.SUPER_ROOT:
                role = DominatorRole.SUPER_ROOT
            elif node in terminals:
                role = DominatorRole.TERMINAL_ENDING
            elif all(node in self.dom_sets.get(t, set()) for t in terminals) and len(terminals) > 0:
                role = DominatorRole.GLOBAL_MILESTONE
            elif len(self.pred.get(node, set())) > 1 and len(self.adj.get(node, set())) <= 1:
                role = DominatorRole.CONFLUENCE_JOIN
            elif len(self.adj.get(node, set())) > 1:
                role = DominatorRole.BRANCH_DIVERGENCE
            else:
                # Check depth and whether it dominates a significant subtree
                descendants_count = len(self._get_all_dominated(node))
                if descendants_count >= len(reachable_nodes) * 0.4:
                    role = DominatorRole.GLOBAL_MILESTONE
                else:
                    role = DominatorRole.BRANCH_DIVERGENCE

            # Depth in dominator tree
            depth = len(self.dom_sets.get(node, set()))

            self.dominator_info[node] = DominatorInfo(
                node_id=node,
                idom=self.idom.get(node),
                dom_set=self.dom_sets.get(node, set()),
                dominated_children=self.dom_tree_children.get(node, []),
                role=role,
                depth_in_dom_tree=depth,
            )

    def dominates(self, u: str, v: str) -> bool:
        """True if u dominates v (every path from root to v must pass through u)."""
        return u in self.dom_sets.get(v, set())

    def strictly_dominates(self, u: str, v: str) -> bool:
        """True if u dominates v and u != v."""
        return u != v and self.dominates(u, v)

    def get_mandatory_dominators(self, v: str) -> Set[str]:
        """Returns the full mandatory dominator set Dom(v)."""
        doms = set(self.dom_sets.get(v, set()))
        doms.discard(self.SUPER_ROOT)
        return doms

    def get_global_milestones(self) -> List[str]:
        """Returns all nodes classified as mandatory global story milestones."""
        return [
            nid for nid, info in self.dominator_info.items()
            if info.role == DominatorRole.GLOBAL_MILESTONE and nid != self.SUPER_ROOT
        ]

    def get_divergence_branches(self, branch_fork_node: str) -> Dict[str, Set[str]]:
        """
        Given a branch divergence node (e.g. menu choice or if-fork), partitions
        the downstream nodes into isolated branch sets.
        """
        succs = sorted(list(self.adj.get(branch_fork_node, set())))
        branch_sets: Dict[str, Set[str]] = {}

        for succ in succs:
            visited: Set[str] = set()
            queue = [succ]
            while queue:
                curr = queue.pop(0)
                if curr in visited or curr == branch_fork_node:
                    continue
                visited.add(curr)
                for nxt in self.adj.get(curr, set()):
                    if nxt not in visited:
                        queue.append(nxt)
            branch_sets[succ] = visited

        return branch_sets

    def _get_all_dominated(self, u: str) -> Set[str]:
        """Collects all nodes in the dominator subtree rooted at u."""
        res: Set[str] = set()
        stack = [u]
        while stack:
            curr = stack.pop()
            res.add(curr)
            for child in self.dom_tree_children.get(curr, []):
                if child not in res:
                    stack.append(child)
        res.discard(u)
        return res
