"""
renloc_core/static_graph/honorific_graph.py
Temporal Honorific Graph (Đồ thị xưng hô thời gian).
Tracks dynamic character name assignments (e.g. initial '???' revealed as a real name at dominator node D).
Guarantees 100% of ancestors of D retain UNKNOWN_IDENTITY (strictly forbidding spoiler name leakage
backward to the beginning of the story).
Maintains directed temporal edges (speaker, listener, epoch) -> {pronoun_self, pronoun_other, formality_tier}.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .dominator import LengauerTarjanDominator
from .icfg import ICFG


UNKNOWN_IDENTITY = "???"


class FormalityTier(str, Enum):
    FORMAL = "FORMAL"       # Kính trọng / trang trọng (ngài, quý khách, tiểu thư)
    POLITE = "POLITE"       # Lịch sự / nhã nhặn (tôi - bạn, anh - em xã giao)
    CASUAL = "CASUAL"       # Thân mật bình thường (cậu - tớ, mình - bạn)
    INTIMATE = "INTIMATE"   # Rất thân mật / người yêu / ruột thịt (anh - em, chàng - thiếp)
    ARROGANT = "ARROGANT"   # trịch thượng / kẻ trên (ta - ngươi, bổn tọa)
    NEUTRAL = "NEUTRAL"     # Trung tính


@dataclass
class CharacterIdentity:
    """Tracks dynamic character identity state across the story timeline."""
    character_id: str
    initial_name: str = UNKNOWN_IDENTITY
    revealed_name: Optional[str] = None
    reveal_node: Optional[str] = None
    reveal_epoch: Optional[str] = None


@dataclass
class HonorificProfile:
    """Directed temporal honorific relationship between speaker and listener in an epoch."""
    speaker_id: str
    listener_id: str
    epoch_id: str
    pronoun_self: str                         # Đại từ nhân xưng ngôi 1 (tôi, anh, em, ta)
    pronoun_other: str                        # Đại từ nhân xưng ngôi 2 (bạn, em, anh, ngươi)
    formality_tier: FormalityTier = FormalityTier.POLITE
    evidence_span: str = ""                   # Câu thoại hoặc nguồn gốc thiết lập xưng hô

    def __repr__(self) -> str:
        return (
            f"HonorificProfile({self.speaker_id} -> {self.listener_id} in {self.epoch_id}: "
            f"self='{self.pronoun_self}', other='{self.pronoun_other}', {self.formality_tier.value})"
        )


class TemporalHonorificGraph:
    """
    Manages dynamic character identity resolution and temporal honorific edges.
    Enforces non-leakage of character real names back into story nodes preceding their reveal milestone.
    """
    def __init__(
        self,
        dominator_engine: Optional[LengauerTarjanDominator] = None,
        icfg: Optional[ICFG] = None,
    ):
        self.dominator: Optional[LengauerTarjanDominator] = dominator_engine
        self.icfg: Optional[ICFG] = icfg
        self.characters: Dict[str, CharacterIdentity] = {}
        # (speaker_id, listener_id, epoch_id) -> HonorificProfile
        self.profiles: Dict[Tuple[str, str, str], HonorificProfile] = {}
        # character_id -> Set of node_ids that are strictly ancestors of reveal_node D
        self._ancestors_cache: Dict[str, Set[str]] = {}

    def register_character(
        self,
        character_id: str,
        initial_name: str = UNKNOWN_IDENTITY,
    ) -> CharacterIdentity:
        """Registers a character with an initial (potentially unknown '???') identity."""
        identity = CharacterIdentity(
            character_id=character_id,
            initial_name=initial_name,
        )
        self.characters[character_id] = identity
        return identity

    def record_reveal(
        self,
        character_id: str,
        revealed_name: str,
        dominator_node: str,
        epoch_id: str = "post_reveal",
    ) -> None:
        """
        Records the real name reveal at dominator node D.
        Immediately computes and freezes the ancestor set of D to guarantee zero leakage.
        """
        if character_id not in self.characters:
            self.register_character(character_id, UNKNOWN_IDENTITY)

        char = self.characters[character_id]
        char.revealed_name = revealed_name
        char.reveal_node = dominator_node
        char.reveal_epoch = epoch_id

        # Compute all ancestors of dominator_node in the control flow graph
        ancestors = self._find_ancestors(dominator_node)
        self._ancestors_cache[character_id] = ancestors

    def _find_ancestors(self, target_node: str) -> Set[str]:
        """
        Traverses backward along incoming edges to find all nodes from which target_node is reachable.
        Excludes target_node itself.
        """
        ancestors: Set[str] = set()
        if not self.icfg and not self.dominator:
            return ancestors

        # Source graph predecessors
        pred_map: Dict[str, Set[str]] = {}
        if self.icfg:
            pred_map = self.icfg.pred
        elif self.dominator:
            pred_map = self.dominator.pred

        queue: List[str] = list(pred_map.get(target_node, set()))
        while queue:
            curr = queue.pop(0)
            if curr in ancestors or curr == target_node or curr == LengauerTarjanDominator.SUPER_ROOT:
                continue
            ancestors.add(curr)
            for p in pred_map.get(curr, set()):
                if p not in ancestors and p != target_node:
                    queue.append(p)

        return ancestors

    def get_character_name_at_node(self, character_id: str, node_id: str) -> str:
        """
        Resolves the display name of character_id at a specific graph node.
        STRICT NON-LEAKAGE RULE:
          - If node_id is in ancestors(D): returns UNKNOWN_IDENTITY (100% forbidden to leak real name).
          - If node_id is D or D is in Dom(node_id) (or node is a downstream descendant of D): returns revealed_name.
          - Otherwise, returns initial_name.
        """
        char = self.characters.get(character_id)
        if not char:
            return UNKNOWN_IDENTITY

        if not char.revealed_name or not char.reveal_node:
            return char.initial_name

        reveal_node = char.reveal_node
        ancestors = self._ancestors_cache.get(character_id, set())

        # Exact reveal node itself
        if node_id == reveal_node:
            return char.revealed_name

        # 1. 100% of ancestors of D MUST retain initial_name (UNKNOWN_IDENTITY)
        if node_id in ancestors:
            return char.initial_name

        # 2. Check if D dominates node_id (node is guaranteed downstream of reveal)
        if self.dominator and self.dominator.dominates(reveal_node, node_id):
            return char.revealed_name

        # 3. Check if node_id is reachable from D downstream
        if self._is_descendant(reveal_node, node_id):
            return char.revealed_name

        return char.initial_name

    def _is_descendant(self, start_node: str, query_node: str) -> bool:
        """True if query_node is reachable downstream from start_node."""
        if not self.icfg:
            return False
        visited: Set[str] = set()
        queue = [start_node]
        while queue:
            curr = queue.pop(0)
            if curr == query_node and curr != start_node:
                return True
            if curr in visited:
                continue
            visited.add(curr)
            for s in self.icfg.get_successors(curr):
                if s not in visited:
                    queue.append(s)
        return False

    def set_honorific_profile(
        self,
        speaker_id: str,
        listener_id: str,
        epoch_id: str,
        pronoun_self: str,
        pronoun_other: str,
        formality_tier: FormalityTier = FormalityTier.POLITE,
        evidence_span: str = "",
    ) -> HonorificProfile:
        """Sets a directed honorific profile for a speaker-listener pair in an epoch."""
        profile = HonorificProfile(
            speaker_id=speaker_id,
            listener_id=listener_id,
            epoch_id=epoch_id,
            pronoun_self=pronoun_self,
            pronoun_other=pronoun_other,
            formality_tier=formality_tier,
            evidence_span=evidence_span,
        )
        self.profiles[(speaker_id, listener_id, epoch_id)] = profile
        return profile

    def get_honorific_profile(
        self,
        speaker_id: str,
        listener_id: str,
        epoch_id: str,
    ) -> Optional[HonorificProfile]:
        """Retrieves honorific profile for (speaker, listener, epoch)."""
        return self.profiles.get((speaker_id, listener_id, epoch_id))

    def verify_no_leakage(self, character_id: str) -> Tuple[bool, List[str]]:
        """
        Formally audits all ancestor nodes of character's reveal milestone D.
        Verifies that 100% of ancestors resolve to UNKNOWN_IDENTITY ('???').
        Returns (is_leakage_free, list_of_leaked_nodes).
        """
        char = self.characters.get(character_id)
        if not char or not char.reveal_node or not char.revealed_name:
            return True, []

        ancestors = self._ancestors_cache.get(character_id, set())
        leaked_nodes: List[str] = []

        for anc_node in sorted(list(ancestors)):
            resolved_name = self.get_character_name_at_node(character_id, anc_node)
            if resolved_name == char.revealed_name:
                leaked_nodes.append(anc_node)

        return len(leaked_nodes) == 0, leaked_nodes
