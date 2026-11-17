"""
renloc_core/synthesizer/incremental_matcher.py
4-Tier Incremental Update Matcher for Ren'Py translations across game updates:
  Tier 1: Exact translation identifier match (Say.get_code() MD5 / UI string key)
  Tier 2: Structural AST & label match (same who, what, and label context despite display attribute edits)
  Tier 3: Fuzzy AST string similarity (Levenshtein distance / similarity >= 0.70 -> fast review queue)
  Tier 4: New translation units (queued for translation DAG scheduler)
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from renloc_core.static_graph.dialogue_normalizer import DialogueEntity


class MatchTier(str, Enum):
    TIER_1_EXACT_ID = "TIER_1_EXACT_ID"
    TIER_2_STRUCTURAL_AST = "TIER_2_STRUCTURAL_AST"
    TIER_3_FUZZY_SIMILARITY = "TIER_3_FUZZY_SIMILARITY"
    TIER_4_NEW_TASK = "TIER_4_NEW_TASK"


@dataclass
class IncrementalMatchResult:
    unit_id: str
    target_translation: Optional[str]
    tier: MatchTier
    confidence: float
    matched_old_id: Optional[str] = None
    similarity_score: float = 0.0
    notes: str = ""


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Computes normalized Levenshtein string similarity between 0.0 and 1.0."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    dist = dp[len1][len2]
    max_len = max(len1, len2)
    return max(0.0, 1.0 - (dist / max_len))


class IncrementalMatcher:
    """
    Compares AST dialogue entities from a new game release against the historical
    translation database across 4 distinct tiers.
    """

    FUZZY_THRESHOLD = 0.70

    def __init__(self, fuzzy_threshold: float = 0.70):
        self.fuzzy_threshold = fuzzy_threshold

    def match_units(
        self,
        new_units: List[DialogueEntity],
        old_corpus: Dict[str, Tuple[DialogueEntity, str]], # old_translation_id -> (old_entity, translation_text)
    ) -> List[IncrementalMatchResult]:
        """
        Matches a list of new DialogueEntity items against the prior translation corpus.
        """
        results: List[IncrementalMatchResult] = []

        # Index old units for fast lookup
        exact_id_map: Dict[str, Tuple[DialogueEntity, str]] = dict(old_corpus)
        structural_map: Dict[Tuple[Optional[str], str], List[Tuple[str, DialogueEntity, str]]] = {}

        for old_tlid, (old_entity, old_tl) in old_corpus.items():
            key = (old_entity.who, old_entity.what)
            if key not in structural_map:
                structural_map[key] = []
            structural_map[key].append((old_tlid, old_entity, old_tl))

        matched_old_ids: Set[str] = set()

        for new_u in new_units:
            tlid = new_u.translation_id

            # ---------------------------------------------------------
            # Tier 1: Exact ID Match
            # ---------------------------------------------------------
            if tlid in exact_id_map:
                old_ent, old_tl = exact_id_map[tlid]
                matched_old_ids.add(tlid)
                results.append(IncrementalMatchResult(
                    unit_id=new_u.entity_id,
                    target_translation=old_tl,
                    tier=MatchTier.TIER_1_EXACT_ID,
                    confidence=1.0,
                    matched_old_id=tlid,
                    similarity_score=1.0,
                    notes="100% exact translation ID match.",
                ))
                continue

            # ---------------------------------------------------------
            # Tier 2: Structural AST & Content Match (Minor Attribute Edits)
            # ---------------------------------------------------------
            struct_key = (new_u.who, new_u.what)
            if struct_key in structural_map and structural_map[struct_key]:
                candidates = structural_map[struct_key]
                # Pick first unmatched candidate if available
                cand_match = None
                for c_id, c_ent, c_tl in candidates:
                    if c_id not in matched_old_ids:
                        cand_match = (c_id, c_ent, c_tl)
                        break
                if not cand_match:
                    cand_match = candidates[0]

                c_id, c_ent, c_tl = cand_match
                matched_old_ids.add(c_id)
                results.append(IncrementalMatchResult(
                    unit_id=new_u.entity_id,
                    target_translation=c_tl,
                    tier=MatchTier.TIER_2_STRUCTURAL_AST,
                    confidence=0.95,
                    matched_old_id=c_id,
                    similarity_score=1.0,
                    notes="Structural match: identical dialogue text with altered display/transition attributes.",
                ))
                continue

            # ---------------------------------------------------------
            # Tier 3: Fuzzy AST String Similarity
            # ---------------------------------------------------------
            best_sim = 0.0
            best_match: Optional[Tuple[str, DialogueEntity, str]] = None

            for old_tlid, (old_ent, old_tl) in old_corpus.items():
                if old_tlid in matched_old_ids:
                    continue
                # Same speaker or both narrations
                if old_ent.who == new_u.who:
                    sim = levenshtein_similarity(new_u.what, old_ent.what)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = (old_tlid, old_ent, old_tl)

            if best_match and best_sim >= self.fuzzy_threshold:
                c_id, c_ent, c_tl = best_match
                matched_old_ids.add(c_id)
                results.append(IncrementalMatchResult(
                    unit_id=new_u.entity_id,
                    target_translation=c_tl,
                    tier=MatchTier.TIER_3_FUZZY_SIMILARITY,
                    confidence=round(best_sim, 3),
                    matched_old_id=c_id,
                    similarity_score=round(best_sim, 3),
                    notes=f"Fuzzy match ({best_sim:.1%}): queued for fast proofreading review.",
                ))
                continue

            # ---------------------------------------------------------
            # Tier 4: Completely New Translation Unit
            # ---------------------------------------------------------
            results.append(IncrementalMatchResult(
                unit_id=new_u.entity_id,
                target_translation=None,
                tier=MatchTier.TIER_4_NEW_TASK,
                confidence=0.0,
                matched_old_id=None,
                similarity_score=0.0,
                notes="New translation unit: dispatched to DAG task scheduler.",
            ))

        return results
