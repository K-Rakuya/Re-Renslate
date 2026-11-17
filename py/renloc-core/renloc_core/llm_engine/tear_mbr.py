"""
tear_mbr.py
Implementation of Minimum Bayes Risk (MBR) Decoding and the
TEaR (Translate -> Estimate -> Refine) Quality Optimization Cycle.

Key Features:
1. MBR (Minimum Bayes Risk) decoding with sample size K in [4, 8].
   - Pairwise risk matrix with chrF++ and LCS utility metrics.
   - Optimal pseudo-reference consensus candidate selection.
2. TEaR Cycle:
   - Translate: Generates candidates (optionally through MBR).
   - Estimate: Multi-factor Quality Estimation (COMET-QE client, tag preservation, length ratio, untranslated leakage).
   - Refine: Iterative feedback-guided refinement when QE score is below threshold.
   - Guaranteed deterministic tag recovery.
"""

from __future__ import annotations

import collections
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from renloc_core.llm_engine.constraint_compiler import safe_recover_tags
from renloc_core.llm_engine.interpolation_lexer import (
    PlaceholderMap,
    mask_substitutions,
    restore_substitutions,
    validate_placeholders,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Utility Metrics for MBR (chrF++ & LCS)
# ==============================================================================

def get_char_ngrams(text: str, n: int) -> collections.Counter[str]:
    """Extracts character n-grams from text."""
    # Remove space collapsing if desired, but keep single spaces for boundary n-grams
    clean = re.sub(r"\s+", " ", text)
    ngrams: collections.Counter[str] = collections.Counter()
    for i in range(len(clean) - n + 1):
        ngrams[clean[i : i + n]] += 1
    return ngrams


def compute_chrf(
    hypothesis: str,
    reference: str,
    max_n: int = 6,
    beta: float = 2.0,
) -> float:
    """
    Computes chrF score (character n-gram F-score) between hypothesis and reference.
    Standard utility function for MBR decoding in MT.
    Returns float in [0.0, 1.0].
    """
    if hypothesis == reference:
        return 1.0
    if not hypothesis or not reference:
        return 0.0

    precisions: List[float] = []
    recalls: List[float] = []

    for n in range(1, max_n + 1):
        hyp_ngrams = get_char_ngrams(hypothesis, n)
        ref_ngrams = get_char_ngrams(reference, n)

        total_hyp = sum(hyp_ngrams.values())
        total_ref = sum(ref_ngrams.values())

        if total_hyp == 0 or total_ref == 0:
            continue

        # Intersection of n-grams
        common_count = 0
        for ng, count in hyp_ngrams.items():
            if ng in ref_ngrams:
                common_count += min(count, ref_ngrams[ng])

        prec = common_count / total_hyp if total_hyp > 0 else 0.0
        rec = common_count / total_ref if total_ref > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)

    if not precisions or not recalls:
        return 0.0

    avg_prec = sum(precisions) / len(precisions)
    avg_rec = sum(recalls) / len(recalls)

    if avg_prec + avg_rec == 0.0:
        return 0.0

    # F_beta score
    beta_sq = beta * beta
    f_beta = (1.0 + beta_sq) * (avg_prec * avg_rec) / (beta_sq * avg_prec + avg_rec)
    return min(1.0, max(0.0, f_beta))


def compute_lcs_similarity(s1: str, s2: str) -> float:
    """Computes LCS similarity between two strings."""
    if s1 == s2:
        return 1.0
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return 0.0

    # Space optimized DP
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        c1 = s1[i - 1]
        for j in range(1, m + 1):
            if c1 == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (m + 1)

    lcs_len = prev[m]
    return (2.0 * lcs_len) / (n + m)


# ==============================================================================
# 2. Minimum Bayes Risk (MBR) Decoding (K = 4..8)
# ==============================================================================

@dataclass
class MBRResult:
    """Result of Minimum Bayes Risk selection."""
    best_candidate: str
    best_index: int
    best_score: float
    candidate_scores: List[float]
    utility_matrix: List[List[float]]
    sample_size: int


def minimum_bayes_risk(
    candidates: Sequence[str],
    utility_metric: str = "chrf",
    custom_utility_fn: Optional[Callable[[str, str], float]] = None,
) -> MBRResult:
    """
    Executes Minimum Bayes Risk (MBR) decoding over K sampled candidates (typically K=4..8).
    Treats each candidate as a pseudo-reference and computes expected utility:
        Score(y_i) = 1 / (K - 1) * sum_{j != i} utility(y_i, y_j)
    Selects y* = argmax Score(y_i).
    """
    K = len(candidates)
    if K == 0:
        raise ValueError("Cannot perform MBR decoding on empty candidate list.")
    if K == 1:
        return MBRResult(
            best_candidate=candidates[0],
            best_index=0,
            best_score=1.0,
            candidate_scores=[1.0],
            utility_matrix=[[1.0]],
            sample_size=1,
        )

    # Select utility function
    if custom_utility_fn is not None:
        u_fn = custom_utility_fn
    elif utility_metric == "lcs":
        u_fn = compute_lcs_similarity
    else:
        u_fn = compute_chrf

    # Compute pairwise utility matrix
    matrix = [[0.0] * K for _ in range(K)]
    for i in range(K):
        matrix[i][i] = 1.0
        for j in range(i + 1, K):
            u = u_fn(candidates[i], candidates[j])
            matrix[i][j] = u
            matrix[j][i] = u

    # Compute expected utility per candidate
    scores: List[float] = []
    for i in range(K):
        # Exclude self-comparison
        other_sum = sum(matrix[i][j] for j in range(K) if j != i)
        expected_u = other_sum / (K - 1)
        scores.append(expected_u)

    best_idx = max(range(K), key=lambda idx: scores[idx])

    return MBRResult(
        best_candidate=candidates[best_idx],
        best_index=best_idx,
        best_score=scores[best_idx],
        candidate_scores=scores,
        utility_matrix=matrix,
        sample_size=K,
    )


# ==============================================================================
# 3. Quality Estimation (Estimate Step)
# ==============================================================================

@dataclass
class QualityEstimate:
    """Output of translation quality estimation."""
    score: float
    is_acceptable: bool
    issues: List[str] = field(default_factory=list)
    comet_score: Optional[float] = None
    tag_score: float = 1.0
    length_ratio: float = 1.0


# Regex detecting remaining raw Japanese kana/kanji (leakage)
JAPANESE_CHAR_REGEX = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
# Regex detecting degenerate repetitions
REPETITION_REGEX = re.compile(r"(.{4,})\1{3,}")


def estimate_translation_quality(
    source_text: str,
    translation: str,
    expected_placeholders: Optional[Sequence[str]] = None,
    qe_client: Optional[Any] = None,
    threshold: float = 0.85,
) -> QualityEstimate:
    """
    Evaluates translation quality across 4 dimensions:
    1. COMET-QE score via qe_client (or heuristic proxy).
    2. Invariant tag and placeholder preservation.
    3. Length ratio consistency.
    4. Untranslated character leakage & degenerate repetition check.
    """
    issues: List[str] = []
    penalties = 0.0

    # 1. Tag / placeholder preservation
    tag_score = 1.0
    if expected_placeholders:
        missing_tags = [ph for ph in expected_placeholders if ph not in translation]
        if missing_tags:
            tag_score = max(0.0, 1.0 - (len(missing_tags) / len(expected_placeholders)))
            issues.append(f"missing_placeholders:{','.join(missing_tags)}")
            penalties += (1.0 - tag_score) * 0.4

    # 2. Length ratio check
    len_src = max(1, len(source_text.strip()))
    len_mt = max(1, len(translation.strip()))
    ratio = len_mt / len_src
    if ratio < 0.4:
        issues.append("too_short")
        penalties += 0.25
    elif ratio > 3.0:
        issues.append("too_long")
        penalties += 0.25

    # 3. Degenerate repetition check
    if REPETITION_REGEX.search(translation):
        issues.append("degenerate_repetition")
        penalties += 0.4

    # 4. Untranslated Japanese leakage check (for JA -> VI/EN translations)
    ja_matches = JAPANESE_CHAR_REGEX.findall(translation)
    if ja_matches and len(ja_matches) > 2:
        issues.append(f"untranslated_leakage:{len(ja_matches)}_chars")
        penalties += min(0.3, len(ja_matches) * 0.05)

    # 5. COMET QE score
    comet_score: Optional[float] = None
    base_score = 0.90  # Default optimistic base
    if qe_client is not None:
        try:
            scores = qe_client.score_batch([{"src": source_text, "mt": translation}])
            if scores:
                comet_score = float(scores[0])
                base_score = comet_score
        except Exception as e:
            logger.warning(f"QE client score_batch failed: {e}. Using heuristic scoring.")

    composite_score = max(0.0, min(1.0, base_score - penalties))
    is_acceptable = (composite_score >= threshold) and (tag_score == 1.0) and (not issues)

    return QualityEstimate(
        score=composite_score,
        is_acceptable=is_acceptable,
        issues=issues,
        comet_score=comet_score,
        tag_score=tag_score,
        length_ratio=ratio,
    )


# ==============================================================================
# 4. TEaR Cycle (Translate -> Estimate -> Refine)
# ==============================================================================

@dataclass
class TEaRConfig:
    """Configuration for the TEaR optimization pipeline."""
    k_candidates: int = 4          # Number of candidates for MBR (4..8)
    qe_threshold: float = 0.85     # Minimum acceptance score
    max_refinements: int = 2       # Maximum refinement loops
    utility_metric: str = "chrf"   # 'chrf' or 'lcs'
    auto_recover_tags: bool = True # Enforce safe_recover_tags


@dataclass
class TEaRResult:
    """Final deliverable of the TEaR cycle."""
    final_translation: str
    initial_translation: str
    initial_score: float
    final_score: float
    refinements_count: int
    is_acceptable: bool
    issues: List[str]
    mbr_result: Optional[MBRResult] = None
    history: List[Dict[str, Any]] = field(default_factory=list)


class TEaRPipeline:
    """
    Orchestrates the Translate -> Estimate -> Refine cycle:
    1. Translate: Samples K candidates (K=4..8) and runs MBR consensus.
    2. Estimate: Evaluates candidate with COMET-QE and structural validators.
    3. Refine: If score < threshold, prompts LLM with diagnostic critique to refine.
    4. Recovers all tags deterministically before returning.
    """

    def __init__(
        self,
        config: Optional[TEaRConfig] = None,
        qe_client: Optional[Any] = None,
    ) -> None:
        self.config = config or TEaRConfig()
        self.qe_client = qe_client

    def run(
        self,
        source_text: str,
        translate_fn: Callable[[str, int], List[str]],
        refine_fn: Callable[[str, str, List[str]], str],
        expected_tags: Optional[Sequence[str]] = None,
    ) -> TEaRResult:
        """
        Executes full TEaR pipeline:
        - translate_fn(prompt, k) -> List of K sampled translation strings.
        - refine_fn(source, current_mt, issues) -> Refined translation string.
        """
        expected_tags_list = list(expected_tags or [])
        history: List[Dict[str, Any]] = []

        # ----------------------------------------------------------------------
        # Phase 1: Translate + MBR Decoding
        # ----------------------------------------------------------------------
        K = max(4, min(8, self.config.k_candidates))
        candidates = translate_fn(source_text, K)
        if not candidates:
            raise RuntimeError("translate_fn returned empty candidates list.")

        mbr_res: Optional[MBRResult] = None
        if len(candidates) > 1:
            mbr_res = minimum_bayes_risk(
                candidates=candidates,
                utility_metric=self.config.utility_metric,
            )
            current_mt = mbr_res.best_candidate
        else:
            current_mt = candidates[0]

        # Initial quality estimation
        initial_est = estimate_translation_quality(
            source_text=source_text,
            translation=current_mt,
            expected_placeholders=expected_tags_list,
            qe_client=self.qe_client,
            threshold=self.config.qe_threshold,
        )

        initial_mt = current_mt
        initial_score = initial_est.score
        current_score = initial_est.score
        current_issues = list(initial_est.issues)

        history.append({
            "stage": "initial_mbr",
            "translation": current_mt,
            "score": current_score,
            "issues": list(current_issues),
        })

        # ----------------------------------------------------------------------
        # Phase 2 & 3: Estimate & Refine Loop
        # ----------------------------------------------------------------------
        refinement_steps = 0
        while (
            refinement_steps < self.config.max_refinements
            and (current_score < self.config.qe_threshold or current_issues)
        ):
            refinement_steps += 1
            refined_mt = refine_fn(source_text, current_mt, current_issues)

            refined_est = estimate_translation_quality(
                source_text=source_text,
                translation=refined_mt,
                expected_placeholders=expected_tags_list,
                qe_client=self.qe_client,
                threshold=self.config.qe_threshold,
            )

            history.append({
                "stage": f"refine_{refinement_steps}",
                "translation": refined_mt,
                "score": refined_est.score,
                "issues": list(refined_est.issues),
            })

            # Adopt refinement if score improved or issues reduced
            if refined_est.score >= current_score:
                current_mt = refined_mt
                current_score = refined_est.score
                current_issues = list(refined_est.issues)
            else:
                # Diminishing returns: retain current best
                break

        # ----------------------------------------------------------------------
        # Final Safety Polish
        # ----------------------------------------------------------------------
        final_mt = current_mt
        if self.config.auto_recover_tags:
            final_mt = safe_recover_tags(final_mt, expected_tags=expected_tags_list)

        # Re-check acceptability
        final_acceptable = (current_score >= self.config.qe_threshold) and (not current_issues)

        return TEaRResult(
            final_translation=final_mt,
            initial_translation=initial_mt,
            initial_score=initial_score,
            final_score=current_score,
            refinements_count=refinement_steps,
            is_acceptable=final_acceptable,
            issues=current_issues,
            mbr_result=mbr_res,
            history=history,
        )


class TEaRMBREngine(TEaRPipeline):
    """Alias for TEaRPipeline with additional static helper methods."""

    @staticmethod
    def select_mbr(
        candidates: Sequence[str],
        utility_metric: str = "chrf",
        custom_utility_fn: Optional[Callable[[str, str], float]] = None,
    ) -> MBRResult:
        return minimum_bayes_risk(
            candidates=candidates,
            utility_metric=utility_metric,
            custom_utility_fn=custom_utility_fn,
        )

    @staticmethod
    def compute_chrf(hyp: str, ref: str) -> float:
        return compute_chrf(hyp, ref)

    @staticmethod
    def estimate_quality(
        source_text: str,
        translation: str,
        expected_placeholders: Optional[Sequence[str]] = None,
        qe_client: Optional[Any] = None,
        threshold: float = 0.85,
    ) -> QualityEstimate:
        return estimate_translation_quality(
            source_text=source_text,
            translation=translation,
            expected_placeholders=expected_placeholders,
            qe_client=qe_client,
            threshold=threshold,
        )

