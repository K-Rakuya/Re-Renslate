"""
lcs_segmenter.py
LCS (Longest Common Subsequence) based dialogue segmentation and stitching
for Ren'Py dialogue variants and inline branching statements.

Capabilities:
1. Token-level and character-level LCS alignment.
2. Pairwise and multi-variant segmentation into COMMON and BRANCH segments.
3. Template synthesis: encodes variants into inline branching syntax '{v1|v2}'.
4. Template expansion / stitching: reconstructs individual variant lines from templates.
5. Similarity analysis and variant clustering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


class SegmentKind(Enum):
    COMMON = auto()    # Shared identically across all variants
    BRANCH = auto()    # Divergent choices per variant


@dataclass
class Segment:
    """Represents a chunk of dialogue that is either common or divergent across variants."""
    kind: SegmentKind
    text: str = ""                         # Used when kind == COMMON
    branches: Tuple[str, ...] = field(default_factory=tuple)  # Used when kind == BRANCH

    def for_variant(self, variant_index: int) -> str:
        """Returns the text for the specified variant index."""
        if self.kind == SegmentKind.COMMON:
            return self.text
        if 0 <= variant_index < len(self.branches):
            return self.branches[variant_index]
        return ""


# ==============================================================================
# 1. LCS Algorithms
# ==============================================================================

def tokenize_words(text: str) -> List[str]:
    """Tokenizes text preserving words, punctuation, and spaces."""
    # Matches words, sequences of spaces, or individual punctuation marks
    return re.findall(r"\w+|\s+|[^\w\s]", text, re.UNICODE)


def compute_lcs_indices(
    seq1: Sequence[Any],
    seq2: Sequence[Any],
) -> List[Tuple[int, int]]:
    """
    Computes Longest Common Subsequence matching index pairs (i, j)
    between seq1 and seq2 using dynamic programming.
    """
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return []

    # DP table storing LCS lengths
    # Using 1D space-optimized or 2D table (dialogue lines are typically < 1000 tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        s1 = seq1[i - 1]
        for j in range(1, m + 1):
            if s1 == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find aligned index pairs
    matches: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    matches.reverse()
    return matches


def lcs_similarity(s1: str, s2: str, mode: str = "word") -> float:
    """
    Computes normalized LCS similarity ratio: 2 * |LCS| / (|s1| + |s2|).
    Returns value in [0.0, 1.0].
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    if mode == "word":
        t1, t2 = tokenize_words(s1), tokenize_words(s2)
    else:
        t1, t2 = list(s1), list(s2)

    matches = compute_lcs_indices(t1, t2)
    lcs_len = len(matches)
    total_len = len(t1) + len(t2)
    return (2.0 * lcs_len) / total_len if total_len > 0 else 0.0


# ==============================================================================
# 2. Pairwise & Multi-Variant Segmentation
# ==============================================================================

def segment_pair(
    v1: str,
    v2: str,
    mode: str = "word",
) -> List[Segment]:
    """
    Segments two dialogue variants into common blocks and divergent branch pairs.
    Returns a sequence of Segment objects.
    """
    if v1 == v2:
        return [Segment(kind=SegmentKind.COMMON, text=v1)]

    if mode == "word":
        tokens1 = tokenize_words(v1)
        tokens2 = tokenize_words(v2)
    else:
        tokens1 = list(v1)
        tokens2 = list(v2)

    matches = compute_lcs_indices(tokens1, tokens2)

    segments: List[Segment] = []
    idx1, idx2 = 0, 0

    for m1, m2 in matches:
        # Check if there is divergent content before this match
        diff1 = "".join(tokens1[idx1:m1])
        diff2 = "".join(tokens2[idx2:m2])
        if diff1 or diff2:
            segments.append(Segment(
                kind=SegmentKind.BRANCH,
                branches=(diff1, diff2),
            ))

        # Common token
        common_txt = "".join(tokens1[m1 : m1 + 1])
        if segments and segments[-1].kind == SegmentKind.COMMON:
            # Merge with preceding common segment
            segments[-1].text += common_txt
        else:
            segments.append(Segment(kind=SegmentKind.COMMON, text=common_txt))

        idx1 = m1 + 1
        idx2 = m2 + 1

    # Trailing differences
    tail1 = "".join(tokens1[idx1:])
    tail2 = "".join(tokens2[idx2:])
    if tail1 or tail2:
        segments.append(Segment(
            kind=SegmentKind.BRANCH,
            branches=(tail1, tail2),
        ))

    return segments


def segment_variants(
    variants: Sequence[str],
    mode: str = "word",
) -> List[Segment]:
    """
    Segments multiple dialogue variants (N >= 2) into shared common segments
    and N-way divergent branches.
    """
    n_vars = len(variants)
    if n_vars == 0:
        return []
    if n_vars == 1:
        return [Segment(kind=SegmentKind.COMMON, text=variants[0])]
    if n_vars == 2:
        return segment_pair(variants[0], variants[1], mode=mode)

    # For N > 2, start with base = variants[0] and align progressively
    curr_segments = segment_pair(variants[0], variants[1], mode=mode)

    for k in range(2, n_vars):
        target = variants[k]
        # Reconstruct base string from common + branch[0]
        base_str = "".join(s.for_variant(0) for s in curr_segments)
        new_pair_segments = segment_pair(base_str, target, mode=mode)

        # Merge new alignment back into multi-branch structure
        # If lengths match structurally, expand branches; otherwise fallback to whole-line branch
        # Most VN dialogue variants have matching structural branch points (e.g. pronoun swaps)
        can_align = len(new_pair_segments) == len(curr_segments) and all(
            (s1.kind == s2.kind) for s1, s2 in zip(curr_segments, new_pair_segments)
        )
        if can_align:
            for s_curr, s_new in zip(curr_segments, new_pair_segments):
                if s_curr.kind == SegmentKind.BRANCH:
                    s_curr.branches = s_curr.branches + (s_new.branches[1],)
        else:
            # Whole line divergence fallback
            return [Segment(
                kind=SegmentKind.BRANCH,
                branches=tuple(variants),
            )]

    return curr_segments


# ==============================================================================
# 3. Inline Branching Templates (Tách & Ghép)
# ==============================================================================

def to_template(
    segments: Sequence[Segment],
    open_delim: str = "{",
    close_delim: str = "}",
    sep: str = "|",
) -> str:
    """
    Encodes segments into an inline branching template string:
    e.g. "Chào {anh|chị}, em rất vui được gặp {anh|chị}!"
    """
    parts: List[str] = []
    for seg in segments:
        if seg.kind == SegmentKind.COMMON:
            parts.append(seg.text)
        elif seg.kind == SegmentKind.BRANCH:
            branch_body = sep.join(seg.branches)
            parts.append(f"{open_delim}{branch_body}{close_delim}")
    return "".join(parts)


def from_template(
    template: str,
    open_delim: str = "{",
    close_delim: str = "}",
    sep: str = "|",
) -> List[Segment]:
    """
    Parses an inline branching template string back into a list of Segment objects.
    e.g. "Chào {anh|chị}!" -> [COMMON("Chào "), BRANCH(("anh", "chị")), COMMON("!")]
    """
    # Regex matching {branch1|branch2|...} while escaping delimiters if needed
    esc_open = re.escape(open_delim)
    esc_close = re.escape(close_delim)
    esc_sep = re.escape(sep)

    pattern = re.compile(f"{esc_open}(?P<branch>[^{esc_close}]*?){esc_close}")
    segments: List[Segment] = []
    last_end = 0

    for match in pattern.finditer(template):
        start, end = match.span()
        if start > last_end:
            segments.append(Segment(
                kind=SegmentKind.COMMON,
                text=template[last_end:start],
            ))

        branch_text = match.group("branch")
        branches = tuple(branch_text.split(sep))
        segments.append(Segment(
            kind=SegmentKind.BRANCH,
            branches=branches,
        ))
        last_end = end

    if last_end < len(template):
        segments.append(Segment(
            kind=SegmentKind.COMMON,
            text=template[last_end:],
        ))

    return segments


def expand_template(
    template: str,
    num_variants: Optional[int] = None,
    open_delim: str = "{",
    close_delim: str = "}",
    sep: str = "|",
) -> List[str]:
    """
    Expands an inline branching template into its constituent variant lines.
    If num_variants is omitted, infers the maximum branch count from the template.
    """
    segments = from_template(template, open_delim=open_delim, close_delim=close_delim, sep=sep)

    if num_variants is None:
        max_branches = 1
        for seg in segments:
            if seg.kind == SegmentKind.BRANCH:
                max_branches = max(max_branches, len(seg.branches))
        num_variants = max_branches

    results: List[str] = []
    for var_idx in range(num_variants):
        variant_line = "".join(seg.for_variant(var_idx) for seg in segments)
        results.append(variant_line)

    return results


def stitch_variants(
    segments: Sequence[Segment],
    num_variants: Optional[int] = None,
) -> List[str]:
    """
    Reconstructs individual variant strings directly from segments.
    """
    if num_variants is None:
        num_variants = 1
        for seg in segments:
            if seg.kind == SegmentKind.BRANCH:
                num_variants = max(num_variants, len(seg.branches))

    return [
        "".join(seg.for_variant(v_idx) for seg in segments)
        for v_idx in range(num_variants)
    ]


# ==============================================================================
# 4. Variant Clustering
# ==============================================================================

def find_variant_clusters(
    lines: Sequence[str],
    min_similarity: float = 0.65,
) -> List[List[str]]:
    """
    Groups dialogue lines into clusters of variants sharing high LCS similarity.
    Lines with similarity >= min_similarity are grouped together.
    """
    clusters: List[List[str]] = []
    assigned: Set[int] = set()

    for i, line in enumerate(lines):
        if i in assigned:
            continue
        curr_cluster = [line]
        assigned.add(i)

        for j in range(i + 1, len(lines)):
            if j in assigned:
                continue
            other = lines[j]
            sim = lcs_similarity(line, other)
            if sim >= min_similarity:
                curr_cluster.append(other)
                assigned.add(j)

        clusters.append(curr_cluster)

    return clusters


# ==============================================================================
# 5. Object-Oriented Interface Wrapper
# ==============================================================================

class LCSSegmenter:
    """Convenience class providing unified access to LCS segmentation methods."""

    @staticmethod
    def segment(v1: str, v2: str, mode: str = "word") -> List[Segment]:
        return segment_pair(v1, v2, mode=mode)

    @staticmethod
    def segment_multi(variants: Sequence[str], mode: str = "word") -> List[Segment]:
        return segment_variants(variants, mode=mode)

    @staticmethod
    def to_template(
        segments: Sequence[Segment],
        open_delim: str = "{",
        close_delim: str = "}",
        sep: str = "|",
    ) -> str:
        return to_template(segments, open_delim=open_delim, close_delim=close_delim, sep=sep)

    @staticmethod
    def from_template(
        template: str,
        open_delim: str = "{",
        close_delim: str = "}",
        sep: str = "|",
    ) -> List[Segment]:
        return from_template(template, open_delim=open_delim, close_delim=close_delim, sep=sep)

    @staticmethod
    def expand(
        template: str,
        num_variants: Optional[int] = None,
        open_delim: str = "{",
        close_delim: str = "}",
        sep: str = "|",
    ) -> List[str]:
        return expand_template(template, num_variants=num_variants, open_delim=open_delim, close_delim=close_delim, sep=sep)

    @staticmethod
    def stitch(
        segments: Sequence[Segment],
        num_variants: Optional[int] = None,
    ) -> List[str]:
        return stitch_variants(segments, num_variants=num_variants)

    @staticmethod
    def similarity(s1: str, s2: str, mode: str = "word") -> float:
        return lcs_similarity(s1, s2, mode=mode)

    @staticmethod
    def cluster(lines: Sequence[str], min_similarity: float = 0.65) -> List[List[str]]:
        return find_variant_clusters(lines, min_similarity=min_similarity)

