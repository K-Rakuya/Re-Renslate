"""
renloc_core.synthesizer
Package synthesizing, font shaping/auditing, incremental matching, and runtime hooks.
"""

from .font_shaper import (
    FontAuditor,
    FontAuditResult,
    GuiLayoutConfig,
    GuiInspector,
    DialogueShaper,
    OverflowWarning,
    VIETNAMESE_NFC_CHARS,
)
from .incremental_matcher import (
    IncrementalMatcher,
    IncrementalMatchResult,
    MatchTier,
    levenshtein_similarity,
)
from .package_synthesizer import (
    PackageSynthesizer,
    TranslationItem,
    UIStringTranslationItem,
)
from .runtime_hook import (
    MockRevertableDict,
    MockRevertableList,
    RenlocCircuitBreaker,
    StormCircuitBreaker,
    RuntimeHookManager,
    generate_runtime_hook_rpy,
)

__all__ = [
    "FontAuditor",
    "FontAuditResult",
    "GuiLayoutConfig",
    "GuiInspector",
    "DialogueShaper",
    "OverflowWarning",
    "VIETNAMESE_NFC_CHARS",
    "IncrementalMatcher",
    "IncrementalMatchResult",
    "MatchTier",
    "levenshtein_similarity",
    "PackageSynthesizer",
    "TranslationItem",
    "UIStringTranslationItem",
    "MockRevertableDict",
    "MockRevertableList",
    "RenlocCircuitBreaker",
    "RuntimeHookManager",
]
