"""
llm_engine package for Renslate.
Provides local inference adapters, idempotent DAG scheduling, VRAM monitoring,
and semantic 3-tier routing.
"""

from .vram_monitor import VramMonitor, VramInfo, ModelVramConfig
from .dag_scheduler import DagScheduler, DagTask, TaskStatus, compute_dag_task_key
from .router import SemanticDialogueRouter, DialogueMetadata, RouteResult, RoutingTier
from .adapters import (
    VllmV1BatchUpdateProcessor,
    VllmV2ModelRunnerProcessor,
    VllmCompatAdapter,
    BatchUpdate,
    MoveDirectionality,
    SglangSecurityProxy,
    SglangConstraintProcessor,
    SecurityVulnerabilityError,
    UnauthenticatedPayloadError,
    LlamaCppAdapter,
    GBNFGrammarCompiler,
)

# Optional NLP imports
try:
    from .interpolation_lexer import InterpolationLexer
except ImportError:
    pass

try:
    from .lcs_segmenter import LCSSegmenter
except ImportError:
    pass

try:
    from .constraint_compiler import HybridConstraintCompiler
except ImportError:
    pass

try:
    from .tear_mbr import TEaRMBREngine
except ImportError:
    pass

__all__ = [
    "VramMonitor",
    "VramInfo",
    "ModelVramConfig",
    "DagScheduler",
    "DagTask",
    "TaskStatus",
    "compute_dag_task_key",
    "SemanticDialogueRouter",
    "DialogueMetadata",
    "RouteResult",
    "RoutingTier",
    "VllmV1BatchUpdateProcessor",
    "VllmV2ModelRunnerProcessor",
    "VllmCompatAdapter",
    "BatchUpdate",
    "MoveDirectionality",
    "SglangSecurityProxy",
    "SglangConstraintProcessor",
    "SecurityVulnerabilityError",
    "UnauthenticatedPayloadError",
    "LlamaCppAdapter",
    "GBNFGrammarCompiler",
    "InterpolationLexer",
    "LCSSegmenter",
    "HybridConstraintCompiler",
    "TEaRMBREngine",
]
