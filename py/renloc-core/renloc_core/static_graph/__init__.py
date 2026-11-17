"""
renloc_core/static_graph/__init__.py
Static Analysis Engine & Contextual Knowledge Graph components:
  - lifecycle: Ren'Py 5-phase initialization simulator & WrapNode
  - dialogue_normalizer: 4-form dialogue normalization into DialogueEntity
  - screen_extractor: Screen Language 2 UI string extraction & LayeredImage tracking
  - icfg: k-CFA (k=2 call-string context-sensitive interprocedural CFG)
  - tarjan_scc: Tarjan SCC loop detection & inter-region DAG condensation
  - dominator: Lengauer-Tarjan Dominator Tree & milestone separation
  - trace_partitioning: Path-Sensitive Trace Partitioning for relational variables
  - honorific_graph: Temporal Honorific Graph & zero-leakage character reveal
"""
from .lifecycle import (
    LifecyclePhase,
    WrapNode,
    LifecycleSimulator,
)

from .dialogue_normalizer import (
    DialogueForm,
    DialogueEntity,
    DialogueNormalizer,
)

from .screen_extractor import (
    ExtractedUIString,
    ScreenExtractor,
    LayeredImageTracker,
)

from .icfg import (
    EdgeType,
    CallContext,
    ICFGNode,
    ICFGEdge,
    ICFG,
    ICFGBuilder,
)

from .tarjan_scc import (
    LoopKind,
    SCCComponent,
    CondensationDAG,
    TarjanSCC,
)

from .dominator import (
    DominatorRole,
    DominatorInfo,
    LengauerTarjanDominator,
)

from .trace_partitioning import (
    PartitionKey,
    AbstractState,
    PartitionedTraceNode,
    TracePartitioningEngine,
)

from .honorific_graph import (
    UNKNOWN_IDENTITY,
    FormalityTier,
    CharacterIdentity,
    HonorificProfile,
    TemporalHonorificGraph,
)

__all__ = [
    # Lifecycle
    "LifecyclePhase",
    "WrapNode",
    "LifecycleSimulator",
    # Dialogue Normalizer
    "DialogueForm",
    "DialogueEntity",
    "DialogueNormalizer",
    # Screen Extractor
    "ExtractedUIString",
    "ScreenExtractor",
    "LayeredImageTracker",
    # ICFG
    "EdgeType",
    "CallContext",
    "ICFGNode",
    "ICFGEdge",
    "ICFG",
    "ICFGBuilder",
    # Tarjan SCC
    "LoopKind",
    "SCCComponent",
    "CondensationDAG",
    "TarjanSCC",
    # Dominator Tree
    "DominatorRole",
    "DominatorInfo",
    "LengauerTarjanDominator",
    # Trace Partitioning
    "PartitionKey",
    "AbstractState",
    "PartitionedTraceNode",
    "TracePartitioningEngine",
    # Temporal Honorific Graph
    "UNKNOWN_IDENTITY",
    "FormalityTier",
    "CharacterIdentity",
    "HonorificProfile",
    "TemporalHonorificGraph",
]
