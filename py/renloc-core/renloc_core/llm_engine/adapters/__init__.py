"""
Adapters package for Renloc LLM Engine.
Provides pluggable adapters for vLLM, SGLang, and llama-cpp-python.
"""

from .vllm_adapter import (
    VllmV1BatchUpdateProcessor,
    VllmV2ModelRunnerProcessor,
    VllmCompatAdapter,
    BatchUpdate,
    MoveDirectionality,
)
from .sglang_adapter import (
    SglangSecurityProxy,
    SglangConstraintProcessor,
    SecurityVulnerabilityError,
    UnauthenticatedPayloadError,
)
from .llamacpp_adapter import (
    LlamaCppAdapter,
    GBNFGrammarCompiler,
)

__all__ = [
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
]
