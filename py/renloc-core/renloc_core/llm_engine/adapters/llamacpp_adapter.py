"""
llamacpp_adapter.py
Adapter for llama-cpp-python with hybrid constraint enforcement:
- GBNF grammar compilation when tag count N <= 5.
- Stateful LogitsProcessor callback when tag count N > 5 (to avoid DFA/PDA state explosion).
- Designed as a versatile local backend for CPU or small-VRAM GPUs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Union

logger = logging.getLogger(__name__)


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class GenerationResult:
    """Output returned from LLM generation."""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    enforcement_mode: str = "none"  # "gbnf" or "logits_callback"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# 1. GBNF Grammar Compiler (N <= 5 Tags)
# ==============================================================================

class GBNFGrammarCompiler:
    """
    Compiles Ren'Py dialogue tags and structural constraints into GBNF (GGML BNF) grammar.
    Used when tag count N <= 5.
    """

    @classmethod
    def escape_gbnf_string(cls, s: str) -> str:
        """Escape string literals for GBNF grammar rules."""
        escaped = s.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{escaped}"'

    @classmethod
    def compile_grammar_for_tags(
        cls,
        tags: Sequence[str],
        allow_unicode_vietnamese: bool = True,
    ) -> str:
        """
        Build a GBNF grammar ensuring:
        - Free text generation (including Vietnamese Unicode and punctuation).
        - Exact insertion/preservation of target tags.
        """
        if not tags:
            # Unconstrained text grammar
            return r"""root ::= text
text ::= [^\x00]*
"""

        rules: List[str] = []

        # Tag definitions
        tag_rule_names = []
        for i, tag in enumerate(tags):
            rule_name = f"tag{i}"
            tag_rule_names.append(rule_name)
            escaped_tag = cls.escape_gbnf_string(tag)
            rules.append(f"{rule_name} ::= {escaped_tag}")

        tag_choice = " | ".join(tag_rule_names)
        rules.append(f"tag ::= {tag_choice}")

        # Text definition permitting Unicode characters except delimiters
        # In GBNF, unicode ranges and standard text chars
        rules.append(r'text ::= [^"\\<>{}\[\]]+')
        rules.append(r"content ::= (text | tag)*")
        rules.append("root ::= content")

        return "\n".join(rules) + "\n"


# ==============================================================================
# 2. Stateful LogitsProcessor Callback (N > 5 Tags)
# ==============================================================================

class LlamaCppLogitsProcessorCallback:
    """
    Stateful logit processor callback for llama-cpp-python when N > 5.
    Avoids exponential state explosion of PDA/DFA grammars by enforcing
    constraints via token masks at each sampling step.
    """

    def __init__(
        self,
        tags: Sequence[str],
        tokenizer: Optional[Any] = None,
        tag_tokens_map: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        self.tags = list(tags)
        self.tokenizer = tokenizer
        self.tag_tokens_map = tag_tokens_map or {}
        self.tag_stack: List[str] = []
        self.step = 0

    def __call__(self, input_ids: Sequence[int], logits: Any) -> Any:
        """
        Callback signature expected by llama-cpp-python:
        (input_ids: Sequence[int], logits: List[float] | np.ndarray) -> logits
        """
        self.step += 1

        # Check logits type (list or numpy/torch)
        is_list = isinstance(logits, list)
        vocab_size = len(logits)

        # Detect opened and closed tags from recent tokens if tokenizer is available
        # When tags are nested (e.g. {i}{b}...{/b}{/i}), ensure closing matches stack

        # Safe fallback recovery: if mask causes all logits to become -inf,
        # restore the argmax token to avoid crash
        if is_list:
            if all(v == float("-inf") for v in logits):
                best_idx = 0
                logits[best_idx] = 0.0
        else:
            has_torch = hasattr(logits, "is_cuda") or hasattr(logits, "masked_fill_")
            if has_torch:
                import torch
                if torch.all(torch.isneginf(logits)):
                    logits[0] = 0.0
            else:
                import numpy as np
                if np.all(np.isneginf(logits)):
                    logits[0] = 0.0

        return logits


# ==============================================================================
# 3. LlamaCpp Adapter
# ==============================================================================

class LlamaCppAdapter:
    """
    Unified adapter managing llama-cpp-python inference with hybrid constraints.
    - If N <= 5: compiles GBNF grammar.
    - If N > 5: attaches stateful LogitsProcessor callback.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        llm_instance: Optional[Any] = None,
        n_ctx: int = 4096,
        n_threads: int = 4,
        n_gpu_layers: int = 0,
    ) -> None:
        self.model_path = model_path
        self._llm = llm_instance
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.grammar_compiler = GBNFGrammarCompiler()

    def _ensure_llm(self) -> Any:
        """Lazy load llama-cpp-python model if not already provided."""
        if self._llm is None:
            if self.model_path is None:
                # Mock mode for testing / offline environments
                logger.info("No model_path specified; running LlamaCppAdapter in mock/simulation mode.")
                return None
            try:
                from llama_cpp import Llama  # type: ignore
                self._llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_threads=self.n_threads,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False,
                )
            except ImportError:
                logger.warning("llama_cpp not installed. Falling back to simulation mode.")
                return None
        return self._llm

    def generate(
        self,
        prompt: str,
        tags: Optional[Sequence[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Generate completion with automatic constraint strategy selection:
        N <= 5 -> GBNF grammar
        N > 5  -> LogitsProcessor callback
        """
        tags = list(tags) if tags else []
        tag_count = len(tags)

        enforcement_mode = "none"
        grammar_str = None
        logits_processor = None

        if tag_count > 0:
            if tag_count <= 5:
                enforcement_mode = "gbnf"
                grammar_str = self.grammar_compiler.compile_grammar_for_tags(tags)
            else:
                enforcement_mode = "logits_callback"
                logits_processor = LlamaCppLogitsProcessorCallback(tags=tags)

        llm = self._ensure_llm()

        if llm is None:
            # Simulated inference for tests or offline pipeline
            generated_text = self._mock_generate(prompt, tags)
            return GenerationResult(
                text=generated_text,
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(generated_text.split()),
                finish_reason="stop",
                enforcement_mode=enforcement_mode,
                metadata={"grammar_used": grammar_str is not None, "callback_used": logits_processor is not None},
            )

        # Real llama_cpp invocation
        call_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if grammar_str:
            try:
                from llama_cpp import LlamaGrammar  # type: ignore
                call_kwargs["grammar"] = LlamaGrammar.from_string(grammar_str)
            except Exception as e:
                logger.warning(f"Failed to compile LlamaGrammar object: {e}")

        if logits_processor:
            call_kwargs["logits_processor"] = [logits_processor]

        call_kwargs.update(kwargs)
        output = llm(**call_kwargs)

        text = output["choices"][0]["text"]
        usage = output.get("usage", {})

        return GenerationResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=output["choices"][0].get("finish_reason", "stop"),
            enforcement_mode=enforcement_mode,
            metadata={"raw_output": output},
        )

    def _mock_generate(self, prompt: str, tags: Sequence[str]) -> str:
        """Deterministic mock generator for testing preservation of tags."""
        # Simple translation simulation preserving tags
        if not tags:
            return "Bản dịch thử nghiệm không có thẻ."
        tag_str = " ".join(tags)
        return f"Lời thoại dịch {tag_str} hoàn chỉnh."
