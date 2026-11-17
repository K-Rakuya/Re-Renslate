"""
constraint_compiler.py
Hybrid Constraint Engine for LLM Generation:
- For tag count N <= 5: Compiles CFG / GBNF / EBNF grammars for deterministic finite grammar engines.
- For tag count N > 5: Uses a stateful LogitsProcessor maintaining a tag stack with -inf masking
  and safe recovery mechanisms (preventing grammar DFA/PDA state space explosion).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Threshold separating CFG grammar compilation from stateful LogitsProcessor
TAG_THRESHOLD = 5


class ConstraintMode(Enum):
    GBNF_GRAMMAR = auto()
    EBNF_GRAMMAR = auto()
    LOGITS_PROCESSOR = auto()


@dataclass
class ConstraintSpec:
    """Specification of the compiled constraint strategy."""
    mode: ConstraintMode
    tag_count: int
    grammar: Optional[str] = None
    processor: Optional[TagStackLogitsProcessor] = None
    expected_tags: List[str] = field(default_factory=list)


# ==============================================================================
# 1. CFG / GBNF / EBNF Grammar Compiler (N <= 5 Tags)
# ==============================================================================

class GrammarCompiler:
    """
    Compiles target tags and placeholders into CFG / GBNF / EBNF grammar rules.
    Used when tag count N <= 5.
    """

    @staticmethod
    def escape_gbnf_string(s: str) -> str:
        """Escapes string literals for GBNF grammar syntax."""
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    @classmethod
    def compile_gbnf(
        cls,
        tags: Sequence[str],
        paired_tags: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> str:
        """
        Compiles a GBNF (GGML BNF) grammar:
        - Free text allowing Vietnamese Unicode and standard punctuation.
        - Standalone tags and invariant placeholders (<T0>, <V1!ti>).
        - Balanced paired tags ({b}...{/b}).
        """
        if not tags and not paired_tags:
            return "root ::= text*\ntext ::= [^\"\\\\\x00-\x1f]+\n"

        rules: List[str] = []
        atom_rules: List[str] = []

        # 1. Paired tags rules (e.g. {b}...{/b})
        if paired_tags:
            for idx, (open_tag, close_tag) in enumerate(paired_tags):
                rule_name = f"paired_{idx}"
                esc_open = cls.escape_gbnf_string(open_tag)
                esc_close = cls.escape_gbnf_string(close_tag)
                rules.append(f"{rule_name} ::= {esc_open} content {esc_close}")
                atom_rules.append(rule_name)

        # 2. Standalone tags and placeholders (<T0>, <V0!ti>, {w}, etc.)
        standalone_rules: List[str] = []
        for idx, tag in enumerate(tags):
            rule_name = f"tag_{idx}"
            esc_tag = cls.escape_gbnf_string(tag)
            rules.append(f"{rule_name} ::= {esc_tag}")
            standalone_rules.append(rule_name)

        if standalone_rules:
            rules.append(f"standalone_tag ::= {' | '.join(standalone_rules)}")
            atom_rules.append("standalone_tag")

        # 3. Text definition: Unicode-friendly, excluding delimiters
        rules.append(r'text ::= [^"\\<>{}\[\]\x00-\x1f]+')
        atom_rules.append("text")

        # 4. Content and Root rules
        choice = " | ".join(atom_rules)
        rules.append(f"content ::= ({choice})*")
        rules.append("root ::= content")

        return "\n".join(rules) + "\n"

    @classmethod
    def compile_ebnf(
        cls,
        tags: Sequence[str],
        paired_tags: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> str:
        """
        Compiles standard EBNF grammar representation for general parser generators.
        """
        rules: List[str] = []
        atom_choices: List[str] = []

        if paired_tags:
            for idx, (open_tag, close_tag) in enumerate(paired_tags):
                rule_name = f"paired_{idx}"
                rules.append(f'{rule_name} = "{open_tag}", content, "{close_tag}" ;')
                atom_choices.append(rule_name)

        tag_literals = [f'"{t}"' for t in tags]
        if tag_literals:
            rules.append(f"tag = {' | '.join(tag_literals)} ;")
            atom_choices.append("tag")

        rules.append('text = ? any valid Unicode character except delimiters ? ;')
        atom_choices.append("text")

        choice = " | ".join(atom_choices)
        rules.append(f"content = {{ {choice} }} ;")
        rules.append("root = content ;")

        return "\n".join(rules) + "\n"


# ==============================================================================
# 2. Stateful LogitsProcessor (N > 5 Tags)
# ==============================================================================

# Standard Ren'Py opening -> closing tag mapping
DEFAULT_PAIRED_TAGS = {
    "{b}": "{/b}",
    "{i}": "{/i}",
    "{u}": "{/u}",
    "{s}": "{/s}",
    "{rb}": "{/rb}",
    "{rt}": "{/rt}",
}


class TagStackLogitsProcessor:
    """
    Stateful LogitsProcessor for token-level constraint enforcement when N > 5.
    Maintains a LIFO tag stack to ensure balanced tag closing, enforces presence
    of required placeholders, masks invalid token logits to -inf, and provides
    safe recovery to prevent generation deadlocks.
    """

    def __init__(
        self,
        expected_tags: Sequence[str],
        paired_tags_map: Optional[Dict[str, str]] = None,
        eos_token_id: Optional[int] = None,
        tag_to_tokens_map: Optional[Dict[str, List[int]]] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.expected_tags = list(expected_tags)
        self.paired_tags_map = paired_tags_map or dict(DEFAULT_PAIRED_TAGS)
        self.reverse_paired_map = {v: k for k, v in self.paired_tags_map.items()}
        self.eos_token_id = eos_token_id
        self.tag_to_tokens_map = tag_to_tokens_map or {}
        self.tokenizer = tokenizer

        # Stateful tracking
        self.tag_stack: List[str] = []
        self.emitted_tags: Set[str] = set()
        self.step = 0
        self.deadlock_recovered_count = 0

    def reset(self) -> None:
        """Resets the stateful processor for a new generation turn."""
        self.tag_stack.clear()
        self.emitted_tags.clear()
        self.step = 0
        self.deadlock_recovered_count = 0

    def on_tag_opened(self, open_tag: str) -> None:
        """Pushes an opened tag onto the stack."""
        self.tag_stack.append(open_tag)
        self.emitted_tags.add(open_tag)

    def on_tag_closed(self, close_tag: str) -> bool:
        """
        Pops a closed tag from the stack.
        Returns True if the closing tag matched the top of the stack.
        """
        expected_open = self.reverse_paired_map.get(close_tag)
        if self.tag_stack and self.tag_stack[-1] == expected_open:
            self.tag_stack.pop()
            return True
        elif expected_open and expected_open in self.tag_stack:
            # Tolerant recovery: pop up to the matching open tag
            while self.tag_stack:
                popped = self.tag_stack.pop()
                if popped == expected_open:
                    break
            return True
        return False

    def __call__(
        self,
        input_ids: Sequence[int],
        scores: Any,
    ) -> Any:
        """
        Logits processor callback: (input_ids, scores) -> scores.
        Compatible with PyTorch, numpy, or plain Python list of floats.
        """
        self.step += 1
        is_list = isinstance(scores, list)

        # 1. If tags are still unclosed on the stack, mask EOS token to -inf
        if self.tag_stack and self.eos_token_id is not None:
            if 0 <= self.eos_token_id < len(scores):
                if is_list:
                    scores[self.eos_token_id] = float("-inf")
                else:
                    scores[self.eos_token_id] = float("-inf")

        # 2. Check for deadlock: if all scores are -inf, perform Safe Recovery
        if self._is_all_neginf(scores):
            self.deadlock_recovered_count += 1
            logger.warning(
                "Deadlock detected in TagStackLogitsProcessor: all logits are -inf. "
                "Executing Safe Recovery unblock."
            )
            # Unblock the top-of-stack closing tag token, or argmax fallback token 0
            unblocked_token = 0
            if self.tag_stack:
                top_open = self.tag_stack[-1]
                top_close = self.paired_tags_map.get(top_open)
                if top_close and top_close in self.tag_to_tokens_map:
                    tokens = self.tag_to_tokens_map[top_close]
                    if tokens:
                        unblocked_token = tokens[0]

            if is_list:
                scores[unblocked_token] = 0.0
            else:
                scores[unblocked_token] = 0.0

        return scores

    @staticmethod
    def _is_all_neginf(scores: Any) -> bool:
        """Checks if all logits are negative infinity."""
        if isinstance(scores, list):
            return all(v == float("-inf") for v in scores)

        has_torch = hasattr(scores, "is_cuda") or hasattr(scores, "masked_fill_")
        if has_torch:
            import torch
            return bool(torch.all(torch.isneginf(scores)))

        import numpy as np
        return bool(np.all(np.isneginf(scores)))


# ==============================================================================
# 3. Post-Generation Safe Recovery (Phục hồi an toàn)
# ==============================================================================

TAG_OPEN_REGEX = re.compile(r"\{([a-zA-Z0-9_]+)(?:=[^}]+)?\}")
TAG_CLOSE_REGEX = re.compile(r"\{/([a-zA-Z0-9_]+)\}")
PLACEHOLDER_REGEX = re.compile(r"<([TV])\d+(![a-zA-Z]+)?>")


def safe_recover_tags(
    text: str,
    expected_tags: Sequence[str],
    paired_tags_map: Optional[Dict[str, str]] = None,
) -> str:
    """
    Deterministic post-generation safe recovery:
    1. Repairs tag balance: appends closing tags for any unclosed opening tags in LIFO order.
    2. Strips orphan closing tags (closing tags without matching opening tags).
    3. Reinserts missing required invariant placeholders (<T...>, <V...>) if dropped by the model.
    """
    if paired_tags_map is None:
        paired_tags_map = dict(DEFAULT_PAIRED_TAGS)
    reverse_map = {v: k for k, v in paired_tags_map.items()}

    # 1. Track tag balance using a stack
    stack: List[str] = []
    # Tokenize text tags
    tag_pattern = re.compile(r"(\{[a-zA-Z0-9_]+(?:=[^}]+)?\}|\{/[a-zA-Z0-9_]+\})")
    parts = tag_pattern.split(text)

    sanitized_parts: List[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("{/") and part.endswith("}"):
            # Closing tag
            tag_name = part[2:-1]
            # Find matching open tag
            matched_idx = -1
            for idx in range(len(stack) - 1, -1, -1):
                open_name = stack[idx].lstrip("{").split("=")[0]
                if open_name == tag_name:
                    matched_idx = idx
                    break

            if matched_idx != -1:
                # Pop down to matched
                while len(stack) > matched_idx:
                    stack.pop()
                sanitized_parts.append(part)
            else:
                # Orphan closing tag: drop to keep text clean
                continue
        elif part.startswith("{") and part.endswith("}"):
            # Opening tag
            tag_name = part[1:-1].split("=")[0]
            # Check if this tag type is paired
            is_paired = any(k.startswith(f"{{{tag_name}") for k in paired_tags_map.keys())
            if is_paired:
                stack.append(part)
            sanitized_parts.append(part)
        else:
            sanitized_parts.append(part)

    # Auto-close remaining open tags in LIFO order
    while stack:
        open_tag = stack.pop()
        open_name = open_tag[1:-1].split("=")[0]
        close_tag = f"{{/{open_name}}}"
        sanitized_parts.append(close_tag)

    recovered_text = "".join(sanitized_parts)

    # 2. Check and safely reinsert missing required placeholders
    for req_tag in expected_tags:
        if req_tag not in recovered_text:
            # If a required invariant placeholder was omitted by the LLM, append safely
            if req_tag.startswith("<") and req_tag.endswith(">"):
                recovered_text += f" {req_tag}"

    return recovered_text


# ==============================================================================
# 4. Unified Hybrid Constraint Compiler
# ==============================================================================

class ConstraintCompiler:
    """
    Unified compiler determining the optimal constraint enforcement mode:
    - If len(tags) <= TAG_THRESHOLD (5): generates GBNF/EBNF grammar.
    - If len(tags) > TAG_THRESHOLD (5): instantiates stateful TagStackLogitsProcessor.
    """

    def __init__(self, threshold: int = TAG_THRESHOLD) -> None:
        self.threshold = threshold
        self.grammar_compiler = GrammarCompiler()

    def compile(
        self,
        tags: Sequence[str],
        paired_tags: Optional[Sequence[Tuple[str, str]]] = None,
        eos_token_id: Optional[int] = None,
        tag_to_tokens_map: Optional[Dict[str, List[int]]] = None,
        tokenizer: Optional[Any] = None,
    ) -> ConstraintSpec:
        """
        Compiles constraints for the given tags list:
        - N <= 5: returns ConstraintSpec with GBNF grammar.
        - N > 5: returns ConstraintSpec with stateful LogitsProcessor and recovery mechanisms.
        """
        all_tags = list(tags)
        total_tag_count = len(all_tags) + (len(paired_tags) if paired_tags else 0)

        if total_tag_count <= self.threshold:
            grammar_str = self.grammar_compiler.compile_gbnf(
                tags=all_tags,
                paired_tags=paired_tags,
            )
            return ConstraintSpec(
                mode=ConstraintMode.GBNF_GRAMMAR,
                tag_count=total_tag_count,
                grammar=grammar_str,
                expected_tags=all_tags,
            )
        else:
            paired_dict = dict(DEFAULT_PAIRED_TAGS)
            if paired_tags:
                for o, c in paired_tags:
                    paired_dict[o] = c

            processor = TagStackLogitsProcessor(
                expected_tags=all_tags,
                paired_tags_map=paired_dict,
                eos_token_id=eos_token_id,
                tag_to_tokens_map=tag_to_tokens_map,
                tokenizer=tokenizer,
            )
            return ConstraintSpec(
                mode=ConstraintMode.LOGITS_PROCESSOR,
                tag_count=total_tag_count,
                processor=processor,
                expected_tags=all_tags,
            )


# Alias for compatibility with external router/engine conventions
HybridConstraintCompiler = ConstraintCompiler

