"""
interpolation_lexer.py
Ren'Py substitutions and text tags lexer, invariant placeholder masker/restorer,
and 7-step conversion pipeline.

Key Capabilities:
1. Ren'Py text tag & escape sequence lexing:
   - Escapes: '\\ ' (preserve extra whitespace), '\\n', '\\\\', '\\"', '\\'', '{{', '[[', '}}', ']]'.
   - Ruby text: '【dưới｜trên】' <-> '{rb}dưới{/rb}{rt}trên{/rt}'.
   - Tags: paired tags ({b}, {/b}, {color=...}, etc.) and standalone tags ({w}, {p}, {nw}, {fast}).
2. 7-step conversion pipeline:
   r/s -> t -> i -> q -> (f) -> u -> l -> c.
   Flag normalization: !cl == !lc, redundant '!' and duplicate flags ignored.
3. Invariant placeholders:
   <T0>, <T1>, ... for text tags, escapes, and ruby blocks.
   <V0>, <V1!ti>, ... for variable interpolations with normalized flags.
   Deterministic 1-1 bidirectional masking and restoration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


# ==============================================================================
# Token Definitions
# ==============================================================================

class TokenType(Enum):
    TEXT = auto()
    ESCAPE_SPACE = auto()       # \  (escaped whitespace)
    ESCAPE_CHAR = auto()        # \n, \\, \", \'
    ESCAPED_BRACKET = auto()    # {{, [[, }}, ]]
    RUBY_TEXT = auto()          # 【bottom｜top】 or 【base|ruby】
    TEXT_TAG = auto()           # {b}, {/b}, {color=#fff}, etc.
    VARIABLE = auto()           # [var_name], [var!flags]


@dataclass(frozen=True)
class Token:
    type: TokenType
    raw: str
    value: str
    flags: str = ""
    index: int = 0


# ==============================================================================
# 1. 7-Step Pipeline & Flag Normalization
# ==============================================================================

# Canonical pipeline order:
# 1. r / s (repr or str)
# 2. t     (translate)
# 3. i     (interpolate)
# 4. q     (quote text tags)
# 5. f     (say_menu_text_filter)
# 6. u     (uppercase)
# 7. l     (lowercase)
# 8. c     (capitalize first char)
CANONICAL_FLAG_ORDER = ["r", "s", "t", "i", "q", "f", "u", "l", "c"]
FLAG_PRECEDENCE = {flag: idx for idx, flag in enumerate(CANONICAL_FLAG_ORDER)}


def normalize_flags(raw_flags: str) -> str:
    """
    Normalizes Ren'Py conversion flags:
    - Strips all extra exclamation marks (e.g. '!!ti' -> 'ti', '!t!i' -> 'ti').
    - Deduplicates flags.
    - Sorts into canonical pipeline execution order: r/s -> t -> i -> q -> f -> u -> l -> c.
    - Preserves deterministic equality: !cl == !lc -> '!lc'.
    """
    if not raw_flags:
        return ""

    # Clean redundant '!' and whitespace
    cleaned = raw_flags.replace("!", "").strip()
    if not cleaned:
        return ""

    # Unique flags preserving first occurrence of r vs s
    seen: Set[str] = set()
    unique_flags: List[str] = []

    # If both 'r' and 's' are given, keep the one that appeared first
    for ch in cleaned:
        if ch in FLAG_PRECEDENCE and ch not in seen:
            if (ch == "r" and "s" in seen) or (ch == "s" and "r" in seen):
                continue
            seen.add(ch)
            unique_flags.append(ch)

    # Sort flags by canonical pipeline order
    sorted_flags = sorted(unique_flags, key=lambda f: FLAG_PRECEDENCE[f])
    return "!" + "".join(sorted_flags) if sorted_flags else ""


def apply_pipeline(
    value: Any,
    flags: str,
    translate_fn: Optional[Callable[[str], str]] = None,
    filter_fn: Optional[Callable[[str], str]] = None,
    interpolate_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """
    Evaluates value through the 7-step Ren'Py substitution pipeline with flags:
    1. r / s: repr(value) or str(value)
    2. t: translate_fn(value)
    3. i: interpolate_fn(value)
    4. q: quote '{' to '{{' and '[' to '[['
    5. f: filter_fn(value) (config.say_menu_text_filter)
    6. u: value.upper()
    7. l: value.lower()
    8. c: value.capitalize()

    Ensures that flag order does not affect result:
    e.g. '!cl' and '!lc' both execute step 7 (lower) then step 8 (capitalize).
    """
    norm = normalize_flags(flags)
    active_flags = set(norm.replace("!", ""))

    # Step 1: r / s
    if "r" in active_flags:
        text = repr(value)
    else:
        text = str(value)

    # Step 2: t (translate)
    if "t" in active_flags and translate_fn is not None:
        text = translate_fn(text)

    # Step 3: i (nested interpolation)
    if "i" in active_flags and interpolate_fn is not None:
        text = interpolate_fn(text)

    # Step 4: q (quote text tags)
    if "q" in active_flags:
        text = text.replace("{", "{{").replace("[", "[[")

    # Step 5: f (say_menu_text_filter)
    if "f" in active_flags and filter_fn is not None:
        text = filter_fn(text)

    # Step 6: u (uppercase)
    if "u" in active_flags:
        text = text.upper()

    # Step 7: l (lowercase)
    if "l" in active_flags:
        text = text.lower()

    # Step 8: c (capitalize)
    if "c" in active_flags:
        text = text.capitalize()

    return text


# ==============================================================================
# 2. Ruby Text Conversion
# ==============================================================================

# Regex for ruby text: 【bottom｜top】 or 【base|ruby】
RUBY_BRACKET_REGEX = re.compile(r"【(?P<base>[^｜|]+)[｜|](?P<ruby>[^】]+)】")
# Regex for Ren'Py ruby tags: {rb}base{/rb}{rt}ruby{/rt}
RENPY_RUBY_TAG_REGEX = re.compile(r"\{rb\}(?P<base>.*?)\{/rb\}\s*\{rt\}(?P<ruby>.*?)\{/rt\}")


def ruby_to_renpy(text: str) -> str:
    """Converts '【dưới｜trên】' or '【base|ruby】' into Ren'Py '{rb}dưới{/rb}{rt}trên{/rt}'."""
    return RUBY_BRACKET_REGEX.sub(r"{rb}\g<base>{/rb}{rt}\g<ruby>{/rt}", text)


def renpy_to_ruby(text: str) -> str:
    """Converts '{rb}dưới{/rb}{rt}trên{/rt}' into standard '【dưới｜trên】'."""
    return RENPY_RUBY_TAG_REGEX.sub(r"【\g<base>｜\g<ruby>】", text)


# ==============================================================================
# 3. Comprehensive Lexer
# ==============================================================================

# Regex patterns for Ren'Py components
# Escape space: \ followed by space
# Escaped char: \n, \\, \", \'
# Escaped brackets: {{, [[, }}, ]]
# Ruby: 【...｜...】
# Text tag: {...}
# Variable: [...]
MASTER_LEXER_REGEX = re.compile(
    r"(?P<escape_space>\\ )|"
    r"(?P<escaped_bracket>\{\{|\}\}|\[\[|\]\])|"
    r"(?P<escape_char>\\[n\\\"'])|"
    r"(?P<ruby>【[^｜|]+[｜|][^】]+】)|"
    r"(?P<tag>\{[^}]+\})|"
    r"(?P<variable>\[[^\]]+\])"
)

# Regex to parse inside variable: [var_expr!flags]
VAR_PARSER_REGEX = re.compile(r"^\[(?P<expr>[^!\]]+)(?P<flags>!+.*)?\]$")


def tokenize(text: str) -> List[Token]:
    """
    Tokenizes a dialogue string into text segments, escape sequences,
    ruby tags, Ren'Py text tags, and bracketed variables.
    """
    tokens: List[Token] = []
    last_end = 0
    t_count = 0
    v_count = 0

    for match in MASTER_LEXER_REGEX.finditer(text):
        start, end = match.span()
        if start > last_end:
            # Emit plain text preceding the match
            tokens.append(Token(
                type=TokenType.TEXT,
                raw=text[last_end:start],
                value=text[last_end:start],
            ))

        if match.group("escape_space"):
            tokens.append(Token(
                type=TokenType.ESCAPE_SPACE,
                raw=r"\ ",
                value=" ",
                index=t_count,
            ))
            t_count += 1
        elif match.group("escaped_bracket"):
            raw = match.group("escaped_bracket")
            tokens.append(Token(
                type=TokenType.ESCAPED_BRACKET,
                raw=raw,
                value=raw[0],  # {{ -> {, [[ -> [
                index=t_count,
            ))
            t_count += 1
        elif match.group("escape_char"):
            raw = match.group("escape_char")
            tokens.append(Token(
                type=TokenType.ESCAPE_CHAR,
                raw=raw,
                value=raw,
                index=t_count,
            ))
            t_count += 1
        elif match.group("ruby"):
            raw = match.group("ruby")
            tokens.append(Token(
                type=TokenType.RUBY_TEXT,
                raw=raw,
                value=raw,
                index=t_count,
            ))
            t_count += 1
        elif match.group("tag"):
            raw = match.group("tag")
            tokens.append(Token(
                type=TokenType.TEXT_TAG,
                raw=raw,
                value=raw,
                index=t_count,
            ))
            t_count += 1
        elif match.group("variable"):
            raw = match.group("variable")
            v_match = VAR_PARSER_REGEX.match(raw)
            if v_match:
                expr = v_match.group("expr").strip()
                raw_flags = v_match.group("flags") or ""
                norm_flags = normalize_flags(raw_flags)
            else:
                expr = raw[1:-1].strip()
                norm_flags = ""
            tokens.append(Token(
                type=TokenType.VARIABLE,
                raw=raw,
                value=expr,
                flags=norm_flags,
                index=v_count,
            ))
            v_count += 1

        last_end = end

    if last_end < len(text):
        tokens.append(Token(
            type=TokenType.TEXT,
            raw=text[last_end:],
            value=text[last_end:],
        ))

    return tokens


# ==============================================================================
# 4. Invariant Placeholder Masker & Restorer
# ==============================================================================

@dataclass
class PlaceholderMap:
    """
    Stores 1-to-1 mappings between invariant placeholders (<T0>, <V1!ti>)
    and their exact original strings for deterministic round-trip restoration.
    """
    t_map: Dict[str, str] = field(default_factory=dict)       # e.g. "<T0>" -> "{b}"
    v_map: Dict[str, Tuple[str, str]] = field(default_factory=dict) # e.g. "V1" -> ("[player_name!ti]", "!ti")
    placeholder_order: List[str] = field(default_factory=list)

    def get_all_placeholders(self) -> List[str]:
        return list(self.placeholder_order)


def mask_substitutions(text: str) -> Tuple[str, PlaceholderMap]:
    """
    Transforms Ren'Py substitutions and tags into invariant placeholders:
    - Text tags, escapes, ruby text -> '<T0>', '<T1>', ...
    - Variables -> '<V0>', '<V1!ti>', ...
    Returns (masked_string, placeholder_map).
    """
    tokens = tokenize(text)
    pmap = PlaceholderMap()
    masked_parts: List[str] = []

    t_idx = 0
    v_idx = 0

    for tok in tokens:
        if tok.type == TokenType.TEXT:
            masked_parts.append(tok.raw)
        elif tok.type in (
            TokenType.ESCAPE_SPACE,
            TokenType.ESCAPE_CHAR,
            TokenType.ESCAPED_BRACKET,
            TokenType.RUBY_TEXT,
            TokenType.TEXT_TAG,
        ):
            ph = f"<T{t_idx}>"
            pmap.t_map[ph] = tok.raw
            pmap.placeholder_order.append(ph)
            masked_parts.append(ph)
            t_idx += 1
        elif tok.type == TokenType.VARIABLE:
            # e.g. <V1!ti> or <V0> if flags is empty
            flags_suffix = tok.flags  # already has leading '!' or is ""
            ph = f"<V{v_idx}{flags_suffix}>"
            pmap.v_map[f"V{v_idx}"] = (tok.raw, tok.flags)
            pmap.placeholder_order.append(ph)
            masked_parts.append(ph)
            v_idx += 1

    return "".join(masked_parts), pmap


# Regex to detect placeholders during restoration (flexible to whitespace / casing)
RESTORE_T_REGEX = re.compile(r"<\s*[Tt](?P<idx>\d+)\s*>")
RESTORE_V_REGEX = re.compile(r"<\s*[Vv](?P<idx>\d+)(?P<flags>![^>]*?)?\s*>")


def restore_substitutions(
    masked_text: str,
    pmap: PlaceholderMap,
    preserve_model_flags: bool = False,
) -> str:
    """
    Deterministically restores placeholders in masked_text back to original Ren'Py tokens:
    - '<T0>' -> original tag / escape / ruby
    - '<V1!ti>' -> original variable '[expr!ti]' (or with flags adjusted)
    Handles case variations (<t0>, <V0>) and subtle whitespace introduced by LLMs.
    """
    # 1. Restore <T...> placeholders
    def replace_t(match: re.Match[str]) -> str:
        idx = match.group("idx")
        key = f"<T{idx}>"
        return pmap.t_map.get(key, match.group(0))

    restored = RESTORE_T_REGEX.sub(replace_t, masked_text)

    # 2. Restore <V...> placeholders
    def replace_v(match: re.Match[str]) -> str:
        idx = match.group("idx")
        v_key = f"V{idx}"
        if v_key in pmap.v_map:
            orig_raw, orig_flags = pmap.v_map[v_key]
            if preserve_model_flags and match.group("flags"):
                # If requested and LLM modified flags, use normalized LLM flags
                model_flags = normalize_flags(match.group("flags"))
                # Extract expr from orig_raw
                vm = VAR_PARSER_REGEX.match(orig_raw)
                if vm:
                    expr = vm.group("expr")
                    return f"[{expr}{model_flags}]"
            return orig_raw
        return match.group(0)

    restored = RESTORE_V_REGEX.sub(replace_v, restored)
    return restored


def validate_placeholders(
    generated_text: str,
    pmap: PlaceholderMap,
) -> Tuple[bool, List[str], List[str]]:
    """
    Verifies that all expected placeholders are preserved in generated_text.
    Returns:
    (is_valid, missing_placeholders, extra_placeholders)
    """
    expected = set(pmap.placeholder_order)
    found_t = set(RESTORE_T_REGEX.findall(generated_text))
    found_v = set(m.group(0) for m in RESTORE_V_REGEX.finditer(generated_text))

    # Standardize found
    found_normalized: Set[str] = set()
    for m in RESTORE_T_REGEX.finditer(generated_text):
        found_normalized.add(f"<T{m.group('idx')}>")
    for m in RESTORE_V_REGEX.finditer(generated_text):
        flags = normalize_flags(m.group("flags") or "")
        found_normalized.add(f"<V{m.group('idx')}{flags}>")

    missing = [ph for ph in pmap.placeholder_order if ph not in found_normalized]
    extra = [ph for ph in found_normalized if ph not in expected]

    return (len(missing) == 0, missing, extra)


# ==============================================================================
# 5. Object-Oriented Interface Wrapper
# ==============================================================================

class InterpolationLexer:
    """Convenience class providing unified access to interpolation lexer methods."""

    @staticmethod
    def normalize_flags(raw_flags: str) -> str:
        return normalize_flags(raw_flags)

    @staticmethod
    def apply_pipeline(
        value: Any,
        flags: str,
        translate_fn: Optional[Callable[[str], str]] = None,
        filter_fn: Optional[Callable[[str], str]] = None,
        interpolate_fn: Optional[Callable[[str], str]] = None,
    ) -> str:
        return apply_pipeline(
            value=value,
            flags=flags,
            translate_fn=translate_fn,
            filter_fn=filter_fn,
            interpolate_fn=interpolate_fn,
        )

    @staticmethod
    def tokenize(text: str) -> List[Token]:
        return tokenize(text)

    @staticmethod
    def mask(text: str) -> Tuple[str, PlaceholderMap]:
        return mask_substitutions(text)

    @staticmethod
    def restore(
        masked_text: str,
        pmap: PlaceholderMap,
        preserve_model_flags: bool = False,
    ) -> str:
        return restore_substitutions(
            masked_text=masked_text,
            pmap=pmap,
            preserve_model_flags=preserve_model_flags,
        )

    @staticmethod
    def validate(
        generated_text: str,
        pmap: PlaceholderMap,
    ) -> Tuple[bool, List[str], List[str]]:
        return validate_placeholders(generated_text, pmap)

    @staticmethod
    def ruby_to_renpy(text: str) -> str:
        return ruby_to_renpy(text)

    @staticmethod
    def renpy_to_ruby(text: str) -> str:
        return renpy_to_ruby(text)

