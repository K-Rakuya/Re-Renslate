"""
renloc_core/ast_reader/code_generator.py
Exact Ren'Py SDK code generator for Say, TranslateSay, and related AST nodes.
Replicates canonical string escaping, whitespace preservation, attribute/@ temporary attributes handling,
and 8-character hex MD5 translation identifier calculation matching renpy.translation.
"""
import re
import hashlib
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


def encode_say_string(s: str) -> str:
    """
    Encodes a string in the canonical format used by Ren'Py say statements.
    Exact implementation from renpy.translation.encode_say_string:
      - Escapes backslashes: '\\' -> '\\\\'
      - Escapes newlines: '\\n' -> '\\\\n'
      - Escapes quotes: '"' -> '\\"'
      - Escapes consecutive spaces: (?<= )  -> '\\ '
      - Wraps resulting string in double quotes
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\\", "\\\\")
    s = s.replace("\n", "\\n")
    s = s.replace('"', '\\"')
    s = re.sub(r"(?<= ) ", r"\ ", s)
    return '"' + s + '"'


def get_say_code(
    who: Optional[str] = None,
    what: str = "",
    attributes: Optional[Tuple[str, ...]] = None,
    temporary_attributes: Optional[Tuple[str, ...]] = None,
    interact: bool = True,
    identifier: Optional[str] = None,
    explicit_identifier: bool = False,
    arguments: Optional[Any] = None,
    with_: Optional[str] = None,
    dialogue_filter: Optional[Callable[[str], str]] = None,
) -> str:
    """
    Returns the canonical form of the code corresponding to a Say / TranslateSay statement.
    Faithfully mirrors renpy.ast.Say.get_code(dialogue_filter=...) from Ren'Py SDK 7.x-8.5.3.

    Parameter order and rules:
      1. who (character name or expression string)
      2. attributes (if present, space-separated)
      3. temporary_attributes (if present, preceded by '@')
      4. dialogue_filter(what) -> encode_say_string
      5. 'nointeract' (if interact is False)
      6. 'id <identifier>' (if identifier is present and explicit_identifier is True)
      7. arguments.get_code() (if arguments present)
      8. 'with <with_>' (if with_ is present, always at the end)
    """
    rv: List[str] = []

    if who:
        rv.append(who)

    if attributes is not None:
        rv.extend(str(a) for a in attributes)

    if temporary_attributes:
        rv.append("@")
        rv.extend(str(a) for a in temporary_attributes)

    if dialogue_filter is not None:
        what = dialogue_filter(what)

    rv.append(encode_say_string(what))

    if not interact:
        rv.append("nointeract")

    if identifier and explicit_identifier:
        rv.append("id")
        rv.append(identifier)

    if arguments:
        if hasattr(arguments, "get_code"):
            rv.append(arguments.get_code())
        elif isinstance(arguments, str):
            rv.append(arguments)

    if with_:
        rv.append("with")
        rv.append(str(with_))

    return " ".join(rv)


def get_node_code(node: Any, dialogue_filter: Optional[Callable[[str], str]] = None) -> str:
    """
    Extracts canonical code from any supported AST node.
    """
    if hasattr(node, "get_code"):
        return node.get_code(dialogue_filter=dialogue_filter)

    # Check for Say / TranslateSay duck types
    if hasattr(node, "what"):
        return get_say_code(
            who=getattr(node, "who", None),
            what=getattr(node, "what", ""),
            attributes=getattr(node, "attributes", None),
            temporary_attributes=getattr(node, "temporary_attributes", None),
            interact=getattr(node, "interact", True),
            identifier=getattr(node, "identifier", None),
            explicit_identifier=getattr(node, "explicit_identifier", False),
            arguments=getattr(node, "arguments", None),
            with_=getattr(node, "with_", None),
            dialogue_filter=dialogue_filter,
        )

    if hasattr(node, "line"):
        return getattr(node, "line")

    if hasattr(node, "rest"):
        return "rpy " + " ".join(getattr(node, "rest"))

    raise NotImplementedError(f"Node {type(node).__name__} does not implement get_code()")


def compute_translation_hash(
    statements: Union[str, Any, List[Any]],
    label: Optional[str] = None,
    existing_ids: Optional[Set[str]] = None,
    only_considers_say: bool = True,
) -> str:
    """
    Calculates the 8-character hex MD5 translation identifier strictly identical
    to renpy.translation.TranslateInfo / create_translate.

    Algorithm:
      1. Normalize statements into code strings.
      2. If only_considers_say is True and any statement has a 'what' attribute (Say),
         only hash Say statements.
      3. For each statement code:
           md5.update((code + "\\r\\n").encode("utf-8"))
      4. digest = md5.hexdigest()[:8]
      5. base = (label.replace(".", "_") + "_" + digest) if label else digest
      6. If base in existing_ids, append _1, _2, ... until collision-free.
    """
    if isinstance(statements, str):
        raw_list = [statements]
    elif isinstance(statements, list):
        raw_list = statements
    else:
        raw_list = [statements]

    # Handle renpy.config.tlid_only_considers_say behavior
    digest_items = raw_list
    if only_considers_say and any(hasattr(i, "what") for i in raw_list):
        digest_items = [i for i in raw_list if hasattr(i, "what")]

    codes: List[str] = []
    for item in digest_items:
        if isinstance(item, str):
            codes.append(item)
        else:
            codes.append(get_node_code(item))

    md5 = hashlib.md5()
    for code in codes:
        md5.update((code + "\r\n").encode("utf-8"))

    digest = md5.hexdigest()[:8]

    if label is None:
        base = digest
    else:
        base = label.replace(".", "_") + "_" + digest

    identifier = base
    if existing_ids is not None:
        i = 0
        suffix = ""
        while True:
            candidate = base + suffix
            if candidate not in existing_ids:
                identifier = candidate
                break
            i += 1
            suffix = f"_{i}"

    return identifier


def generate_translation_id(
    statement: Union[str, Any, List[Any]],
    label: Optional[str] = None,
    existing_ids: Optional[Set[str]] = None,
) -> str:
    """
    High-level entry point to compute or resolve a translation identifier.
    Respects explicit statement 'id' clauses if present.
    """
    # Check for explicit identifier on single AST node
    if not isinstance(statement, (str, list)) and getattr(statement, "explicit_identifier", False):
        explicit_id = getattr(statement, "identifier", None)
        if explicit_id:
            return explicit_id

    return compute_translation_hash(statement, label=label, existing_ids=existing_ids)


def format_translate_block(
    identifier: str,
    statements: List[Any],
    language: Optional[str] = None,
    indent: str = "    ",
) -> str:
    """
    Renders a standard Ren'Py translation block:
      translate [language] [identifier]:
          <statements>
    """
    lang_str = f"{language} " if language else ""
    header = f"translate {lang_str}{identifier}:"
    body_lines = []
    for stmt in statements:
        if isinstance(stmt, str):
            body_lines.append(f"{indent}{stmt}")
        else:
            code = get_node_code(stmt)
            body_lines.append(f"{indent}{code}")
    return header + "\n" + "\n".join(body_lines) + "\n"
