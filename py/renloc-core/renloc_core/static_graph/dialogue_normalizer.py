"""
renloc_core/static_graph/dialogue_normalizer.py
Normalizes Ren'Py dialogue across 4 canonical forms into unified DialogueEntity records:
  Form 1: Explicit character dialogue (who != None)
  Form 2: Implicit narration (who == None)
  Form 3: Continuation statements (extend inherits speaker and dialogue counter from prior line)
  Form 4: Direct renpy.say(who, what) invocations and translatable string markers _(s) / __(s)
"""
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from renloc_core.ast_reader.ast_nodes import Say, TranslateSay, Python, EarlyPython, UserStatement
from renloc_core.ast_reader.code_generator import generate_translation_id
from .lifecycle import WrapNode


class DialogueForm(str, Enum):
    EXPLICIT = "EXPLICIT"                       # Form 1: who != None
    NARRATION = "NARRATION"                     # Form 2: who is None
    EXTEND = "EXTEND"                           # Form 3: extend keyword
    RENPY_SAY = "RENPY_SAY"                     # Form 4a: renpy.say(who, what)
    TRANSLATABLE_STRING = "TRANSLATABLE_STRING" # Form 4b: _(s) or __(s)


@dataclass
class DialogueEntity:
    """
    Unified semantic representation of any dialogue or translatable unit in Ren'Py.
    """
    entity_id: str
    who: Optional[str]
    what: str
    form: DialogueForm
    filename: str
    linenumber: int
    block_depth: int
    translation_id: str
    attributes: Tuple[str, ...] = ()
    temporary_attributes: Tuple[str, ...] = ()
    with_: Optional[str] = None
    interact: bool = True
    is_deferred: bool = False                   # True for __(s), False for _(s)
    parent_id: Optional[str] = None             # For extend, points to preceding dialogue
    dialogue_index: int = 0                     # Sequence counter
    context_tags: Set[str] = field(default_factory=set) # Emotional/outfit tags from LayeredImage
    raw_node: Any = None

    def __repr__(self) -> str:
        who_str = self.who or "None"
        what_preview = self.what[:25] + "..." if len(self.what) > 25 else self.what
        return (
            f"DialogueEntity(id={self.entity_id}, form={self.form.value}, who='{who_str}', "
            f"what='{what_preview}', tlid={self.translation_id}, idx={self.dialogue_index})"
        )


class DialogueNormalizer:
    """
    Scans AST streams and normalizes all dialogue and translatable strings
    into unified DialogueEntity instances with correct extend inheritance.
    """

    # Regex patterns for Python-embedded dialogues:
    # renpy.say(who, "what") or renpy.say(s, _("what"))
    RE_RENPY_SAY = re.compile(
        r"""renpy\.say\s*\(\s*(?P<who>[A-Za-z0-9_]+|None)\s*,\s*(?:_\s*\(\s*)?["'](?P<what>(?:\\.|[^"'\\])*)["']\s*\)?""",
        re.DOTALL
    )

    # _("string") or _('string') - immediate translation
    RE_TRANSLATABLE_IMMEDIATE = re.compile(
        r"""(?<!_)\b_\s*\(\s*(?P<quote>["'])(?P<text>(?:\\.|(?!\1).)*)\1\s*\)"""
    )

    # __("string") or __('string') - deferred translation
    RE_TRANSLATABLE_DEFERRED = re.compile(
        r"""\b__\s*\(\s*(?P<quote>["'])(?P<text>(?:\\.|(?!\1).)*)\1\s*\)"""
    )

    def __init__(self):
        self.dialogue_counter = 0
        self.last_dialogue: Optional[DialogueEntity] = None
        self.entities: List[DialogueEntity] = []

    def normalize_stream(
        self,
        nodes: List[Union[WrapNode, Any]],
        current_label: Optional[str] = None,
    ) -> List[DialogueEntity]:
        """
        Processes a sequence of AST nodes or WrapNodes, producing a normalized
        stream of DialogueEntity objects with accurate extend continuity.
        """
        results: List[DialogueEntity] = []

        for item in nodes:
            # Extract underlying node and location metadata
            if isinstance(item, WrapNode):
                node = item.node
                filename = item.filename
                linenumber = item.linenumber
                depth = item.block_depth
            else:
                node = item
                filename = getattr(node, "filename", "script.rpy")
                linenumber = getattr(node, "linenumber", 0)
                depth = 0

            # 1. Say / TranslateSay statement
            if isinstance(node, (Say, TranslateSay)):
                who = getattr(node, "who", None)
                what = getattr(node, "what", "")
                attrs = getattr(node, "attributes", None) or ()
                temp_attrs = getattr(node, "temporary_attributes", None) or ()
                with_ = getattr(node, "with_", None)
                interact = getattr(node, "interact", True)

                self.dialogue_counter += 1
                entity_id = f"dlg_{filename}_{linenumber}_{self.dialogue_counter}"

                # Form 3: extend keyword
                if who == "extend":
                    parent_id = self.last_dialogue.entity_id if self.last_dialogue else None
                    inherited_who = self.last_dialogue.who if self.last_dialogue else None
                    inherited_attrs = attrs if attrs else (self.last_dialogue.attributes if self.last_dialogue else ())
                    tlid = generate_translation_id(node, label=current_label)

                    entity = DialogueEntity(
                        entity_id=entity_id,
                        who=inherited_who,
                        what=what,
                        form=DialogueForm.EXTEND,
                        filename=filename,
                        linenumber=linenumber,
                        block_depth=depth,
                        translation_id=tlid,
                        attributes=tuple(inherited_attrs),
                        temporary_attributes=tuple(temp_attrs),
                        with_=with_,
                        interact=interact,
                        parent_id=parent_id,
                        dialogue_index=self.dialogue_counter,
                        raw_node=node,
                    )

                # Form 1: Explicit character
                elif who is not None:
                    tlid = getattr(node, "identifier", None) or generate_translation_id(node, label=current_label)
                    entity = DialogueEntity(
                        entity_id=entity_id,
                        who=who,
                        what=what,
                        form=DialogueForm.EXPLICIT,
                        filename=filename,
                        linenumber=linenumber,
                        block_depth=depth,
                        translation_id=tlid,
                        attributes=tuple(attrs),
                        temporary_attributes=tuple(temp_attrs),
                        with_=with_,
                        interact=interact,
                        dialogue_index=self.dialogue_counter,
                        raw_node=node,
                    )

                # Form 2: Narration (who is None)
                else:
                    tlid = getattr(node, "identifier", None) or generate_translation_id(node, label=current_label)
                    entity = DialogueEntity(
                        entity_id=entity_id,
                        who=None,
                        what=what,
                        form=DialogueForm.NARRATION,
                        filename=filename,
                        linenumber=linenumber,
                        block_depth=depth,
                        translation_id=tlid,
                        attributes=tuple(attrs),
                        temporary_attributes=tuple(temp_attrs),
                        with_=with_,
                        interact=interact,
                        dialogue_index=self.dialogue_counter,
                        raw_node=node,
                    )

                self.last_dialogue = entity
                results.append(entity)
                continue

            # 2. Python statements ($ renpy.say(...) or _()/__() in code)
            code_str = ""
            if isinstance(node, (Python, EarlyPython)):
                code_obj = getattr(node, "code", None)
                if hasattr(code_obj, "source"):
                    code_str = code_obj.source
                elif isinstance(code_obj, str):
                    code_str = code_obj
            elif isinstance(node, UserStatement):
                code_str = getattr(node, "line", "")

            if code_str:
                # 4a. Check direct renpy.say(who, what)
                for match in self.RE_RENPY_SAY.finditer(code_str):
                    who_val = match.group("who")
                    who = None if who_val == "None" else who_val
                    what = match.group("what").encode().decode("unicode_escape")

                    self.dialogue_counter += 1
                    entity_id = f"dlg_{filename}_{linenumber}_{self.dialogue_counter}"
                    tlid = generate_translation_id(f'{who or ""} "{what}"', label=current_label)

                    entity = DialogueEntity(
                        entity_id=entity_id,
                        who=who,
                        what=what,
                        form=DialogueForm.RENPY_SAY,
                        filename=filename,
                        linenumber=linenumber,
                        block_depth=depth,
                        translation_id=tlid,
                        dialogue_index=self.dialogue_counter,
                        raw_node=node,
                    )
                    self.last_dialogue = entity
                    results.append(entity)

                # 4b. Translatable strings _(s) [immediate]
                for match in self.RE_TRANSLATABLE_IMMEDIATE.finditer(code_str):
                    text = match.group("text")
                    self.dialogue_counter += 1
                    entity_id = f"str_{filename}_{linenumber}_{self.dialogue_counter}"
                    tlid = generate_translation_id(f'"{text}"', label=current_label)

                    entity = DialogueEntity(
                        entity_id=entity_id,
                        who=None,
                        what=text,
                        form=DialogueForm.TRANSLATABLE_STRING,
                        filename=filename,
                        linenumber=linenumber,
                        block_depth=depth,
                        translation_id=tlid,
                        is_deferred=False,
                        dialogue_index=self.dialogue_counter,
                        raw_node=node,
                    )
                    results.append(entity)

                # 4c. Translatable strings __(s) [deferred]
                for match in self.RE_TRANSLATABLE_DEFERRED.finditer(code_str):
                    text = match.group("text")
                    self.dialogue_counter += 1
                    entity_id = f"str_{filename}_{linenumber}_{self.dialogue_counter}"
                    tlid = generate_translation_id(f'"{text}"', label=current_label)

                    entity = DialogueEntity(
                        entity_id=entity_id,
                        who=None,
                        what=text,
                        form=DialogueForm.TRANSLATABLE_STRING,
                        filename=filename,
                        linenumber=linenumber,
                        block_depth=depth,
                        translation_id=tlid,
                        is_deferred=True,
                        dialogue_index=self.dialogue_counter,
                        raw_node=node,
                    )
                    results.append(entity)

        self.entities.extend(results)
        return results
