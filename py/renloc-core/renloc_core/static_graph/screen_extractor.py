"""
renloc_core/static_graph/screen_extractor.py
Extracts UI localization strings by recursively traversing Screen Language 2 (sl2.slast) ASTs
and tracks LayeredImage group/attribute definitions to enrich dialogue entities with emotional/outfit context.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from renloc_core.ast_reader.ast_nodes import (
    Screen,
    SLScreen,
    SLDisplayable,
    SLIf,
    SLShowIf,
    SLFor,
    SLPython,
    SLUse,
    Show,
    Scene,
    LayeredImage,
    RawLayeredImage,
)
from .dialogue_normalizer import DialogueEntity


@dataclass
class ExtractedUIString:
    """Represents a translatable string found within a Screen Language 2 UI declaration."""
    string_id: str
    text: str
    source_type: str            # 'text', 'textbutton', 'tooltip', 'alt', 'action', 'condition'
    screen_name: str
    filename: str = ""
    linenumber: int = 0
    is_translatable: bool = True
    is_deferred: bool = False

    def __repr__(self) -> str:
        return (
            f"ExtractedUIString(screen='{self.screen_name}', type='{self.source_type}', "
            f"text='{self.text[:25]}', loc={self.filename}:{self.linenumber})"
        )


class ScreenExtractor:
    """
    Traverses sl2.slast AST structures to extract all UI strings (text, textbutton, tooltip, alt).
    """

    RE_STRING_LITERAL = re.compile(r"""(?:_\s*\(\s*)?["'](?P<text>(?:\\.|[^"'\\])*)["']\s*\)?""")
    RE_TRANSLATABLE = re.compile(r"""\b(?P<fn>__|_\b)\s*\(\s*["'](?P<text>(?:\\.|[^"'\\])*)["']\s*\)""")

    def __init__(self):
        self.extracted_strings: List[ExtractedUIString] = []
        self.string_counter = 0

    def extract_from_screens(self, nodes: List[Any]) -> List[ExtractedUIString]:
        """Scans a collection of AST nodes and extracts UI strings from all Screen declarations."""
        results: List[ExtractedUIString] = []
        for item in nodes:
            screen_obj = None
            filename = getattr(item, "filename", "")
            linenumber = getattr(item, "linenumber", 0)

            if isinstance(item, Screen):
                screen_obj = getattr(item, "screen", None)
            elif isinstance(item, SLScreen):
                screen_obj = item

            if screen_obj:
                results.extend(self.extract_screen(screen_obj, default_file=filename, default_line=linenumber))

        self.extracted_strings.extend(results)
        return results

    def extract_screen(self, screen: SLScreen, default_file: str = "", default_line: int = 0) -> List[ExtractedUIString]:
        """Recursively traverses a single SLScreen AST node."""
        screen_name = getattr(screen, "name", "anonymous_screen")
        results: List[ExtractedUIString] = []

        children = getattr(screen, "children", []) or []
        for child in children:
            results.extend(self._traverse_sl_node(child, screen_name=screen_name, filename=default_file, linenumber=default_line))

        return results

    def _clean_text(self, raw_expr: Any) -> Optional[Tuple[str, bool, bool]]:
        """
        Parses an expression string into (clean_text, is_translatable, is_deferred).
        """
        if not isinstance(raw_expr, str):
            raw_expr = str(raw_expr)

        # Check for _(...) or __(..)
        tmatch = self.RE_TRANSLATABLE.search(raw_expr)
        if tmatch:
            is_deferred = (tmatch.group("fn") == "__")
            text = tmatch.group("text").encode().decode("unicode_escape")
            return text, True, is_deferred

        # Plain string literal
        smatch = self.RE_STRING_LITERAL.search(raw_expr)
        if smatch:
            text = smatch.group("text").encode().decode("unicode_escape")
            return text, True, False

        return None

    def _traverse_sl_node(
        self,
        node: Any,
        screen_name: str,
        filename: str = "",
        linenumber: int = 0,
    ) -> List[ExtractedUIString]:
        """Recursively parses an arbitrary SLNode (SLDisplayable, SLIf, SLFor, SLPython, SLUse)."""
        results: List[ExtractedUIString] = []
        loc = getattr(node, "location", (filename, linenumber))
        file_loc = loc[0] if isinstance(loc, (tuple, list)) and len(loc) > 0 else filename
        line_loc = loc[1] if isinstance(loc, (tuple, list)) and len(loc) > 1 else linenumber

        # 1. SLDisplayable
        if isinstance(node, SLDisplayable):
            disp_type = getattr(node, "name", "") or getattr(node, "displayable_type", "")
            positional = getattr(node, "positional", []) or []
            keywords = getattr(node, "keyword", []) or []

            # (a) Check positional text
            if positional and disp_type in ("text", "textbutton", "label"):
                first_arg = positional[0]
                parsed = self._clean_text(first_arg)
                if parsed:
                    text, is_tl, is_def = parsed
                    self.string_counter += 1
                    s = ExtractedUIString(
                        string_id=f"ui_{screen_name}_{self.string_counter}",
                        text=text,
                        source_type=disp_type,
                        screen_name=screen_name,
                        filename=file_loc,
                        linenumber=line_loc,
                        is_translatable=is_tl,
                        is_deferred=is_def,
                    )
                    results.append(s)

            # (b) Check keyword arguments: tooltip, alt, text
            for kw_name, kw_val in keywords:
                if kw_name in ("tooltip", "alt", "text"):
                    parsed = self._clean_text(kw_val)
                    if parsed:
                        text, is_tl, is_def = parsed
                        self.string_counter += 1
                        s = ExtractedUIString(
                            string_id=f"ui_{screen_name}_{kw_name}_{self.string_counter}",
                            text=text,
                            source_type=kw_name,
                            screen_name=screen_name,
                            filename=file_loc,
                            linenumber=line_loc,
                            is_translatable=is_tl,
                            is_deferred=is_def,
                        )
                        results.append(s)

            # (c) Recurse children
            for child in getattr(node, "children", []) or []:
                results.extend(self._traverse_sl_node(child, screen_name, file_loc, line_loc))

        # 2. SLIf / SLShowIf
        elif isinstance(node, (SLIf, SLShowIf)):
            entries = getattr(node, "entries", []) or []
            for cond_expr, branch_children in entries:
                # Scan condition for translatable strings
                if isinstance(cond_expr, str):
                    for match in self.RE_TRANSLATABLE.finditer(cond_expr):
                        is_deferred = (match.group("fn") == "__")
                        text = match.group("text").encode().decode("unicode_escape")
                        self.string_counter += 1
                        results.append(ExtractedUIString(
                            string_id=f"ui_{screen_name}_cond_{self.string_counter}",
                            text=text,
                            source_type="condition",
                            screen_name=screen_name,
                            filename=file_loc,
                            linenumber=line_loc,
                            is_translatable=True,
                            is_deferred=is_deferred,
                        ))
                for child in branch_children:
                    results.extend(self._traverse_sl_node(child, screen_name, file_loc, line_loc))

        # 3. SLFor
        elif isinstance(node, SLFor):
            for child in getattr(node, "children", []) or []:
                results.extend(self._traverse_sl_node(child, screen_name, file_loc, line_loc))

        # 4. SLPython
        elif isinstance(node, SLPython):
            code_obj = getattr(node, "code", None)
            code_src = getattr(code_obj, "source", "") or (str(code_obj) if code_obj else "")
            for match in self.RE_TRANSLATABLE.finditer(code_src):
                is_deferred = (match.group("fn") == "__")
                text = match.group("text").encode().decode("unicode_escape")
                self.string_counter += 1
                results.append(ExtractedUIString(
                    string_id=f"ui_{screen_name}_py_{self.string_counter}",
                    text=text,
                    source_type="python",
                    screen_name=screen_name,
                    filename=file_loc,
                    linenumber=line_loc,
                    is_translatable=True,
                    is_deferred=is_deferred,
                ))

        # 5. SLUse
        elif isinstance(node, SLUse):
            target = getattr(node, "target", "")
            block = getattr(node, "block", None)
            if block:
                for child in getattr(block, "children", []) or []:
                    results.extend(self._traverse_sl_node(child, screen_name, file_loc, line_loc))

        return results


class LayeredImageTracker:
    """
    Maintains layered image definitions (attributes, groups, conditions)
    and tracks active character visual state across Show and Scene AST statements.
    """

    def __init__(self):
        # character_tag -> set of defined attributes/groups
        self.image_definitions: Dict[str, Set[str]] = {}
        # character_tag -> set of active attributes (e.g. {'happy', 'uniform'})
        self.active_state: Dict[str, Set[str]] = {}

    def register_layered_image(self, node: Union[LayeredImage, RawLayeredImage, Any]):
        """Registers a LayeredImage definition and extracts all possible attribute tags."""
        name = getattr(node, "name", None)
        if not name:
            return

        tag = name if isinstance(name, str) else name[0]
        if tag not in self.image_definitions:
            self.image_definitions[tag] = set()

        raw_attrs = getattr(node, "attributes", []) or []
        for a in raw_attrs:
            attr_name = getattr(a, "attribute", None) or getattr(a, "name", None)
            if attr_name:
                self.image_definitions[tag].add(str(attr_name))

    def process_show_or_scene(self, stmt: Union[Show, Scene, Any]) -> Dict[str, Set[str]]:
        """
        Updates active character state based on a Show or Scene statement.
        imspec format: (character_name, attribute1, attribute2, ...)
        """
        imspec = getattr(stmt, "imspec", None)
        if not imspec:
            return self.active_state

        char_tag = imspec[0]
        attributes = set(imspec[1:])

        if isinstance(stmt, Scene):
            # Scene clears layer and sets background/character
            self.active_state.clear()
            self.active_state[char_tag] = attributes
        elif isinstance(stmt, Show):
            # Show updates or adds attributes for the given character
            if char_tag not in self.active_state:
                self.active_state[char_tag] = set()
            self.active_state[char_tag].update(attributes)

        return self.active_state

    def get_active_tags(self, character: Optional[str]) -> Set[str]:
        """Returns the current emotional / costume tags active for the given character."""
        if not character or character not in self.active_state:
            return set()
        return set(self.active_state[character])

    def tag_dialogue(self, entity: DialogueEntity):
        """Attaches active visual/costume state tags to a DialogueEntity."""
        if entity.who:
            tags = self.get_active_tags(entity.who)
            if tags:
                entity.context_tags.update(tags)
