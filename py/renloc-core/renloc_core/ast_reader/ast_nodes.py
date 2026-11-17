"""
renloc_core/ast_reader/ast_nodes.py
Pure data container representations (inert mock AST nodes) for Ren'Py AST objects.
These classes hold AST fields safely without executing any methods or unpickling untrusted logic.
Supports Ren'Py 7.x, 8.0-8.3, and 8.4-8.5.3 (including cslots compressed slots).
"""
from typing import Any, List, Tuple, Optional, Dict, Set, Callable

class FakeStrictBase:
    """Base class for all inert AST objects created by NativeRenpyUnpickler."""
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __setstate__(self, state):
        if isinstance(state, dict):
            for k, v in state.items():
                setattr(self, k, v)
        elif isinstance(state, tuple):
            for item in state:
                if isinstance(item, dict):
                    for k, v in item.items():
                        setattr(self, k, v)

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"{self.__class__.__name__}({attrs})"


# --- renpy.ast nodes ---
class Node(FakeStrictBase):
    filename: str = ""
    linenumber: int = 0
    col_offset: int = 0
    next: Optional['Node'] = None
    name: Any = None
    _name: Any = None
    name_version: int = 0
    name_serial: int = 0
    translatable: bool = False

    def __init__(self, loc=("", 0), *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(loc, (tuple, list)) and len(loc) >= 2:
            self.filename = loc[0]
            self.linenumber = loc[1]
        elif isinstance(loc, str):
            self.filename = loc


class Say(Node):
    who: Optional[str] = None
    who_fast: bool = False
    what: str = ""
    with_: Optional[str] = None
    interact: bool = True
    attributes: Optional[Tuple[str, ...]] = None
    arguments: Optional[Any] = None
    temporary_attributes: Optional[Tuple[str, ...]] = None
    rollback: str = "force"
    identifier: Optional[str] = None
    explicit_identifier: bool = False
    translatable: bool = True

    def __init__(
        self,
        who=None,
        what="",
        with_=None,
        interact=True,
        attributes=None,
        arguments=None,
        temporary_attributes=None,
        rollback="force",
        identifier=None,
        explicit_identifier=False,
        loc=("", 0),
        **kwargs
    ):
        super().__init__(loc=loc, **kwargs)
        self.who = who
        self.what = what
        self.with_ = with_
        self.interact = interact
        self.attributes = attributes
        self.arguments = arguments
        self.temporary_attributes = temporary_attributes
        self.rollback = rollback
        self.identifier = identifier
        self.explicit_identifier = explicit_identifier or (identifier is not None)

    def get_code(self, dialogue_filter: Optional[Callable[[str], str]] = None) -> str:
        from .code_generator import get_say_code
        return get_say_code(
            who=self.who,
            what=self.what,
            attributes=self.attributes,
            temporary_attributes=self.temporary_attributes,
            interact=self.interact,
            identifier=self.identifier,
            explicit_identifier=self.explicit_identifier,
            arguments=self.arguments,
            with_=self.with_,
            dialogue_filter=dialogue_filter,
        )


class Translate(Node):
    identifier: str = ""
    alternate: Optional[str] = None
    language: Optional[str] = None
    block: List[Node] = []

    def __init__(self, loc=("", 0), identifier="", language=None, block=None, alternate=None, **kwargs):
        super().__init__(loc=loc, **kwargs)
        self.identifier = identifier
        self.language = language
        self.block = block or []
        self.alternate = alternate


class TranslateSay(Say):
    """
    A node combining a Translate block and an inner Say statement.
    Inherits from Say so all dialogue properties and get_code() work identically.
    """
    translatable: bool = True
    translation_relevant: bool = True
    alternate: Optional[str] = None
    language: Optional[str] = None

    def __init__(
        self,
        loc=("", 0),
        who=None,
        what="",
        with_=None,
        interact=True,
        attributes=None,
        arguments=None,
        temporary_attributes=None,
        identifier=None,
        language=None,
        alternate=None,
        explicit_identifier=False,
        **kwargs
    ):
        super().__init__(
            who=who,
            what=what,
            with_=with_,
            interact=interact,
            attributes=attributes,
            arguments=arguments,
            temporary_attributes=temporary_attributes,
            identifier=None,
            explicit_identifier=explicit_identifier,
            loc=loc,
            **kwargs
        )
        self.identifier = identifier or ""
        self.language = language
        self.alternate = alternate
        self.explicit_identifier = explicit_identifier


class EndTranslate(Node):
    pass


class Label(Node):
    name: str = ""
    parameters: Optional[Any] = None
    block: List[Node] = []
    hide: bool = False

    def __init__(self, name="", loc=("", 0), **kwargs):
        super().__init__(loc=loc, **kwargs)
        self.name = name


class Jump(Node):
    target: str = ""
    expression: bool = False


class Call(Node):
    label: str = ""
    arguments: Optional[Any] = None
    expression: bool = False


class Return(Node):
    expression: Optional[str] = None


class Scene(Node):
    imspec: Optional[Tuple[str, ...]] = None
    layer: Optional[str] = None
    atl: Optional[Any] = None


class Show(Node):
    imspec: Optional[Tuple[str, ...]] = None
    atl: Optional[Any] = None


class Hide(Node):
    imspec: Optional[Tuple[str, ...]] = None


class With(Node):
    expr: str = ""


class Python(Node):
    code: Any = None
    hide: bool = False
    store: str = "store"


class EarlyPython(Node):
    code: Any = None
    store: str = "store"


class Init(Node):
    priority: int = 0
    block: List[Node] = []

    def __init__(self, priority=0, block=None, loc=("", 0), **kwargs):
        super().__init__(loc=loc, **kwargs)
        self.priority = priority
        self.block = block or []


class Default(Node):
    varname: str = ""
    code: Any = None
    store: str = "store"


class Define(Node):
    varname: str = ""
    code: Any = None
    store: str = "store"
    operator: str = "="
    index: Optional[Any] = None


class UserStatement(Node):
    line: str = ""
    block: List[Tuple[str, int, List[Any]]] = []
    translatable: bool = False

    def __init__(self, loc=("", 0), line="", **kwargs):
        super().__init__(loc=loc, **kwargs)
        self.line = line

    def get_code(self, dialogue_filter: Optional[Callable[[str], str]] = None) -> str:
        return self.line


class RPY(Node):
    rest: List[str] = []

    def __init__(self, loc=("", 0), rest=None, **kwargs):
        super().__init__(loc=loc, **kwargs)
        self.rest = rest or []

    def get_code(self, dialogue_filter: Optional[Callable[[str], str]] = None) -> str:
        return "rpy " + " ".join(self.rest)


class Image(Node):
    imgname: Tuple[str, ...] = ()
    code: Any = None
    atl: Any = None


class Screen(Node):
    screen: Any = None


class PyCode(FakeStrictBase):
    source: str = ""
    location: Tuple[str, int] = ("", 0)
    mode: str = "exec"
    bytecode: Optional[bytes] = None
    py: int = 3
    hashcode: Optional[int] = None
    col_offset: int = 0
    filename: str = ""
    linenumber: int = 0

    def __init__(
        self,
        source: str = "",
        loc: Tuple[str, int] = ("", 0),
        mode: str = "exec",
        py: int = 3,
        bytecode: Optional[bytes] = None,
        hashcode: Optional[int] = None,
        col_offset: int = 0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.source = source
        self.location = loc
        if isinstance(loc, (tuple, list)) and len(loc) >= 2:
            self.filename = loc[0]
            self.linenumber = loc[1]
        self.mode = mode
        self.py = py
        self.bytecode = bytecode
        self.hashcode = hashcode
        self.col_offset = col_offset

    def __setstate__(self, state):
        if isinstance(state, dict):
            super().__setstate__(state)
            return

        if isinstance(state, tuple):
            # Check for (None, slots_dict) from cslots or __slots__
            has_dict = False
            for item in state:
                if isinstance(item, dict):
                    has_dict = True
                    for k, v in item.items():
                        setattr(self, k, v)
            if has_dict:
                return

            # Match Ren'Py PyCode.__getstate__ variations:
            # (version, source, location, mode[, py[, hashcode[, col_offset]]])
            if len(state) >= 4:
                self.source = state[1]
                self.location = state[2]
                if isinstance(self.location, (tuple, list)) and len(self.location) >= 2:
                    self.filename = self.location[0]
                    self.linenumber = self.location[1]
                self.mode = state[3]
                if len(state) >= 5:
                    self.py = state[4]
                if len(state) >= 6:
                    self.hashcode = state[5]
                if len(state) >= 7:
                    self.col_offset = state[6]
                return

        super().__setstate__(state)


# --- renpy.parameter ---
class ArgumentInfo(FakeStrictBase):
    """Inert representation of renpy.parameter.ArgumentInfo."""
    arguments: List[Tuple[Optional[str], str]] = []
    starred_indexes: Set[int] = set()
    doublestarred_indexes: Set[int] = set()

    def __init__(self, arguments=None, starred_indexes=None, doublestarred_indexes=None, **kwargs):
        super().__init__(**kwargs)
        self.arguments = list(arguments) if arguments is not None else []
        self.starred_indexes = set(starred_indexes) if starred_indexes is not None else set()
        self.doublestarred_indexes = set(doublestarred_indexes) if doublestarred_indexes is not None else set()

    def get_code(self) -> str:
        l = []
        for i, (keyword, expression) in enumerate(self.arguments):
            if i in self.starred_indexes:
                l.append("*" + str(expression))
            elif i in self.doublestarred_indexes:
                l.append("**" + str(expression))
            elif keyword is not None:
                l.append(f"{keyword}={expression}")
            else:
                l.append(str(expression))
        return "(" + ", ".join(l) + ")"

    __str__ = get_code


class ParameterInfo(FakeStrictBase):
    pass


class Signature(FakeStrictBase):
    pass


# --- Revertable Containers ---
class RevertableList(list):
    pass


class RevertableDict(dict):
    pass


class RevertableSet(set):
    pass


class RevertableObject(FakeStrictBase):
    pass


# --- SL2 / SLAST (Screen Language AST) ---
class SLNode(FakeStrictBase):
    serial: int = 0
    location: Tuple[str, int] = ("", 0)


class SLBlock(SLNode):
    keyword: List[Tuple[str, Any]] = []
    children: List[Any] = []
    variable: Optional[str] = None
    atl_transform: Any = None
    transform: Any = None


class SLScreen(SLBlock):
    name: str = ""
    modal: str = "False"
    zorder: str = "0"
    tag: Optional[str] = None
    variant: Optional[str] = None
    layer: str = "'screens'"
    sensitive: str = "True"
    roll_forward: str = "None"
    docstring: Optional[str] = None
    version: int = 0
    const_ast: Optional['SLScreen'] = None
    not_const_ast: Optional['SLScreen'] = None
    analysis: Any = None


class SLDisplayable(SLBlock):
    displayable: Any = None
    displayable_type: str = ""
    positional: List[Any] = []
    keyword: List[Tuple[str, Any]] = []
    children: List[Any] = []
    scope: bool = False
    child_or_fixed: bool = False
    style: Optional[str] = None
    text_style: Optional[str] = None
    pass_context: bool = False
    imagemap: bool = False
    replaces: bool = False
    default_keywords: Any = None
    hotspot: bool = False
    variable: Optional[str] = None
    name: str = ""
    unique: bool = True
    local_constant: List[Any] = []
    constant: int = 0
    has_keyword: bool = False
    last_keyword: bool = False


class SLIf(SLNode):
    entries: List[Tuple[str, List[Any]]] = []


class SLShowIf(SLNode):
    entries: List[Tuple[str, List[Any]]] = []


class SLFor(SLBlock):
    variable: str = ""
    expression: str = ""
    children: List[Any] = []


class SLPython(SLNode):
    code: Any = None


class SLUse(SLNode):
    target: str = ""
    args: Any = None
    block: Optional[Any] = None


class SLCustomUse(SLNode):
    pass


class SLTransclude(SLNode):
    pass


class SLDefault(SLNode):
    pass


class SLPass(SLNode):
    pass


class SLBreak(SLNode):
    pass


class SLContinue(SLNode):
    pass


# --- LayeredImage ---
class Layer(FakeStrictBase):
    pass


class Attribute(Layer):
    group: Optional[str] = None
    attribute: str = ""


class Condition(Layer):
    condition: str = ""


class ConditionGroup(Layer):
    pass


class Always(Layer):
    pass


class LayeredImage(FakeStrictBase):
    name: Optional[str] = None
    attributes: List[Any] = []
    at: List[Any] = []
    image_format: Optional[str] = None
    format_function: Any = None
    attribute_function: Any = None
    offer_screen: Optional[bool] = None


class RawLayeredImage(FakeStrictBase):
    name: Optional[str] = None
    attributes: List[Any] = []
    at: List[Any] = []


class RawAttribute(FakeStrictBase):
    group: Optional[str] = None
    attribute: str = ""
    displayable: Any = None


class RawAttributeGroup(FakeStrictBase):
    name: str = ""
    attributes: List[Any] = []


class RawCondition(FakeStrictBase):
    condition: str = ""
    attributes: List[Any] = []


class RawConditionGroup(FakeStrictBase):
    conditions: List[Any] = []


class RawAlways(FakeStrictBase):
    displayable: Any = None


# --- ATL (Animation and Transformation Language) ---
class RawBlock(FakeStrictBase):
    statements: List[Any] = []


class RawMultiloop(FakeStrictBase):
    pass


class RawChild(FakeStrictBase):
    pass


class RawChoice(FakeStrictBase):
    pass


class RawContains(FakeStrictBase):
    pass


class RawEvent(FakeStrictBase):
    pass


class RawFunction(FakeStrictBase):
    pass


class RawIf(FakeStrictBase):
    pass


class RawOn(FakeStrictBase):
    pass


class RawParallel(FakeStrictBase):
    pass


class RawRepeat(FakeStrictBase):
    pass


class RawTime(FakeStrictBase):
    pass
