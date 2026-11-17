"""
renloc_core/ast_reader/ast_nodes.py
Pure data container representations (inert mock AST nodes) for Ren'Py AST objects.
These classes hold AST fields safely without executing any methods or unpickling untrusted logic.
"""
from typing import Any, List, Tuple, Optional, Dict

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
        elif isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
            for k, v in state[1].items():
                setattr(self, k, v)

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"{self.__class__.__name__}({attrs})"

# --- renpy.ast nodes ---
class Node(FakeStrictBase):
    filename: str = ""
    linenumber: int = 0
    next: Optional['Node'] = None

class Say(Node):
    who: Optional[str] = None
    what: str = ""
    with_: Optional[str] = None
    interact: bool = True
    attributes: Optional[Tuple[str, ...]] = None
    arguments: Optional[Any] = None
    temporary_attributes: Optional[Tuple[str, ...]] = None
    rollback: str = "force"

class Translate(Node):
    identifier: str = ""
    alternate: Optional[str] = None
    language: Optional[str] = None
    block: List[Node] = []

class TranslateSay(Node):
    identifier: str = ""
    language: Optional[str] = None
    who: Optional[str] = None
    what: str = ""
    with_: Optional[str] = None
    interact: bool = True
    attributes: Optional[Tuple[str, ...]] = None
    arguments: Optional[Any] = None
    temporary_attributes: Optional[Tuple[str, ...]] = None

class EndTranslate(Node):
    pass

class Label(Node):
    name: str = ""
    parameters: Optional[Any] = None
    block: List[Node] = []
    hide: bool = False

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

class Screen(Node):
    screen: Any = None

class PyCode(FakeStrictBase):
    source: str = ""
    location: Tuple[str, int] = ("", 0)
    mode: str = "exec"
    bytecode: Optional[bytes] = None

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
class SLScreen(FakeStrictBase):
    name: str = ""
    modal: str = "False"
    zorder: str = "0"
    tag: Optional[str] = None
    children: List[Any] = []

class SLDisplayable(FakeStrictBase):
    displayable_type: str = ""
    positional: List[Any] = []
    keyword: List[Tuple[str, Any]] = []
    children: List[Any] = []

class SLIf(FakeStrictBase):
    entries: List[Tuple[str, List[Any]]] = []

class SLFor(FakeStrictBase):
    variable: str = ""
    expression: str = ""
    children: List[Any] = []

class SLPython(FakeStrictBase):
    code: Any = None

class SLUse(FakeStrictBase):
    target: str = ""
    args: Any = None
