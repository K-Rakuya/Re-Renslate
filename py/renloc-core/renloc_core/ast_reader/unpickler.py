"""
renloc_core/ast_reader/unpickler.py
Safe, inert Ren'Py AST Unpickler (NativeRenpyUnpickler).
Parses .rpyc and .rpymc archives without executing untrusted bytecode.
Supports Python 2 (PROTOCOL <= 2) and Python 3 (PROTOCOL 3..5) pickles.
Fully handles binary slot directories (RPYC2), alternate slot tables, and raw zlib streams.
Detects and rejects RPYCv3 stub structures requiring sandbox fallback.
"""
import io
import pickle
import zlib
import struct
from typing import Any, Dict, List, Optional, Tuple, Set

from . import ast_nodes
from .schema_registry import AstSchemaRegistry, RpycHeaderInfo, MAGIC_RPC2_STANDARD, MAGIC_RPC2_ALT


class SafeUnpicklingError(Exception):
    """Raised when an unpickling security policy is violated."""
    pass


class Rpycv3StubError(SafeUnpicklingError):
    """Raised when an RPYCv3 stub structure is encountered, requiring sandbox un.rpyc extraction."""
    pass


def _safe_reconstructor(cls, base, state):
    """Safe wrapper for copyreg._reconstructor."""
    if not isinstance(cls, type) or not issubclass(cls, (ast_nodes.FakeStrictBase, list, dict, set, tuple)):
        raise SafeUnpicklingError(f"Security Alert: Unauthorized reconstructor target: {cls}")
    obj = cls.__new__(cls, *state)
    return obj


def _safe_newobj(cls, *args):
    """Safe wrapper for copyreg.__newobj__."""
    if not isinstance(cls, type) or not issubclass(cls, (ast_nodes.FakeStrictBase, list, dict, set, tuple)):
        raise SafeUnpicklingError(f"Security Alert: Unauthorized __newobj__ target: {cls}")
    return cls.__new__(cls, *args)


class NativeRenpyUnpickler(pickle.Unpickler):
    """
    Inert pickle VM whitelist unpickler for Ren'Py AST objects.
    Maps all renpy.ast.*, renpy.sl2.slast.*, renpy.atl.*, renpy.parameter.*,
    and renpy.revertable.* to safe inert representations.
    Strictly rejects any system calls, arbitrary code execution, and unsafe globals.
    """

    SAFE_MODULES: Set[str] = {
        "renpy.ast",
        "renpy.sl2.slast",
        "renpy.atl",
        "renpy.parameter",
        "renpy.object",
        "renpy.python",
        "renpy.revertable",
        "renpy.display.layeredimage",
        "renpy.common.00layeredimage_ren",
        "builtins",
        "__builtin__",
        "collections",
        "copyreg",
        "copy_reg",
        "renloc_core.ast_reader.ast_nodes",
    }

    CLASS_MAPPINGS: Dict[Tuple[str, str], Any] = {
        # renpy.ast
        ("renpy.ast", "Node"): ast_nodes.Node,
        ("renpy.ast", "Say"): ast_nodes.Say,
        ("renpy.ast", "Translate"): ast_nodes.Translate,
        ("renpy.ast", "TranslateSay"): ast_nodes.TranslateSay,
        ("renpy.ast", "EndTranslate"): ast_nodes.EndTranslate,
        ("renpy.ast", "Label"): ast_nodes.Label,
        ("renpy.ast", "Jump"): ast_nodes.Jump,
        ("renpy.ast", "Call"): ast_nodes.Call,
        ("renpy.ast", "Return"): ast_nodes.Return,
        ("renpy.ast", "Scene"): ast_nodes.Scene,
        ("renpy.ast", "Show"): ast_nodes.Show,
        ("renpy.ast", "Hide"): ast_nodes.Hide,
        ("renpy.ast", "With"): ast_nodes.With,
        ("renpy.ast", "Python"): ast_nodes.Python,
        ("renpy.ast", "EarlyPython"): ast_nodes.EarlyPython,
        ("renpy.ast", "Init"): ast_nodes.Init,
        ("renpy.ast", "Default"): ast_nodes.Default,
        ("renpy.ast", "Define"): ast_nodes.Define,
        ("renpy.ast", "UserStatement"): ast_nodes.UserStatement,
        ("renpy.ast", "RPY"): ast_nodes.RPY,
        ("renpy.ast", "Image"): ast_nodes.Image,
        ("renpy.ast", "Screen"): ast_nodes.Screen,
        ("renpy.ast", "PyCode"): ast_nodes.PyCode,

        # renpy.object
        ("renpy.object", "Object"): ast_nodes.FakeStrictBase,

        # renpy.parameter
        ("renpy.parameter", "ArgumentInfo"): ast_nodes.ArgumentInfo,
        ("renpy.parameter", "ParameterInfo"): ast_nodes.ParameterInfo,
        ("renpy.parameter", "Signature"): ast_nodes.Signature,

        # renpy.sl2.slast
        ("renpy.sl2.slast", "SLNode"): ast_nodes.SLNode,
        ("renpy.sl2.slast", "SLBlock"): ast_nodes.SLBlock,
        ("renpy.sl2.slast", "SLScreen"): ast_nodes.SLScreen,
        ("renpy.sl2.slast", "SLDisplayable"): ast_nodes.SLDisplayable,
        ("renpy.sl2.slast", "SLIf"): ast_nodes.SLIf,
        ("renpy.sl2.slast", "SLShowIf"): ast_nodes.SLShowIf,
        ("renpy.sl2.slast", "SLFor"): ast_nodes.SLFor,
        ("renpy.sl2.slast", "SLPython"): ast_nodes.SLPython,
        ("renpy.sl2.slast", "SLUse"): ast_nodes.SLUse,
        ("renpy.sl2.slast", "SLCustomUse"): ast_nodes.SLCustomUse,
        ("renpy.sl2.slast", "SLTransclude"): ast_nodes.SLTransclude,
        ("renpy.sl2.slast", "SLDefault"): ast_nodes.SLDefault,
        ("renpy.sl2.slast", "SLPass"): ast_nodes.SLPass,
        ("renpy.sl2.slast", "SLBreak"): ast_nodes.SLBreak,
        ("renpy.sl2.slast", "SLContinue"): ast_nodes.SLContinue,

        # renpy.atl
        ("renpy.atl", "RawBlock"): ast_nodes.RawBlock,
        ("renpy.atl", "RawMultiloop"): ast_nodes.RawMultiloop,
        ("renpy.atl", "RawChild"): ast_nodes.RawChild,
        ("renpy.atl", "RawChoice"): ast_nodes.RawChoice,
        ("renpy.atl", "RawContains"): ast_nodes.RawContains,
        ("renpy.atl", "RawEvent"): ast_nodes.RawEvent,
        ("renpy.atl", "RawFunction"): ast_nodes.RawFunction,
        ("renpy.atl", "RawIf"): ast_nodes.RawIf,
        ("renpy.atl", "RawOn"): ast_nodes.RawOn,
        ("renpy.atl", "RawParallel"): ast_nodes.RawParallel,
        ("renpy.atl", "RawRepeat"): ast_nodes.RawRepeat,
        ("renpy.atl", "RawTime"): ast_nodes.RawTime,

        # LayeredImage
        ("renpy.display.layeredimage", "LayeredImage"): ast_nodes.LayeredImage,
        ("renpy.display.layeredimage", "RawLayeredImage"): ast_nodes.RawLayeredImage,
        ("renpy.display.layeredimage", "RawAttribute"): ast_nodes.RawAttribute,
        ("renpy.display.layeredimage", "RawAttributeGroup"): ast_nodes.RawAttributeGroup,
        ("renpy.display.layeredimage", "RawCondition"): ast_nodes.RawCondition,
        ("renpy.display.layeredimage", "RawConditionGroup"): ast_nodes.RawConditionGroup,
        ("renpy.display.layeredimage", "RawAlways"): ast_nodes.RawAlways,
        ("renpy.common.00layeredimage_ren", "LayeredImage"): ast_nodes.LayeredImage,
        ("renpy.common.00layeredimage_ren", "RawLayeredImage"): ast_nodes.RawLayeredImage,

        # revertable
        ("renpy.python", "RevertableList"): ast_nodes.RevertableList,
        ("renpy.python", "RevertableDict"): ast_nodes.RevertableDict,
        ("renpy.python", "RevertableSet"): ast_nodes.RevertableSet,
        ("renpy.python", "RevertableObject"): ast_nodes.RevertableObject,
        ("renpy.revertable", "RevertableList"): ast_nodes.RevertableList,
        ("renpy.revertable", "RevertableDict"): ast_nodes.RevertableDict,
        ("renpy.revertable", "RevertableSet"): ast_nodes.RevertableSet,
        ("renpy.revertable", "RevertableObject"): ast_nodes.RevertableObject,

        # copyreg / copy_reg safe constructors
        ("copyreg", "_reconstructor"): _safe_reconstructor,
        ("copy_reg", "_reconstructor"): _safe_reconstructor,
        ("copyreg", "__newobj__"): _safe_newobj,
        ("copy_reg", "__newobj__"): _safe_newobj,
        ("copyreg", "__newobj_ex__"): _safe_newobj,
        ("copy_reg", "__newobj_ex__"): _safe_newobj,
        ("__builtin__", "__newobj__"): _safe_newobj,
        ("builtins", "__newobj__"): _safe_newobj,
    }

    SAFE_BUILTINS: Dict[str, Any] = {
        "int": int,
        "float": float,
        "bool": bool,
        "str": str,
        "bytes": bytes,
        "bytearray": bytearray,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "frozenset": frozenset,
        "slice": slice,
        "complex": complex,
        "range": range,
        # Python 2 aliases
        "unicode": str,
        "long": int,
        "NoneType": type(None),
    }

    def __init__(self, file, *args, **kwargs):
        super().__init__(file, *args, **kwargs)
        self.encoding = "utf-8"
        self.errors = "surrogateescape"

    def find_class(self, module: str, name: str) -> Any:
        # 1. Check explicit whitelist mappings
        if (module, name) in self.CLASS_MAPPINGS:
            return self.CLASS_MAPPINGS[(module, name)]

        # 2. Allow safe builtins
        if module in ("builtins", "__builtin__"):
            if name in self.SAFE_BUILTINS:
                return self.SAFE_BUILTINS[name]

        # 3. Safe collections
        if module == "collections":
            if name in ("OrderedDict", "defaultdict", "deque"):
                import collections
                return getattr(collections, name)

        # 4. Safe internal inert AST nodes
        if module == "renloc_core.ast_reader.ast_nodes" and hasattr(ast_nodes, name):
            return getattr(ast_nodes, name)

        # 4. If it's a known renpy namespace, dynamically create inert subclass
        if (
            module.startswith("renpy.ast")
            or module.startswith("renpy.sl2")
            or module.startswith("renpy.atl")
            or module.startswith("renpy.revertable")
            or module.startswith("renpy.python")
            or module.startswith("renpy.parameter")
            or module.startswith("renpy.display")
            or module.startswith("renpy.character")
        ):
            dynamic_cls = type(name, (ast_nodes.FakeStrictBase,), {"__module__": module})
            return dynamic_cls

        # Strict safety check: strictly reject any unauthorized modules
        raise SafeUnpicklingError(f"Security Alert: Attempted to unpickle unauthorized class: {module}.{name}")


def read_rpyc_data(data: bytes, slot: int = 1) -> Any:
    """
    Parses a raw .rpyc or .rpymc binary payload into its inert AST tree.
    Fully handles RENPY RPC2 / RPC-2 headers, binary slot tables, and zlib decompression.

    Parameters:
      data: Binary contents of the .rpyc file.
      slot: Slot index to load (default 1 for pre-transform AST, 2 for post-transform AST).

    Returns:
      Deserialized inert AST root (typically a list of AST nodes).

    Raises:
      Rpycv3StubError: If an RPYCv3 stub structure is encountered.
      SafeUnpicklingError: If untrusted/dangerous opcodes are present.
      ValueError: If payload is malformed or empty.
    """
    if not data:
        raise ValueError("Empty RPYC payload")

    # Check for RPYCv3 stub first
    if AstSchemaRegistry.is_rpycv3_stub(data):
        raise Rpycv3StubError("RPYCv3 stub detected: file requires sandbox un.rpyc fallback extraction")

    header_info = AstSchemaRegistry.parse_rpyc_header(data)

    compressed_bytes: Optional[bytes] = None

    # 1. Slot table available from parsed header
    if header_info.slots:
        target_slot = slot
        if target_slot not in header_info.slots:
            # Fallback to alternate slot if slot 1 was requested but only slot 2 exists
            if target_slot == 1 and 2 in header_info.slots:
                target_slot = 2
            elif target_slot == 2 and 1 in header_info.slots:
                target_slot = 1
            else:
                # Use first available slot
                target_slot = next(iter(header_info.slots.keys()))

        start, length = header_info.slots[target_slot]
        if start < len(data):
            compressed_bytes = data[start:start + length]

    # 2. If compressed bytes not resolved, attempt direct zlib marker scan
    if not compressed_bytes or not (compressed_bytes.startswith(b"\x78\x9c") or compressed_bytes.startswith(b"\x78\x01") or compressed_bytes.startswith(b"\x78\xda")):
        for zmagic in (b"\x78\x9c", b"\x78\x01", b"\x78\xda"):
            zpos = data.find(zmagic)
            if zpos != -1:
                compressed_bytes = data[zpos:]
                break

    if not compressed_bytes:
        # Check if already raw uncompressed pickle
        if data.startswith(b"\x80"):
            decompressed = data
        else:
            raise ValueError("Unable to locate valid compressed AST slot or zlib stream in RPYC file")
    else:
        try:
            decompressed = zlib.decompress(compressed_bytes)
        except Exception as e:
            # If length was specified, try decompressing full remaining
            try:
                zpos = data.find(b"\x78")
                if zpos != -1:
                    decompressed = zlib.decompress(data[zpos:])
                else:
                    raise e
            except Exception as final_e:
                raise ValueError(f"Failed to decompress RPYC zlib payload: {final_e}") from final_e

    # Safe unpickling
    unpickler = NativeRenpyUnpickler(io.BytesIO(decompressed))
    ast_tree = unpickler.load()
    return ast_tree


def read_rpymc_data(data: bytes, slot: int = 1) -> Any:
    """
    Parses a compiled Ren'Py macro/screen archive (.rpymc) safely into its inert AST tree.
    .rpymc files share the exact same binary slot archive structure as .rpyc.
    """
    return read_rpyc_data(data, slot=slot)


def read_rpyc_slots(data: bytes) -> Dict[int, Any]:
    """
    Reads and deserializes all slots available in the RPYC file into a dictionary:
      {slot_number: unpickled_ast}
    """
    if not data:
        raise ValueError("Empty RPYC payload")

    if AstSchemaRegistry.is_rpycv3_stub(data):
        raise Rpycv3StubError("RPYCv3 stub detected: file requires sandbox un.rpyc fallback extraction")

    header_info = AstSchemaRegistry.parse_rpyc_header(data)
    result: Dict[int, Any] = {}

    for slot_id, (start, length) in header_info.slots.items():
        if start >= len(data):
            continue
        chunk = data[start:start + length]
        try:
            decompressed = zlib.decompress(chunk)
            unpickler = NativeRenpyUnpickler(io.BytesIO(decompressed))
            result[slot_id] = unpickler.load()
        except Exception:
            continue

    if not result:
        # Fallback to standard read_rpyc_data
        result[1] = read_rpyc_data(data, slot=1)

    return result
