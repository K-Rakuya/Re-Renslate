"""
renloc_core/ast_reader/unpickler.py
Safe, inert Ren'Py AST Unpickler (NativeRenpyUnpickler).
Parses .rpyc and .rpymc archives without executing untrusted bytecode.
Supports Python 2 (PROTOCOL <= 2) and Python 3 (PROTOCOL 3..5) pickles.
"""
import io
import pickle
import zlib
import hashlib
from typing import Any, List, Optional, Tuple, Dict
from . import ast_nodes

RENPY_RPC2_MAGIC = b"RENPY RPC-2"

class SafeUnpicklingError(Exception):
    pass

class NativeRenpyUnpickler(pickle.Unpickler):
    """
    Inert pickle VM whitelist unpickler for Ren'Py AST objects.
    Maps all renpy.ast.*, renpy.sl2.slast.*, and renpy.revertable.* to safe inert representations.
    Rejects any system calls, arbitrary code execution, and unsafe globals.
    """

    SAFE_MODULES = {
        "renpy.ast",
        "renpy.sl2.slast",
        "renpy.atl",
        "renpy.python",
        "renpy.revertable",
        "builtins",
        "__builtin__",
        "collections",
        "copyreg",
    }

    CLASS_MAPPINGS = {
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
        ("renpy.ast", "Screen"): ast_nodes.Screen,
        ("renpy.ast", "PyCode"): ast_nodes.PyCode,

        # renpy.sl2.slast
        ("renpy.sl2.slast", "SLScreen"): ast_nodes.SLScreen,
        ("renpy.sl2.slast", "SLDisplayable"): ast_nodes.SLDisplayable,
        ("renpy.sl2.slast", "SLIf"): ast_nodes.SLIf,
        ("renpy.sl2.slast", "SLFor"): ast_nodes.SLFor,
        ("renpy.sl2.slast", "SLPython"): ast_nodes.SLPython,
        ("renpy.sl2.slast", "SLUse"): ast_nodes.SLUse,

        # revertable
        ("renpy.python", "RevertableList"): ast_nodes.RevertableList,
        ("renpy.python", "RevertableDict"): ast_nodes.RevertableDict,
        ("renpy.python", "RevertableSet"): ast_nodes.RevertableSet,
        ("renpy.python", "RevertableObject"): ast_nodes.RevertableObject,
        ("renpy.revertable", "RevertableList"): ast_nodes.RevertableList,
        ("renpy.revertable", "RevertableDict"): ast_nodes.RevertableDict,
        ("renpy.revertable", "RevertableSet"): ast_nodes.RevertableSet,
        ("renpy.revertable", "RevertableObject"): ast_nodes.RevertableObject,
    }

    def __init__(self, file, *args, **kwargs):
        super().__init__(file, *args, **kwargs)
        # Enable python 2 string compatibility fallback
        self.encoding = "utf-8"
        self.errors = "surrogateescape"

    def find_class(self, module: str, name: str) -> Any:
        # Check explicit whitelist mappings
        if (module, name) in self.CLASS_MAPPINGS:
            return self.CLASS_MAPPINGS[(module, name)]

        # Allow basic safe builtins (int, float, bool, str, list, dict, tuple, set)
        if module in ("builtins", "__builtin__"):
            if name in ("int", "float", "bool", "str", "unicode", "list", "dict", "tuple", "set", "frozenset", "bytes", "bytearray"):
                return getattr(ast_nodes, name, getattr(dict, name, type(None)))

        # If it's a known renpy AST or SL2 class not in mapping, dynamically create inert subclass
        if module.startswith("renpy.ast") or module.startswith("renpy.sl2") or module.startswith("renpy.atl"):
            dynamic_cls = type(name, (ast_nodes.FakeStrictBase,), {"__module__": module})
            return dynamic_cls

        # Strict safety check: strictly reject any dangerous modules
        raise SafeUnpicklingError(f"Security Alert: Attempted to unpickle unauthorized class: {module}.{name}")

def read_rpyc_data(data: bytes) -> Any:
    """
    Parses a raw .rpyc or .rpymc binary payload into its inert AST tree.
    Handles RENPY RPC-2 header, slot tables, and zlib decompression.
    """
    if len(data) < len(RENPY_RPC2_MAGIC):
        raise ValueError("Invalid RPYC file: file too short")

    if not data.startswith(RENPY_RPC2_MAGIC) and not data.startswith(b"\x78\x9c"):
        # Could be raw zlib or legacy RPC-1
        pass

    stream = io.BytesIO(data)
    if data.startswith(RENPY_RPC2_MAGIC):
        stream.seek(len(RENPY_RPC2_MAGIC))
        # Ren'Py RPC-2 format has slot table: 4-byte position and length pairs or slot dict
        # Skip slot header to first data segment or unpickle slot directory
        try:
            slot_unpickler = NativeRenpyUnpickler(stream)
            slot_table = slot_unpickler.load()
        except Exception:
            # Fallback to direct zlib search
            stream.seek(len(RENPY_RPC2_MAGIC))

    # Read remaining stream and decompress zlib blocks
    remaining = stream.read()
    if not remaining:
        raise ValueError("Empty RPYC payload")

    # Locate first zlib header
    zlib_start = remaining.find(b"\x78\x9c")
    if zlib_start == -1:
        zlib_start = remaining.find(b"\x78\x01")
    if zlib_start == -1:
        zlib_start = remaining.find(b"\x78\xda")

    if zlib_start != -1:
        decompressed = zlib.decompress(remaining[zlib_start:])
    else:
        decompressed = remaining

    # Unpickle using NativeRenpyUnpickler
    unpickler = NativeRenpyUnpickler(io.BytesIO(decompressed))
    ast_tree = unpickler.load()
    return ast_tree
