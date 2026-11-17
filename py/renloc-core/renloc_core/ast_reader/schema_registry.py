"""
renloc_core/ast_reader/schema_registry.py
AstSchemaRegistry for loading canonical __slots__ across Ren'Py SDK versions:
  - 7.x (Python 2.7, pickle protocol <= 2)
  - 8.0-8.3 (Python 3.9-3.11, pickle protocol 4/5, TranslateSay introduction)
  - 8.4-8.5.3 (Python 3.12, cslots compressed slots architecture, col_offset)
Supports schema queries, version detection, node validation, and RPYCv3 stub identification.
"""
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Version Constants
VERSION_7X = "7.x"
VERSION_8_0_8_3 = "8.0-8.3"
VERSION_8_4_8_5 = "8.4-8.5.3"
ALL_VERSIONS = (VERSION_7X, VERSION_8_0_8_3, VERSION_8_4_8_5)

# Magic headers
MAGIC_RPC2_STANDARD = b"RENPY RPC2"      # Standard Ren'Py 8.x / 7.x (10 bytes)
MAGIC_RPC2_ALT = b"RENPY RPC-2"          # Alternate / fixture 11 bytes
MAGIC_RPC3_STANDARD = b"RENPY RPC3"      # RPYCv3 stub (10 bytes)
MAGIC_RPC3_ALT = b"RENPY RPC-3"          # RPYCv3 stub variant (11 bytes)
MAGIC_RPC3_DOT = b"RENPY RPC-3.0"        # RPYCv3 stub variant (13 bytes)


@dataclass(frozen=True)
class ClassSchema:
    """Represents canonical AST node schema for a specific Ren'Py version."""
    class_name: str
    module_name: str
    version: str
    slots: Tuple[str, ...]
    has_cslots: bool = False
    linenumbers: bool = False


@dataclass
class RpycHeaderInfo:
    """Parsed RPYC header and slot table metadata."""
    magic: bytes
    version: int
    is_stub: bool = False
    slots: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    requires_sandbox_fallback: bool = False
    stub_type: Optional[str] = None
    error_message: Optional[str] = None


class AstSchemaRegistry:
    """
    Central schema registry for Ren'Py AST structures across SDK 7.x-8.5.3.
    Provides standard __slots__ definitions, schema validation, and stub detection.
    """

    # --- Pre-populated Schemas ---
    # 1. PyCode
    _PYCODE_7X = ("source", "location", "mode", "bytecode")
    _PYCODE_8_0 = ("source", "location", "mode", "bytecode", "py", "hashcode")
    _PYCODE_8_4 = ("source", "location", "mode", "bytecode", "py", "hashcode", "filename", "linenumber", "col_offset")

    # 2. Say
    _SAY_7X = (
        "filename", "linenumber", "name", "next", "who", "what", "with_",
        "interact", "attributes", "arguments", "temporary_attributes", "rollback"
    )
    _SAY_8_0 = (
        "filename", "linenumber", "name", "next", "who", "who_fast", "what", "with_",
        "interact", "attributes", "arguments", "temporary_attributes", "rollback",
        "identifier", "explicit_identifier"
    )
    _SAY_8_4 = (
        "filename", "_name", "name_version", "name_serial", "next", "linenumber", "col_offset",
        "who", "who_fast", "what", "with_", "interact", "attributes", "arguments",
        "temporary_attributes", "rollback", "identifier", "explicit_identifier"
    )

    # 3. TranslateSay
    _TRANSLATESAY_8_0 = _SAY_8_0 + ("language", "alternate")
    _TRANSLATESAY_8_4 = _SAY_8_4 + ("language", "alternate")

    # 4. Screen
    _SCREEN_7X = ("filename", "linenumber", "name", "next", "screen")
    _SCREEN_8_0 = ("filename", "linenumber", "name", "next", "screen")
    _SCREEN_8_4 = ("filename", "_name", "name_version", "name_serial", "next", "linenumber", "col_offset", "screen")

    # 5. sl2.slast - SLScreen
    _SLSCREEN_7X = (
        "location", "serial", "keyword", "children", "name", "modal", "zorder",
        "tag", "variant", "style_prefix", "layer", "sensitive", "roll_forward"
    )
    _SLSCREEN_8_0 = _SLSCREEN_7X + ("const_ast", "not_const_ast", "analysis")
    _SLSCREEN_8_4 = _SLSCREEN_8_0 + ("docstring", "version")

    # 6. sl2.slast - SLDisplayable
    _SLDISPLAYABLE_7X = (
        "location", "serial", "keyword", "children", "displayable", "scope",
        "child_or_fixed", "style", "text_style", "pass_context", "imagemap",
        "replaces", "default_keywords", "hotspot", "variable", "name"
    )
    _SLDISPLAYABLE_8_0 = _SLDISPLAYABLE_7X + ("unique", "local_constant")
    _SLDISPLAYABLE_8_4 = _SLDISPLAYABLE_8_0 + ("constant", "has_keyword", "last_keyword")

    # 7. LayeredImage
    _LAYEREDIMAGE_7X = ("attributes", "at", "name", "image_format", "format_function", "attribute_function")
    _LAYEREDIMAGE_8_0 = _LAYEREDIMAGE_7X + ("offer_screen", "kwargs")
    _LAYEREDIMAGE_8_4 = _LAYEREDIMAGE_8_0 + ("layers",)

    def __init__(self):
        self._schemas: Dict[Tuple[str, str], ClassSchema] = {}
        self._register_default_schemas()

    def _register_default_schemas(self):
        """Populates the canonical schemas for all supported versions."""
        # PyCode
        self._add(VERSION_7X, "PyCode", "renpy.ast", self._PYCODE_7X, has_cslots=False, linenumbers=True)
        self._add(VERSION_8_0_8_3, "PyCode", "renpy.ast", self._PYCODE_8_0, has_cslots=False, linenumbers=True)
        self._add(VERSION_8_4_8_5, "PyCode", "renpy.ast", self._PYCODE_8_4, has_cslots=True, linenumbers=True)

        # Say
        self._add(VERSION_7X, "Say", "renpy.ast", self._SAY_7X, has_cslots=False, linenumbers=True)
        self._add(VERSION_8_0_8_3, "Say", "renpy.ast", self._SAY_8_0, has_cslots=False, linenumbers=True)
        self._add(VERSION_8_4_8_5, "Say", "renpy.ast", self._SAY_8_4, has_cslots=True, linenumbers=True)

        # TranslateSay (Ren'Py 8+)
        self._add(VERSION_8_0_8_3, "TranslateSay", "renpy.ast", self._TRANSLATESAY_8_0, has_cslots=False, linenumbers=True)
        self._add(VERSION_8_4_8_5, "TranslateSay", "renpy.ast", self._TRANSLATESAY_8_4, has_cslots=True, linenumbers=True)

        # Screen
        self._add(VERSION_7X, "Screen", "renpy.ast", self._SCREEN_7X, has_cslots=False, linenumbers=True)
        self._add(VERSION_8_0_8_3, "Screen", "renpy.ast", self._SCREEN_8_0, has_cslots=False, linenumbers=True)
        self._add(VERSION_8_4_8_5, "Screen", "renpy.ast", self._SCREEN_8_4, has_cslots=True, linenumbers=True)

        # sl2.slast.SLScreen
        self._add(VERSION_7X, "SLScreen", "renpy.sl2.slast", self._SLSCREEN_7X, has_cslots=False)
        self._add(VERSION_8_0_8_3, "SLScreen", "renpy.sl2.slast", self._SLSCREEN_8_0, has_cslots=False)
        self._add(VERSION_8_4_8_5, "SLScreen", "renpy.sl2.slast", self._SLSCREEN_8_4, has_cslots=False)

        # sl2.slast.SLDisplayable
        self._add(VERSION_7X, "SLDisplayable", "renpy.sl2.slast", self._SLDISPLAYABLE_7X, has_cslots=False)
        self._add(VERSION_8_0_8_3, "SLDisplayable", "renpy.sl2.slast", self._SLDISPLAYABLE_8_0, has_cslots=False)
        self._add(VERSION_8_4_8_5, "SLDisplayable", "renpy.sl2.slast", self._SLDISPLAYABLE_8_4, has_cslots=False)

        # LayeredImage
        self._add(VERSION_7X, "LayeredImage", "renpy.display.layeredimage", self._LAYEREDIMAGE_7X, has_cslots=False)
        self._add(VERSION_8_0_8_3, "LayeredImage", "renpy.display.layeredimage", self._LAYEREDIMAGE_8_0, has_cslots=False)
        self._add(VERSION_8_4_8_5, "LayeredImage", "renpy.display.layeredimage", self._LAYEREDIMAGE_8_4, has_cslots=False)

    def _add(self, version: str, class_name: str, module_name: str, slots: Tuple[str, ...], has_cslots: bool = False, linenumbers: bool = False):
        schema = ClassSchema(
            class_name=class_name,
            module_name=module_name,
            version=version,
            slots=slots,
            has_cslots=has_cslots,
            linenumbers=linenumbers,
        )
        self._schemas[(version, class_name)] = schema

    def register_schema(
        self,
        version: str,
        class_name: str,
        slots: Tuple[str, ...],
        module_name: str = "renpy.ast",
        has_cslots: bool = False,
        linenumbers: bool = False,
    ):
        """Allows registering or extending custom schemas dynamically."""
        self._add(version, class_name, module_name, slots, has_cslots=has_cslots, linenumbers=linenumbers)

    def get_schema(self, version: str, class_name: str) -> Optional[ClassSchema]:
        """Retrieves schema for a specific version and class name."""
        return self._schemas.get((version, class_name))

    def get_slots(self, version: str, class_name: str) -> Tuple[str, ...]:
        """Returns slot names tuple for a given version and class name."""
        schema = self.get_schema(version, class_name)
        if schema:
            return schema.slots
        return ()

    @staticmethod
    def is_rpycv3_stub(data: bytes) -> bool:
        """
        Detects whether the binary data represents an RPYCv3 stub structure.
        Checks for:
          - 'RENPY RPC3' / 'RENPY RPC-3' / 'RENPY RPC-3.0' headers
          - Explicit stub markers in slot directory
        """
        if not data:
            return False
        if (
            data.startswith(MAGIC_RPC3_STANDARD)
            or data.startswith(MAGIC_RPC3_ALT)
            or data.startswith(MAGIC_RPC3_DOT)
            or data.startswith(b"RPYC3")
        ):
            return True

        # Check for stub markers in slot table
        if len(data) >= 16 and (data.startswith(MAGIC_RPC2_STANDARD) or data.startswith(MAGIC_RPC2_ALT)):
            # If all slot offsets point past end of file or have length 0 with stub flags
            header_len = len(MAGIC_RPC2_STANDARD) if data.startswith(MAGIC_RPC2_STANDARD) else len(MAGIC_RPC2_ALT)
            if len(data) >= header_len + 12:
                slot, start, length = struct.unpack("III", data[header_len:header_len + 12])
                if slot == 3 and length == 0:
                    return True
        return False

    @classmethod
    def parse_rpyc_header(cls, data: bytes) -> RpycHeaderInfo:
        """
        Parses the binary header and slot directory of a .rpyc or .rpymc payload.
        Handles:
          - RPYCv3 stub detection -> flags requires_sandbox_fallback=True
          - Standard RPYC2 (10-byte header + binary 'III' slot records)
          - Alternate RPYC2 (11-byte header + binary or pickled slot table)
          - Legacy RPYC1 (uncompressed or raw zlib stream without header)
        """
        if not data:
            return RpycHeaderInfo(magic=b"", version=0, error_message="Empty data payload")

        # 1. RPYCv3 Stub Detection
        if cls.is_rpycv3_stub(data):
            magic = (
                MAGIC_RPC3_DOT if data.startswith(MAGIC_RPC3_DOT)
                else MAGIC_RPC3_ALT if data.startswith(MAGIC_RPC3_ALT)
                else MAGIC_RPC3_STANDARD if data.startswith(MAGIC_RPC3_STANDARD)
                else data[:10]
            )
            return RpycHeaderInfo(
                magic=magic,
                version=3,
                is_stub=True,
                requires_sandbox_fallback=True,
                stub_type="RPYCv3_STUB",
                error_message="RPYCv3 stub detected: requires un.rpyc sandbox fallback extraction",
            )

        # 2. Standard RPYC2 (b"RENPY RPC2", 10 bytes)
        if data.startswith(MAGIC_RPC2_STANDARD):
            slots = {}
            pos = len(MAGIC_RPC2_STANDARD)
            while pos + 12 <= len(data):
                slot_id, start, length = struct.unpack("III", data[pos:pos + 12])
                if slot_id == 0:
                    break
                slots[slot_id] = (start, length)
                pos += 12
            return RpycHeaderInfo(magic=MAGIC_RPC2_STANDARD, version=2, slots=slots)

        # 3. Alternate RPYC-2 (b"RENPY RPC-2", 11 bytes)
        if data.startswith(MAGIC_RPC2_ALT):
            slots = {}
            pos = len(MAGIC_RPC2_ALT)
            # Try unpacking binary 'III' records first
            if pos + 12 <= len(data):
                try:
                    test_slot, test_start, test_len = struct.unpack("III", data[pos:pos + 12])
                    if test_slot in (1, 2) and test_start < len(data):
                        while pos + 12 <= len(data):
                            s_id, s_start, s_len = struct.unpack("III", data[pos:pos + 12])
                            if s_id == 0:
                                break
                            slots[s_id] = (s_start, s_len)
                            pos += 12
                        return RpycHeaderInfo(magic=MAGIC_RPC2_ALT, version=2, slots=slots)
                except Exception:
                    pass

            # Fallback: scan for zlib stream location in payload
            for zmagic in (b"\x78\x9c", b"\x78\x01", b"\x78\xda"):
                zpos = data.find(zmagic, pos)
                if zpos != -1:
                    slots[1] = (zpos, len(data) - zpos)
                    break
            return RpycHeaderInfo(magic=MAGIC_RPC2_ALT, version=2, slots=slots)

        # 4. Legacy RPYC1 / Raw Zlib stream
        if data.startswith(b"\x78\x9c") or data.startswith(b"\x78\x01") or data.startswith(b"\x78\xda"):
            return RpycHeaderInfo(magic=b"", version=1, slots={1: (0, len(data))})

        # 5. Raw pickle protocol stream (b"\x80\x02", b"\x80\x04", etc.)
        if data.startswith(b"\x80"):
            return RpycHeaderInfo(magic=b"", version=1, slots={1: (0, len(data))})

        return RpycHeaderInfo(magic=data[:10], version=0, error_message="Unrecognized RPYC header structure")

    def detect_version(self, data_or_ast: Union[bytes, Any]) -> str:
        """
        Infers the target Ren'Py SDK version tier:
          - If bytes: examines pickle protocol (PROTO 2 vs 4/5), cslots markers, or RPC version.
          - If AST node/list: examines presence of col_offset (8.4+), who_fast / identifier (8.0+),
            or legacy attributes (7.x).
        """
        if isinstance(data_or_ast, (bytes, bytearray)):
            data = bytes(data_or_ast)
            # Check protocol byte in raw or decompressed stream
            header_info = self.parse_rpyc_header(data)
            if header_info.version == 3:
                return VERSION_8_4_8_5

            stream_data = data
            if header_info.slots and 1 in header_info.slots:
                start, length = header_info.slots[1]
                slot_slice = data[start:start + length]
                try:
                    stream_data = zlib.decompress(slot_slice)
                except Exception:
                    stream_data = slot_slice

            if b"col_offset" in stream_data or b"_cslot" in stream_data:
                return VERSION_8_4_8_5
            if b"\x80\x04" in stream_data or b"\x80\x05" in stream_data or b"who_fast" in stream_data:
                return VERSION_8_0_8_3
            if b"\x80\x02" in stream_data:
                return VERSION_7X
            return VERSION_8_0_8_3

        # Check AST nodes
        node = data_or_ast
        if isinstance(node, list) and node:
            node = node[0]

        if hasattr(node, "col_offset") and getattr(node, "col_offset", 0) != 0:
            return VERSION_8_4_8_5
        if hasattr(node, "who_fast") or hasattr(node, "explicit_identifier"):
            return VERSION_8_0_8_3
        return VERSION_7X

    def validate_node_schema(self, node: Any, version: Optional[str] = None) -> bool:
        """
        Validates that an AST node contains expected attributes/slots for the given version.
        """
        class_name = type(node).__name__
        target_version = version or self.detect_version(node)
        schema = self.get_schema(target_version, class_name)
        if not schema:
            return True  # Unknown classes pass gracefully

        # Check core required slots
        for slot in schema.slots:
            if not hasattr(node, slot) and slot not in node.__dict__:
                # Allow optional slots (e.g. arguments, temporary_attributes)
                if slot in ("attributes", "arguments", "temporary_attributes", "with_", "identifier", "screen"):
                    continue
                return False
        return True
