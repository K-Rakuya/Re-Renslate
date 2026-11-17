"""
tests/test_ast_reader.py
Comprehensive unit test suite for renloc_core.ast_reader:
  1. code_generator: Say.get_code(), TranslateSay.get_code(), string escaping, translation MD5 hashing.
  2. schema_registry: AstSchemaRegistry slots across Ren'Py 7.x, 8.0-8.3, 8.4-8.5.3, and RPYCv3 stub detection.
  3. unpickler: NativeRenpyUnpickler inert VM, safe slot tables, multi-slot parsing, rpymc support, and security rejection.
"""
import io
import struct
import zlib
import pickle
import hashlib
from pathlib import Path
import pytest

from renloc_core.ast_reader import (
    FakeStrictBase,
    Node,
    Say,
    Translate,
    TranslateSay,
    Label,
    UserStatement,
    RPY,
    PyCode,
    ArgumentInfo,
    SLScreen,
    SLDisplayable,
    LayeredImage,
    RawLayeredImage,
    RevertableList,
    RevertableDict,
    encode_say_string,
    get_say_code,
    get_node_code,
    compute_translation_hash,
    generate_translation_id,
    format_translate_block,
    AstSchemaRegistry,
    RpycHeaderInfo,
    VERSION_7X,
    VERSION_8_0_8_3,
    VERSION_8_4_8_5,
    NativeRenpyUnpickler,
    SafeUnpicklingError,
    Rpycv3StubError,
    read_rpyc_data,
    read_rpymc_data,
    read_rpyc_slots,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


# =========================================================================
# 1. Code Generator Tests
# =========================================================================

class TestCodeGenerator:
    def test_encode_say_string_basic(self):
        assert encode_say_string("Hello world") == '"Hello world"'

    def test_encode_say_string_whitespace_and_escaping(self):
        # Consecutive spaces must be escaped with "\ "
        assert encode_say_string("Hello  world") == '"Hello \\ world"'
        assert encode_say_string("One   Two") == '"One \\ \\ Two"'
        # Quotes and newlines
        assert encode_say_string('She said: "Hello!"') == r'"She said: \"Hello!\""'
        assert encode_say_string("Line 1\nLine 2") == r'"Line 1\nLine 2"'
        assert encode_say_string("Back\\slash") == r'"Back\\slash"'

    def test_encode_say_string_vietnamese_unicode(self):
        vn_text = "Xin chào thế giới! Đêm nay trăng sáng dịu êm."
        encoded = encode_say_string(vn_text)
        assert encoded == f'"{vn_text}"'

    def test_say_get_code_all_attributes(self):
        say = Say(
            who="sylvia",
            what="Greetings, traveler!",
            attributes=("happy", "smile"),
            temporary_attributes=("wink",),
            with_="dissolve",
            interact=True,
        )
        expected = 'sylvia happy smile @ wink "Greetings, traveler!" with dissolve'
        assert say.get_code() == expected
        assert get_node_code(say) == expected

    def test_say_get_code_nointeract_and_id(self):
        say = Say(
            who=None,
            what="Deep in thoughts...",
            interact=False,
            identifier="deep_01",
            explicit_identifier=True,
        )
        expected = '"Deep in thoughts..." nointeract id deep_01'
        assert say.get_code() == expected

    def test_say_get_code_dialogue_filter(self):
        say = Say(who="narrator", what="secret password")
        filtered = say.get_code(dialogue_filter=lambda t: t.upper())
        assert filtered == 'narrator "SECRET PASSWORD"'

    def test_say_get_code_arguments(self):
        args = ArgumentInfo(
            arguments=[(None, '"pos1"'), ("color", '"#ff0000"')],
            starred_indexes={0},
        )
        say = Say(who="eileen", what="Stylized line", arguments=args)
        assert say.get_code() == 'eileen "Stylized line" (*"pos1", color="#ff0000")'

    def test_translatesay_compatibility(self):
        ts = TranslateSay(
            identifier="start_12345678",
            language="vietnamese",
            who="e",
            what="Welcome to the novel!",
            attributes=("cheer",),
            with_="fade",
        )
        expected = 'e cheer "Welcome to the novel!" with fade'
        assert ts.get_code() == expected
        assert ts.identifier == "start_12345678"
        assert ts.language == "vietnamese"

    def test_user_statement_and_rpy_get_code(self):
        us = UserStatement(line="play music audio_bgm loop")
        assert us.get_code() == "play music audio_bgm loop"

        rpy = RPY(rest=["python", "early"])
        assert rpy.get_code() == "rpy python early"

    def test_translation_identifier_hash_exact(self):
        code = 's "Welcome to the old world!"'
        # Ren'Py md5 calculation: md5.update((code + "\r\n").encode("utf-8"))[:8]
        expected_md5 = hashlib.md5((code + "\r\n").encode("utf-8")).hexdigest()[:8]
        expected_id = "start_" + expected_md5

        generated_id = generate_translation_id(code, label="start")
        assert generated_id == expected_id

    def test_translation_identifier_dotted_label(self):
        code = '"The rain continues."'
        tid = generate_translation_id(code, label="act1.scene2.start")
        assert tid.startswith("act1_scene2_start_")

    def test_translation_identifier_disambiguation(self):
        code = '"Repeated line."'
        existing = set()
        id1 = generate_translation_id(code, label="loop", existing_ids=existing)
        existing.add(id1)
        id2 = generate_translation_id(code, label="loop", existing_ids=existing)
        assert id1 != id2
        assert id2 == id1 + "_1"

    def test_format_translate_block(self):
        say = Say(who="s", what="Xin chào!")
        block = format_translate_block("start_d9ca682b", [say], language="vietnamese")
        assert "translate vietnamese start_d9ca682b:" in block
        assert '    s "Xin chào!"' in block


# =========================================================================
# 2. Schema Registry Tests
# =========================================================================

class TestSchemaRegistry:
    def setup_method(self):
        self.registry = AstSchemaRegistry()

    def test_pycode_slots_progression(self):
        slots_7x = self.registry.get_slots(VERSION_7X, "PyCode")
        slots_8_0 = self.registry.get_slots(VERSION_8_0_8_3, "PyCode")
        slots_8_4 = self.registry.get_slots(VERSION_8_4_8_5, "PyCode")

        # 7.x base slots
        assert "source" in slots_7x and "bytecode" in slots_7x
        assert "py" not in slots_7x
        assert "col_offset" not in slots_7x

        # 8.0-8.3 slots
        assert "py" in slots_8_0 and "hashcode" in slots_8_0
        assert "col_offset" not in slots_8_0

        # 8.4-8.5.3 cslots
        assert "col_offset" in slots_8_4
        assert "filename" in slots_8_4
        schema_8_4 = self.registry.get_schema(VERSION_8_4_8_5, "PyCode")
        assert schema_8_4.has_cslots is True

    def test_say_and_translatesay_slots(self):
        say_slots_7x = self.registry.get_slots(VERSION_7X, "Say")
        say_slots_8_4 = self.registry.get_slots(VERSION_8_4_8_5, "Say")
        ts_slots_8_4 = self.registry.get_slots(VERSION_8_4_8_5, "TranslateSay")

        assert "who_fast" not in say_slots_7x
        assert "who_fast" in say_slots_8_4
        assert "col_offset" in say_slots_8_4
        assert "language" in ts_slots_8_4
        assert "alternate" in ts_slots_8_4

    def test_screen_and_slast_slots(self):
        sls_7x = self.registry.get_slots(VERSION_7X, "SLScreen")
        sls_8_4 = self.registry.get_slots(VERSION_8_4_8_5, "SLScreen")

        assert "docstring" not in sls_7x
        assert "docstring" in sls_8_4
        assert "const_ast" in sls_8_4

        sld_8_4 = self.registry.get_slots(VERSION_8_4_8_5, "SLDisplayable")
        assert "constant" in sld_8_4
        assert "has_keyword" in sld_8_4

    def test_layeredimage_slots(self):
        li_7x = self.registry.get_slots(VERSION_7X, "LayeredImage")
        li_8_4 = self.registry.get_slots(VERSION_8_4_8_5, "LayeredImage")
        assert "offer_screen" not in li_7x
        assert "offer_screen" in li_8_4
        assert "layers" in li_8_4

    def test_rpycv3_stub_detection(self):
        assert AstSchemaRegistry.is_rpycv3_stub(b"RENPY RPC3\x00\x00\x00") is True
        assert AstSchemaRegistry.is_rpycv3_stub(b"RENPY RPC-3\x00\x00") is True
        assert AstSchemaRegistry.is_rpycv3_stub(b"RENPY RPC-3.0\x00") is True
        assert AstSchemaRegistry.is_rpycv3_stub(b"RPYC3_STUB") is True
        # Standard RPC2 is not a stub
        assert AstSchemaRegistry.is_rpycv3_stub(b"RENPY RPC2\x00\x00\x00\x00") is False

    def test_parse_rpyc_header_stub_fallback(self):
        stub_data = b"RENPY RPC3" + b"\x00" * 32
        info = AstSchemaRegistry.parse_rpyc_header(stub_data)
        assert info.version == 3
        assert info.is_stub is True
        assert info.requires_sandbox_fallback is True

    def test_parse_rpyc_header_rpc2_binary(self):
        header = b"RENPY RPC2"
        # 2 slots: slot 1 (start 100, len 50), slot 2 (start 150, len 60), slot 0 terminates
        slot_records = (
            struct.pack("III", 1, 100, 50) +
            struct.pack("III", 2, 150, 60) +
            struct.pack("III", 0, 0, 0)
        )
        data = header + slot_records + b"\x00" * 200
        info = AstSchemaRegistry.parse_rpyc_header(data)
        assert info.version == 2
        assert info.is_stub is False
        assert info.slots[1] == (100, 50)
        assert info.slots[2] == (150, 60)


# =========================================================================
# 3. Unpickler & Security Sandbox Tests
# =========================================================================

class TestNativeRenpyUnpickler:
    def test_fixture_sample_7x_unpickling(self):
        path = FIXTURES_DIR / "rpyc" / "sample_7x.rpyc"
        nodes = read_rpyc_data(path.read_bytes(), slot=1)
        assert isinstance(nodes, list)
        assert len(nodes) >= 2
        assert isinstance(nodes[0], Label) and nodes[0].name == "start"
        assert isinstance(nodes[1], Say) and nodes[1].who == "Sylvia"

    def test_fixture_sample_8x_unpickling(self):
        path = FIXTURES_DIR / "rpyc" / "sample_8x.rpyc"
        nodes = read_rpyc_data(path.read_bytes(), slot=1)
        assert isinstance(nodes, list)
        assert len(nodes) >= 2
        assert isinstance(nodes[0], Label) and nodes[0].name == "start"
        assert isinstance(nodes[1], Say) and nodes[1].who == "Sylvia"

    def test_malicious_rpyc_rejection(self):
        path = FIXTURES_DIR / "rpyc" / "malicious.rpyc"
        with pytest.raises(SafeUnpicklingError):
            read_rpyc_data(path.read_bytes())

    def test_rpycv3_stub_raises_error(self):
        stub = b"RENPY RPC-3" + b"\x00" * 32
        with pytest.raises(Rpycv3StubError) as exc_info:
            read_rpyc_data(stub)
        assert "sandbox" in str(exc_info.value).lower()

    def test_read_rpymc_data(self):
        # Create valid compiled rpymc binary (screen AST payload)
        screens = [SLScreen(name="hud")]
        payload = pickle.dumps(screens, protocol=4)
        compressed = zlib.compress(payload)

        # Build RPC2 format
        header = b"RENPY RPC2"
        slot1_rec = struct.pack("III", 1, len(header) + 24, len(compressed))
        term_rec = struct.pack("III", 0, 0, 0)
        rpymc_bytes = header + slot1_rec + term_rec + compressed

        result = read_rpymc_data(rpymc_bytes, slot=1)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], SLScreen)
        assert result[0].name == "hud"

    def test_read_rpyc_slots_multi(self):
        ast1 = [Label(name="slot1_label")]
        ast2 = [Label(name="slot2_label")]

        c1 = zlib.compress(pickle.dumps(ast1, protocol=4))
        c2 = zlib.compress(pickle.dumps(ast2, protocol=4))

        header = b"RENPY RPC2"
        pos1 = len(header) + 36
        pos2 = pos1 + len(c1)

        slots_dir = (
            struct.pack("III", 1, pos1, len(c1)) +
            struct.pack("III", 2, pos2, len(c2)) +
            struct.pack("III", 0, 0, 0)
        )
        binary = header + slots_dir + c1 + c2

        all_slots = read_rpyc_slots(binary)
        assert 1 in all_slots and 2 in all_slots
        assert all_slots[1][0].name == "slot1_label"
        assert all_slots[2][0].name == "slot2_label"

    def test_pycode_state_tuples_all_versions(self):
        # 4-tuple (7.x)
        p4 = PyCode()
        p4.__setstate__((1, "x = 10", ("script.rpy", 5), "exec"))
        assert p4.source == "x = 10"
        assert p4.filename == "script.rpy"
        assert p4.linenumber == 5

        # 5-tuple (8.0)
        p5 = PyCode()
        p5.__setstate__((1, "y = 20", ("script.rpy", 10), "exec", 3))
        assert p5.py == 3

        # 6-tuple (8.1-8.3)
        p6 = PyCode()
        p6.__setstate__((1, "z = 30", ("script.rpy", 15), "exec", 3, 99999))
        assert p6.hashcode == 99999

        # 7-tuple (8.4-8.5.3 with col_offset)
        p7 = PyCode()
        p7.__setstate__((1, "w = 40", ("script.rpy", 20), "exec", 3, 99999, 4))
        assert p7.col_offset == 4

        # cslots (None, dict)
        p_cslot = PyCode()
        p_cslot.__setstate__((None, {"source": "c = 50", "col_offset": 8}))
        assert p_cslot.source == "c = 50"
        assert p_cslot.col_offset == 8

    def test_revertable_containers(self):
        rl = RevertableList([1, 2, 3])
        rd = RevertableDict({"a": 10})
        data = pickle.dumps((rl, rd), protocol=4)
        unpickler = NativeRenpyUnpickler(io.BytesIO(data))
        res_list, res_dict = unpickler.load()
        assert isinstance(res_list, RevertableList) and res_list == [1, 2, 3]
        assert isinstance(res_dict, RevertableDict) and res_dict["a"] == 10
