#!/usr/bin/env python3
"""
scripts/generate_fixtures.py
Deterministic fixture generator for Ren'Py SDK 7.x - 8.5.3 test fixtures:
  - tests/fixtures/scripts/ (sample .rpy scripts for Ren'Py 7.x & 8.x)
  - tests/fixtures/rpyc/ (compiled .rpyc binaries, protocols 2 & 4, and malicious payload)
  - tests/fixtures/rpa/ (RPA-1.0 paired, RPA-2.0, RPA-3.0 with XOR key, and custom 8-byte header ARC-3.0)
"""

import os
import sys
import zlib
import pickle
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

def create_scripts():
    scripts_dir = FIXTURES_DIR / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # 1. sample_7x.rpy
    sample_7x = """# Ren'Py 7.x Sample Script (Legacy Python 2 compatible syntax)
init -1 python:
    score = 0
    persistent.game_beaten = False

define s = Character('Sylvia', color="#c8ffc8")
define m = Character('Narrator')

default player_name = "Traveler"

label start:
    $ score = 0
    s "Welcome to the old world!"
    extend " Are you prepared for the journey?"

    "The wind blows through the ancient ruins."

    menu:
        "Proceed forward":
            jump forward_path
        "Turn back":
            jump retreat_path

label forward_path:
    $ renpy.say(s, "Courage is rewarded.")
    s "Let us move forward, [player_name]!"
    return

label retreat_path:
    s "Perhaps another day."
    return
"""
    (scripts_dir / "sample_7x.rpy").write_text(sample_7x, encoding="utf-8")

    # 2. sample_8x.rpy
    sample_8x = """# Ren'Py 8.x Modern Script (Python 3.12, SDK 8.4 - 8.5.3)
python early:
    EARLY_COMPILER_FLAG = True

init python:
    from renpy.revertable import RevertableDict, RevertableList

define e = Character(_("Eileen"), color="#4db2ff")
define mystery = DynamicCharacter("mystery_name", color="#ff77aa")

default player_name = "Player"
default mystery_name = "???"
default inventory = ["compass", "journal"]
default affection = 0

label start:
    mystery "You do not yet know who I am."
    "The protagonist observes the hooded figure."
    extend " There is a quiet stillness in the air."

    e "Hello, [player_name]! Welcome to Ren'Py 8.5.3."
    
    $ mystery_name = "Sylvia"
    mystery "At last, my true name is revealed."

    $ renpy.say(e, _("The journey begins now."))
    return
"""
    (scripts_dir / "sample_8x.rpy").write_text(sample_8x, encoding="utf-8")

    # 3. screens.rpy
    screens_rpy = """# Screen Language 2 (SL2) Fixture
screen main_menu():
    tag menu
    style_prefix "main_menu"

    vbox:
        text _("Renslate Test Game") size 36
        textbutton _("Start") action Start() tooltip _("Begin a new adventure")
        textbutton _("Load") action ShowMenu("load") tooltip _("Load saved game state")
        textbutton _("Preferences") action ShowMenu("preferences")
        textbutton _("Quit") action Quit(confirm=False)

screen say(who, what):
    style_prefix "say"

    window:
        id "window"
        if who is not None:
            text who id "who"
        text what id "what"
"""
    (scripts_dir / "screens.rpy").write_text(screens_rpy, encoding="utf-8")

    # 4. dialogue_normalization.rpy
    norm_rpy = """# Dialogue Normalization Test Script (4 Say forms & dynamic identity)
define a = Character("Alice")
define anon = DynamicCharacter("curr_name")
default curr_name = "???"

label dialogue_test:
    # 1. Explicit who != None
    a "Form 1: Explicit character dialogue."

    # 2. Narration who == None
    "Form 2: Implicit narration dialogue."

    # 3. Extend
    extend " Continued narration text."

    # 4. Direct renpy.say invocation
    $ renpy.say(a, "Form 4: Direct renpy.say call.")

    # UI localization strings
    $ msg1 = _("Save Game Now")
    $ msg2 = __("Deferred String Resolution")

    return
"""
    (scripts_dir / "dialogue_normalization.rpy").write_text(norm_rpy, encoding="utf-8")
    print(f"[+] Scripts created in {scripts_dir}")

import types
sys.modules['renpy'] = types.ModuleType('renpy')
sys.modules['renpy.ast'] = types.ModuleType('renpy.ast')

class Say:
    def __init__(self, who, what):
        self.who = who
        self.what = what
        self.filename = "script.rpy"
        self.linenumber = 10
        self.with_ = None
        self.interact = True
        self.attributes = None
        self.arguments = None
        self.temporary_attributes = None
        self.rollback = "force"
Say.__module__ = "renpy.ast"
sys.modules['renpy.ast'].Say = Say

class Label:
    def __init__(self, name):
        self.name = name
        self.filename = "script.rpy"
        self.linenumber = 1
        self.parameters = None
        self.block = []
        self.hide = False
Label.__module__ = "renpy.ast"
sys.modules['renpy.ast'].Label = Label

class Init:
    def __init__(self, priority, block):
        self.priority = priority
        self.block = block
        self.filename = "script.rpy"
        self.linenumber = 1
Init.__module__ = "renpy.ast"
sys.modules['renpy.ast'].Init = Init

def create_rpyc():
    rpyc_dir = FIXTURES_DIR / "rpyc"
    rpyc_dir.mkdir(parents=True, exist_ok=True)

    ast_nodes_list = [
        Label("start"),
        Say("Sylvia", "Hello from compiled RPYC!"),
        Say(None, "Narration follows."),
    ]

    # Helper to build RPYC binary: RENPY RPC-2 header + slot directory + zlib pickle
    def build_rpyc_bytes(nodes, proto):
        header = b"RENPY RPC-2"
        # Slot 1: AST payload
        raw_pickle = pickle.dumps(nodes, protocol=proto)
        compressed = zlib.compress(raw_pickle)
        slot_data = compressed
        # Slot table dict: {slot_number: offset} or direct stream
        slot_table = {1: (len(header) + 64, len(slot_data))}
        slot_pickle = pickle.dumps(slot_table, protocol=proto)

        # Pad to slot offset
        padding_needed = 64 - len(slot_pickle)
        if padding_needed > 0:
            slot_pickle += b" " * padding_needed

        payload = header + slot_pickle + slot_data
        return payload

    # 1. Ren'Py 7.x (Python 2 pickle protocol 2)
    rpyc_7x = build_rpyc_bytes(ast_nodes_list, proto=2)
    (rpyc_dir / "sample_7x.rpyc").write_bytes(rpyc_7x)

    # 2. Ren'Py 8.x (Python 3 pickle protocol 4)
    rpyc_8x = build_rpyc_bytes(ast_nodes_list, proto=4)
    (rpyc_dir / "sample_8x.rpyc").write_bytes(rpyc_8x)

    # 3. Malicious RPYC containing os.system opcode
    # Protocol 2 pickle with GLOBAL 'os.system' and REDUCE
    malicious_inner = b"\x80\x02cos\nsystem\n(X\x06\x00\x00\x00whoamiiqtR."
    malicious_compressed = zlib.compress(malicious_inner)
    malicious_payload = b"RENPY RPC-2" + b" " * 64 + malicious_compressed
    (rpyc_dir / "malicious.rpyc").write_bytes(malicious_payload)

    print(f"[+] RPYC fixtures created in {rpyc_dir}")

def create_rpa():
    rpa_dir = FIXTURES_DIR / "rpa"
    rpa_dir.mkdir(parents=True, exist_ok=True)

    # Common files to bundle
    file_data = {
        "script.rpy": b"# In-archive script\nlabel archive_start:\n    'Inside RPA archive!'\n    return\n",
        "assets/ui_button.png": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRmock_png_data_for_rpa_test",
        "data/dialogue.json": b'{"dialogue": ["Line 1 in archive", "Line 2 in archive"]}',
    }

    # 1. RPA-1.0: paired .rpa and .rpi
    rpa1_data_path = rpa_dir / "archive_v1.rpa"
    rpa1_idx_path = rpa_dir / "archive_v1.rpi"

    rpa1_bytes = bytearray()
    rpa1_index = {}
    for filename, content in file_data.items():
        offset = len(rpa1_bytes)
        length = len(content)
        rpa1_bytes.extend(content)
        rpa1_index[filename] = [(offset, length, "")]

    rpa1_data_path.write_bytes(rpa1_bytes)
    # .rpi is zlib compressed pickle (protocol 2)
    rpi_pickle = pickle.dumps(rpa1_index, protocol=2)
    rpa1_idx_path.write_bytes(zlib.compress(rpi_pickle))

    # 2. RPA-2.0: "RPA-2.0 <16-hex-offset>\n"
    rpa2_path = rpa_dir / "archive_v2.rpa"
    rpa2_bytes = bytearray()
    header_v2_len = len(f"RPA-2.0 {0:016x}\n".encode("latin1"))

    # Reserve header space
    rpa2_bytes.extend(b"\x00" * header_v2_len)
    rpa2_index = {}
    for filename, content in file_data.items():
        offset = len(rpa2_bytes)
        length = len(content)
        rpa2_bytes.extend(content)
        rpa2_index[filename] = [(offset, length, "")]

    idx_offset_v2 = len(rpa2_bytes)
    idx_v2_compressed = zlib.compress(pickle.dumps(rpa2_index, protocol=2))
    rpa2_bytes.extend(idx_v2_compressed)

    # Write true header
    header_v2 = f"RPA-2.0 {idx_offset_v2:016x}\n".encode("latin1")
    rpa2_bytes[:header_v2_len] = header_v2
    rpa2_path.write_bytes(rpa2_bytes)

    # 3. RPA-3.0: "RPA-3.0 <16-hex-offset> <8-hex-key>\n"
    rpa3_path = rpa_dir / "archive_v3.rpa"
    key = 0x42f01984
    header_v3_len = 34  # "RPA-3.0 " (8) + 16-hex + " " (1) + 8-hex + "\n" (1) = 34
    rpa3_bytes = bytearray(b"\x00" * header_v3_len)
    rpa3_index = {}
    for filename, content in file_data.items():
        offset = len(rpa3_bytes)
        length = len(content)
        rpa3_bytes.extend(content)
        # In RPA-3.0 index, offset and length are XOR'ed with 32-bit key
        rpa3_index[filename] = [(offset ^ key, length ^ key, b"")]

    idx_offset_v3 = len(rpa3_bytes)
    idx_v3_compressed = zlib.compress(pickle.dumps(rpa3_index, protocol=3))
    rpa3_bytes.extend(idx_v3_compressed)

    header_v3 = f"RPA-3.0 {idx_offset_v3:016x} {key:08x}\n".encode("latin1")
    assert len(header_v3) == header_v3_len
    rpa3_bytes[:header_v3_len] = header_v3
    rpa3_path.write_bytes(rpa3_bytes)

    # 4. Custom 8-byte header: "ARC-3.0 <16-hex-offset> <8-hex-key>\n"
    rpa_custom_path = rpa_dir / "archive_custom.arc"
    cust_key = 0x11223344
    header_cust_len = 34
    cust_bytes = bytearray(b"\x00" * header_cust_len)
    cust_index = {}
    for filename, content in file_data.items():
        offset = len(cust_bytes)
        length = len(content)
        cust_bytes.extend(content)
        cust_index[filename] = [(offset ^ cust_key, length ^ cust_key, "")]

    idx_offset_cust = len(cust_bytes)
    idx_cust_compressed = zlib.compress(pickle.dumps(cust_index, protocol=2))
    cust_bytes.extend(idx_cust_compressed)

    header_cust = f"ARC-3.0 {idx_offset_cust:016x} {cust_key:08x}\n".encode("latin1")
    assert len(header_cust) == header_cust_len
    cust_bytes[:header_cust_len] = header_cust
    rpa_custom_path.write_bytes(cust_bytes)

    print(f"[+] RPA fixtures created in {rpa_dir}")

def main():
    print(f"[*] Generating Ren'Py fixtures into: {FIXTURES_DIR}")
    create_scripts()
    create_rpyc()
    create_rpa()
    print("[+] All fixtures generated successfully!")

if __name__ == "__main__":
    main()
