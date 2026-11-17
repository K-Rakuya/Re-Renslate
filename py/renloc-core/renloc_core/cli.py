"""
cli.py
Unified Command-Line Interface for Renslate.
Usage:
  renslate scan <game_dir> [--db <path>]
  renslate font-audit <game_dir> [--fallback-font <path>]
  renslate translate <game_dir> [--engine <vllm|sglang|llamacpp>] [--model <name>]
  renslate export <game_dir> [--output <path>]
  renslate serve [--socket <path>] [--token <hex>]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from renloc_core.ast_reader.unpickler import NativeRenpyUnpickler
from renloc_core.db_writer import DbWriter
from renloc_core.synthesizer import (
    FontAuditor,
    GuiInspector,
    DialogueShaper,
    PackageSynthesizer,
    TranslationItem,
    UIStringTranslationItem,
)
from renloc_core.static_graph.dialogue_normalizer import DialogueNormalizer
from renloc_core.static_graph.screen_extractor import ScreenExtractor

try:
    import renloc_native
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False


def cmd_scan(args: argparse.Namespace) -> None:
    """Scans game directory, unpacks RPAs via VFS, extracts dialogue ASTs into SQLite."""
    game_dir = Path(args.game_dir).resolve()
    if not game_dir.exists():
        print(f"[!] Error: Game directory does not exist: {game_dir}", file=sys.stderr)
        sys.exit(1)

    db_path = Path(args.db).resolve() if args.db else game_dir / "renslate_project.sqlite"
    print(f"[*] Scanning Ren'Py project: {game_dir}")
    print(f"[*] SQLite Database: {db_path}")

    # 1. Initialize VFS layer
    if HAS_NATIVE:
        vfs = renloc_native.VfsLayer(str(game_dir))
        vfs.mount_game_directory()
        files = vfs.list_all_files()
        print(f"[+] VFS mounted: {len(files)} files discovered (including loose & .rpa archives)")
    else:
        files = [str(p.relative_to(game_dir)) for p in game_dir.rglob("*") if p.is_file()]
        print(f"[+] Loose filesystem scanned: {len(files)} files discovered")

    # 2. Extract dialogue and UI strings
    from renloc_core.db_writer.schema import init_schema
    normalizer = DialogueNormalizer()
    writer = DbWriter(db_path)
    init_schema(writer.raw_connection)

    rpyc_files = [f for f in files if f.endswith(".rpyc") or f.endswith(".rpymc")]
    print(f"[*] Found {len(rpyc_files)} compiled Ren'Py script files (.rpyc/.rpymc)")

    extracted_count = 0
    with writer.write_transaction() as cursor:
        for rpyc_rel in rpyc_files:
            try:
                if HAS_NATIVE:
                    data = vfs.read_file(rpyc_rel)
                else:
                    data = (game_dir / rpyc_rel).read_bytes()

                # Parse AST safely via NativeRenpyUnpickler
                unpickler = NativeRenpyUnpickler(data)
                stmts = unpickler.load_stmts()
                if stmts:
                    dialogues = normalizer.normalize_statements(stmts, rpyc_rel)
                    for d in dialogues:
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO translation_units
                            (unit_identifier, source_text, speaker, status, row_order, file_path, line_number)
                            VALUES (?, ?, ?, 'PENDING', ?, ?, ?)
                            """,
                            (
                                d.translation_id,
                                d.what,
                                d.who,
                                extracted_count + 1,
                                d.filename,
                                d.linenumber,
                            ),
                        )
                        extracted_count += 1
            except Exception as e:
                # Continue on non-fatal script parse warnings
                continue

    print(f"[✓] Scan Complete! Extracted {extracted_count} dialogue units into {db_path}")


def cmd_font_audit(args: argparse.Namespace) -> None:
    """Audits game fonts against 134 Vietnamese NFC glyphs and checks dialogue overflow."""
    game_dir = Path(args.game_dir).resolve()
    print(f"[*] Auditing fonts in Ren'Py project: {game_dir}")

    # Inspect gui.rpy if present
    gui_file = game_dir / "gui.rpy"
    if not gui_file.exists():
        gui_file = game_dir / "game" / "gui.rpy"

    gui_config = None
    if gui_file.exists():
        gui_config = GuiInspector.parse_gui_file(gui_file.read_text(encoding="utf-8"))
        print(f"[+] Loaded GUI parameters: dialogue_width={gui_config.dialogue_width}px, text_size={gui_config.text_size}pt")

    # Find fonts
    font_files = list(game_dir.rglob("*.ttf")) + list(game_dir.rglob("*.otf"))
    if not font_files:
        print("[!] No TTF/OTF fonts found in project directory.")
        return

    print(f"[*] Found {len(font_files)} font files. Checking 134 Vietnamese NFC glyph coverage...")
    all_compatible = True
    for ff in font_files:
        try:
            audit = FontAuditor.audit_font_file(ff)
            status_str = "COMPATIBLE (134/134)" if audit.is_fully_compatible else f"MISSING {len(audit.missing_chars)} GLYPHS ({audit.supported_count}/134)"
            print(f"  - {ff.name}: {status_str}")
            if not audit.is_fully_compatible:
                all_compatible = False
        except Exception as e:
            print(f"  - {ff.name}: Failed to read cmap ({e})")

    if not all_compatible:
        fallback = args.fallback_font or "game/tl/vietnamese/fonts/fallback.ttf"
        dsl = FontAuditor.generate_font_group_dsl(
            group_name="vietnamese_font_group",
            primary_font_path="game/fonts/original.ttf",
            fallback_font_path=fallback,
        )
        print("\n[!] Recommendation: Missing Vietnamese glyphs detected. Auto-generated FontGroup fallback DSL:")
        print("-" * 60)
        print(dsl.strip())
        print("-" * 60)
    else:
        print("\n[✓] All scanned fonts provide 100% coverage for the 134 Vietnamese NFC characters!")


def cmd_export(args: argparse.Namespace) -> None:
    """Exports translation database into game/tl/vietnamese/ directory structure."""
    game_dir = Path(args.game_dir).resolve()
    db_path = Path(args.db).resolve() if args.db else game_dir / "renslate_project.sqlite"

    if not db_path.exists():
        print(f"[!] Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output).resolve() if args.output else game_dir / "game"
    print(f"[*] Exporting translations from: {db_path}")
    print(f"[*] Target directory: {output_dir / 'tl' / 'vietnamese'}")

    writer = DbWriter(db_path)
    dialogue_items = []
    with writer.read_transaction() as cursor:
        rows = cursor.execute(
            """
            SELECT unit_identifier, source_text, target_text, speaker, file_path, line_number
            FROM translation_units
            WHERE target_text IS NOT NULL
            ORDER BY row_order ASC
            """
        ).fetchall()

        for r in rows:
            tlid, src, tgt, who, fpath, line = r
            dialogue_items.append(
                TranslationItem(
                    translation_id=tlid,
                    original_statement_code=f'{who} "{src}"' if who else f'"{src}"',
                    translated_text=tgt,
                    who=who,
                    filename=fpath or "script.rpy",
                    linenumber=line or 1,
                )
            )

    synthesizer = PackageSynthesizer(language="vietnamese")
    created = synthesizer.export_package(output_dir=output_dir, dialogue_items=dialogue_items, string_items=[])

    # Also deploy renloc_runtime_hook.rpy
    hook_src = Path(__file__).resolve().parent.parent.parent.parent / "runtime" / "renpy_hook" / "renloc_runtime_hook.rpy"
    if hook_src.exists():
        target_hook = output_dir / "tl" / "vietnamese" / "renloc_runtime_hook.rpy"
        target_hook.write_text(hook_src.read_text(encoding="utf-8"), encoding="utf-8")
        created["renloc_runtime_hook.rpy"] = target_hook

    print(f"[✓] Export Complete! Created {len(created)} translation and hook files with 0 tabs & 4-space indent:")
    for name in created:
        print(f"  - tl/vietnamese/{name}")


def cmd_gui(args: argparse.Namespace) -> None:
    """Launches the Renslate Desktop GUI (Tauri v2)."""
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    desktop_dir = project_root / "apps" / "desktop"
    print(f"[*] Launching Renslate Desktop GUI from {desktop_dir}...")
    import subprocess
    cmd = ["npm", "run", "dev"]
    subprocess.run(cmd, cwd=str(desktop_dir))


def cmd_serve(args: argparse.Namespace) -> None:
    """Starts the MessagePack RPC sidecar daemon."""
    import asyncio
    from renloc_core.rpc_server import RpcServer, setup_death_pipe_listener

    socket_path = args.socket or os.environ.get("RENSLATE_RPC_SOCKET", "/tmp/renslate_rpc.sock")
    token = args.token or os.environ.get("RENSLATE_AUTH_TOKEN", "")

    server = RpcServer(endpoint_path=socket_path, auth_token=token)

    def handle_death():
        print("[*] Parent process death detected via death-pipe. Terminating sidecar...")
        sys.exit(0)

    setup_death_pipe_listener(handle_death)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    print(f"[*] Starting Renslate RPC Sidecar on {socket_path} (PID: {os.getpid()})")
    loop.run_until_complete(server.start())
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        loop.run_until_complete(server.stop())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="renslate",
        description="Renslate: Autonomous Local-First Ren'Py Localization System",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scan
    p_scan = subparsers.add_parser("scan", help="Scan Ren'Py game project and extract dialogue into SQLite")
    p_scan.add_argument("game_dir", help="Path to Ren'Py game folder")
    p_scan.add_argument("--db", help="Path to output SQLite database")
    p_scan.set_defaults(func=cmd_scan)

    # font-audit
    p_font = subparsers.add_parser("font-audit", help="Audit project fonts for 134 Vietnamese NFC glyphs")
    p_font.add_argument("game_dir", help="Path to Ren'Py game folder")
    p_font.add_argument("--fallback-font", help="Path to fallback font for FontGroup DSL generation")
    p_font.set_defaults(func=cmd_font_audit)

    # export
    p_export = subparsers.add_parser("export", help="Export translations to game/tl/vietnamese/ with 4 spaces and 0 tabs")
    p_export.add_argument("game_dir", help="Path to Ren'Py game folder")
    p_export.add_argument("--db", help="Path to SQLite database")
    p_export.add_argument("--output", help="Output directory (defaults to game/)")
    p_export.set_defaults(func=cmd_export)

    # gui
    p_gui = subparsers.add_parser("gui", help="Launch the Renslate Desktop GUI (Tauri v2)")
    p_gui.set_defaults(func=cmd_gui)

    # serve
    p_serve = subparsers.add_parser("serve", help="Run MessagePack RPC server daemon")
    p_serve.add_argument("--socket", help="Path to Unix Domain Socket or Named Pipe")
    p_serve.add_argument("--token", help="256-bit CSPRNG hex auth token")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
