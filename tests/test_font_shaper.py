"""
tests/test_font_shaper.py
Unit and Integration Tests for Native OpenType Text Shaper, 134-Glyph Vietnamese
NFC Cmap Auditor, and UI Overflow Detection Pipeline.
"""

from pathlib import Path
import pytest
import renloc_native

from renloc_core.shaper.font_shaper import (
    GuiMetrics,
    GuiParser,
    FontResolver,
    CmapAuditor,
    TextShaper,
    OverflowAuditor,
    FontShaperPipeline,
    strip_renpy_tags,
    VIETNAMESE_NFC_CHARS,
)

ADWAITA_FONT = Path("/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf")
VERA_FONT = Path("/usr/share/fonts/TTF/Vera.ttf")
DEJAVU_FONT = Path("/usr/share/fonts/TTF/DejaVuSans.ttf")


def test_vietnamese_134_char_constant():
    """Verify reference constant contains exactly 134 unique characters."""
    chars = set(VIETNAMESE_NFC_CHARS)
    assert len(chars) == 134
    assert "đ" in chars and "Đ" in chars
    assert "ả" in chars and "ế" in chars and "ự" in chars


def test_strip_renpy_tags():
    """Verify Ren'Py formatting tags and escape sequences are properly stripped."""
    raw = '{b}Xin chào{/b} [player_name]! {color=#ff0000}Cảnh báo:{/color} {{ngoặc nhọn}} [[ngoặc vuông]]\\nThêm dòng mới\\ '
    cleaned = strip_renpy_tags(raw)
    assert "{b}" not in cleaned
    assert "{/b}" not in cleaned
    assert "{color=#ff0000}" not in cleaned
    assert "{ngoặc nhọn}" in cleaned
    assert "[ngoặc vuông]" in cleaned
    assert "\nThêm dòng mới " in cleaned
    assert "Xin chào [player_name]! Cảnh báo:" in cleaned


def test_gui_parser_content():
    """Verify parsing of gui.rpy metrics and font declarations."""
    sample_gui = """
    init -1 python:
        gui.init(1920, 1080)

    define gui.text_font = "fonts/NotoSans-Regular.ttf"
    define gui.name_text_font = "fonts/NotoSans-Bold.ttf"
    define gui.interface_text_font = "fonts/NotoSans-Medium.otf"

    define gui.text_size = 24
    define gui.name_text_size = 32
    define gui.dialogue_width = 820
    define gui.dialogue_height = 200
    define gui.choice_button_width = 850
    define gui.choice_button_height = 50

    style my_text:
        font "fonts/SpecialTitle.ttf"
    """
    metrics = GuiParser.parse_gui_content(sample_gui)
    assert metrics.text_font == "fonts/NotoSans-Regular.ttf"
    assert metrics.name_text_font == "fonts/NotoSans-Bold.ttf"
    assert metrics.interface_text_font == "fonts/NotoSans-Medium.otf"
    assert metrics.text_size == 24.0
    assert metrics.name_text_size == 32.0
    assert metrics.dialogue_width == 820.0
    assert metrics.dialogue_height == 200.0
    assert metrics.choice_button_width == 850.0
    assert "fonts/SpecialTitle.ttf" in metrics.all_declared_fonts
    assert len(metrics.all_declared_fonts) >= 4


def test_cmap_auditor_full_coverage():
    """Verify auditing font that completely supports 134 Vietnamese NFC glyphs."""
    font_path = ADWAITA_FONT if ADWAITA_FONT.exists() else DEJAVU_FONT
    assert font_path.exists(), "Test font must exist on system"
    font_bytes = font_path.read_bytes()

    result = CmapAuditor.audit_font("AdwaitaMono", font_bytes)
    assert result["total_required"] == 134
    assert result["supported_count"] == 134
    assert result["coverage_ratio"] == 1.0
    assert result["is_fully_compatible"] is True
    assert result["missing_chars"] == ""


def test_cmap_auditor_missing_glyphs_and_fontgroup_generation():
    """Verify auditing font lacking Vietnamese glyphs (e.g. Vera) and generating FontGroup DSL."""
    assert VERA_FONT.exists(), "Vera font must exist for missing glyphs test"
    vera_bytes = VERA_FONT.read_bytes()

    result = CmapAuditor.audit_font("Vera", vera_bytes)
    assert result["total_required"] == 134
    # Vera lacks Vietnamese diacritics
    assert result["supported_count"] < 134
    assert result["is_fully_compatible"] is False
    assert len(result["missing_chars"]) > 0

    dsl = CmapAuditor.generate_fontgroup_dsl(
        group_name="vera_vietnamese",
        primary_font_path="fonts/Vera.ttf",
        fallback_font_path="fonts/DejaVuSans.ttf",
        audit_result=result,
    )
    assert "init python:" in dsl
    assert 'renpy.register_font_group("vera_vietnamese",' in dsl
    assert 'FontGroup().add("fonts/Vera.ttf", 0x0020, 0x007e)' in dsl
    assert '.add("fonts/DejaVuSans.ttf", 0x00c0, 0x1ef9)' in dsl
    assert '.add("fonts/Vera.ttf")' in dsl


def test_text_shaping_and_bounding_box_measurement():
    """Verify text shaping metrics and bounding box calculation with native HarfBuzz."""
    font_path = ADWAITA_FONT if ADWAITA_FONT.exists() else DEJAVU_FONT
    font_bytes = font_path.read_bytes()

    text = "Hệ thống bản địa hóa Visual Novel Renslate hỗ trợ tiếng Việt."
    metrics = TextShaper.shape_and_measure(
        font_data=font_bytes,
        font_size_pt=18.0,
        text=text,
        max_box_width_px=1200.0,
        max_box_height_px=100.0,
    )
    assert metrics["total_advance_pixels"] > 0
    assert metrics["max_height_pixels"] > 0
    assert metrics["glyph_count"] > 0
    assert metrics["is_overflowing"] is False


def test_overflow_auditor_detection():
    """Verify overflow detection when dialogue exceeds dialogue box limits."""
    font_path = ADWAITA_FONT if ADWAITA_FONT.exists() else DEJAVU_FONT
    font_bytes = font_path.read_bytes()

    short_dialogue = {"id": "diag_01", "who": "Eileen", "text": "Xin chào!"}
    long_dialogue = {
        "id": "diag_02",
        "who": "Narrator",
        "text": "Đây là một đoạn hội thoại rất dài nhằm mục đích kiểm tra tính năng phát hiện tràn khung chữ của Renslate HarfBuzz Text Shaper khi hiển thị trên giao diện Ren'Py.",
    }

    # Width limit 300px: short fits, long overflows
    res_short = OverflowAuditor.audit_dialogue_line(
        dialogue_item=short_dialogue,
        font_data=font_bytes,
        font_size_pt=20.0,
        max_width_px=300.0,
    )
    assert res_short is None

    res_long = OverflowAuditor.audit_dialogue_line(
        dialogue_item=long_dialogue,
        font_data=font_bytes,
        font_size_pt=20.0,
        max_width_px=300.0,
    )
    assert res_long is not None
    assert res_long["id"] == "diag_02"
    assert res_long["is_overflowing"] is True
    assert res_long["overflow_pixels"] > 0
    assert res_long["total_advance_pixels"] > 300.0

    # Batch test
    reports, summary = OverflowAuditor.audit_dialogue_batch(
        dialogues=[short_dialogue, long_dialogue],
        font_data=font_bytes,
        font_size_pt=20.0,
        max_width_px=300.0,
    )
    assert len(reports) == 1
    assert summary["total_dialogues_checked"] == 2
    assert summary["overflow_count"] == 1
    assert summary["overflow_rate"] == 0.5


def test_font_shaper_pipeline_e2e(tmp_path):
    """Verify complete FontShaperPipeline on an end-to-end simulated game directory."""
    game_dir = tmp_path / "game"
    game_dir.mkdir()
    fonts_dir = game_dir / "fonts"
    fonts_dir.mkdir()

    # Place Vera font as primary game font
    assert VERA_FONT.exists()
    vera_target = fonts_dir / "Vera.ttf"
    vera_target.write_bytes(VERA_FONT.read_bytes())

    gui_rpy = game_dir / "gui.rpy"
    gui_rpy.write_text(
        """
        define gui.text_font = "fonts/Vera.ttf"
        define gui.text_size = 22
        define gui.dialogue_width = 400
        define gui.dialogue_height = 100
        """,
        encoding="utf-8",
    )

    dialogues = [
        {"id": "line_1", "who": "Sylvia", "text": "Hôm nay trời đẹp quá."},
        {"id": "line_2", "who": "Sylvia", "text": "Dòng văn bản này cực kỳ dài và chắc chắn sẽ bị tràn hộp thoại với chiều rộng 400 pixels!"},
    ]

    pipeline = FontShaperPipeline(game_dir=game_dir, default_fallback_font="fonts/DejaVuSans.ttf")
    report = pipeline.run_full_audit(gui_rpy_path=gui_rpy, dialogues=dialogues)

    # 1. Check GUI metrics parsed
    assert report["gui_metrics"]["dialogue_width"] == 400.0
    assert report["gui_metrics"]["text_font"] == "fonts/Vera.ttf"

    # 2. Check font audits: Vera detected as missing Vietnamese
    font_audits = report["font_audits"]
    assert "fonts/Vera.ttf" in font_audits
    assert font_audits["fonts/Vera.ttf"]["is_fully_compatible"] is False

    # 3. Check FontGroup DSL generated
    dsls = report["fontgroup_dsls"]
    assert "fonts/Vera.ttf" in dsls
    assert "renpy.register_font_group" in dsls["fonts/Vera.ttf"]

    # 4. Check dialogue overflow detected
    assert len(report["overflow_reports"]) >= 1
    assert report["overflow_reports"][0]["id"] == "line_2"
    assert report["overflow_summary"]["overflow_count"] >= 1
