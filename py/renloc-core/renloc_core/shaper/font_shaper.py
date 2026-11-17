"""
renloc_core/shaper/font_shaper.py
Native OpenType Text Shaper and 134-Glyph Vietnamese NFC Cmap Auditor Pipeline.
Performs 3-step audit:
  1. Collects font declarations and metrics from gui.rpy, screens.rpy, and scripts.
  2. Scans font cmap tables against the 134 Vietnamese NFC Unicode glyphs and generates
     automatic Ren'Py FontGroup DSL fallback configurations.
  3. Shapes dialogue lines with native HarfBuzz / rustybuzz and detects pixel-level UI
     bounding box overflow against dialogue box / button limits.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import os
import re
from pathlib import Path

import renloc_native

# 134 Vietnamese NFC characters reference string
VIETNAMESE_NFC_CHARS = (
    "àÀảẢãÃáÁạẠ"
    "ăĂằẰẳẲẵẴắẮặẶ"
    "âÂầẦẩẨẫẪấẤậẬ"
    "èÈẻẺẽẼéÉẹẸ"
    "êÊềỀểỂễỄếẾệỆ"
    "ìÌỉỈĩĨíÍịỊ"
    "òÒỏỎõÕóÓọỌ"
    "ôÔồỒổỔỗỖốỐộỘ"
    "ơƠờỜởỞỡỠớỚợỢ"
    "ùÙủỦũŨúÚụỤ"
    "ưƯừỪửỬữỮứỨựỰ"
    "ỳỲỷỶỹỸýÝỵỴ"
    "đĐ"
)


def strip_renpy_tags(text: str) -> str:
    """
    Strips Ren'Py text styling tags ({b}, {i}, {color=...}, {size=...}, {w}, {p}, etc.)
    and unescapes escaped brackets ({{ -> {, [[ -> [) and escaped spaces (\\  ->  )
    so HarfBuzz shapes only the actual rendered visual text.
    """
    if not text:
        return ""
    # Unescape double brackets before removing single tags
    placeholder_open_brace = "\uE000"
    placeholder_open_bracket = "\uE001"
    t = text.replace("{{", placeholder_open_brace).replace("[[", placeholder_open_bracket)
    
    # Strip Ren'Py text formatting tags: {tag} or {tag=value} or {/tag}
    t = re.sub(r"\{/?[a-zA-Z0-9_#]+(?:=[^{}]*)?\}", "", t)
    
    # Restore escaped brackets
    t = t.replace(placeholder_open_brace, "{").replace(placeholder_open_bracket, "[")
    
    # Handle common escape sequences
    t = t.replace("\\ ", " ").replace("\\n", "\n").replace('\\"', '"')
    return t


@dataclass
class GuiMetrics:
    dialogue_width: float = 740.0
    dialogue_height: Optional[float] = 180.0
    text_size: float = 22.0
    name_text_size: float = 30.0
    choice_button_width: float = 790.0
    choice_button_height: Optional[float] = 45.0
    text_font: Optional[str] = None
    name_text_font: Optional[str] = None
    interface_text_font: Optional[str] = None
    all_declared_fonts: Set[str] = field(default_factory=set)


class GuiParser:
    """
    Parses gui.rpy and other Ren'Py script files to extract font paths and layout dimensions.
    """

    @classmethod
    def parse_gui_content(cls, content: str) -> GuiMetrics:
        metrics = GuiMetrics()

        # Regular expressions for gui.rpy declarations
        re_text_font = re.compile(r'define\s+gui\.text_font\s*=\s*["\']([^"\']+)["\']')
        re_name_font = re.compile(r'define\s+gui\.name_text_font\s*=\s*["\']([^"\']+)["\']')
        re_interface_font = re.compile(r'define\s+gui\.interface_text_font\s*=\s*["\']([^"\']+)["\']')

        re_text_size = re.compile(r'define\s+gui\.text_size\s*=\s*([0-9]+(?:\.[0-9]+)?)')
        re_name_text_size = re.compile(r'define\s+gui\.name_text_size\s*=\s*([0-9]+(?:\.[0-9]+)?)')
        re_dialogue_width = re.compile(r'define\s+gui\.dialogue_width\s*=\s*([0-9]+(?:\.[0-9]+)?)')
        re_dialogue_height = re.compile(r'define\s+gui\.dialogue_height\s*=\s*([0-9]+(?:\.[0-9]+)?)')
        re_choice_width = re.compile(r'define\s+gui\.choice_button_width\s*=\s*([0-9]+(?:\.[0-9]+)?)')
        re_choice_height = re.compile(r'define\s+gui\.choice_button_height\s*=\s*([0-9]+(?:\.[0-9]+)?)')

        if m := re_text_font.search(content):
            metrics.text_font = m.group(1)
            metrics.all_declared_fonts.add(m.group(1))

        if m := re_name_font.search(content):
            metrics.name_text_font = m.group(1)
            metrics.all_declared_fonts.add(m.group(1))

        if m := re_interface_font.search(content):
            metrics.interface_text_font = m.group(1)
            metrics.all_declared_fonts.add(m.group(1))

        if m := re_text_size.search(content):
            metrics.text_size = float(m.group(1))

        if m := re_name_text_size.search(content):
            metrics.name_text_size = float(m.group(1))

        if m := re_dialogue_width.search(content):
            metrics.dialogue_width = float(m.group(1))

        if m := re_dialogue_height.search(content):
            metrics.dialogue_height = float(m.group(1))

        if m := re_choice_width.search(content):
            metrics.choice_button_width = float(m.group(1))

        if m := re_choice_height.search(content):
            metrics.choice_button_height = float(m.group(1))

        # Scan for any additional font declarations: font "..." or style ... font "..."
        re_generic_font = re.compile(r'\bfont\s+["\']([^"\']+\.(?:ttf|otf|ttc))["\']', re.IGNORECASE)
        for m in re_generic_font.finditer(content):
            metrics.all_declared_fonts.add(m.group(1))

        return metrics

    @classmethod
    def scan_directory_for_fonts(cls, game_dir: Path) -> Set[str]:
        fonts: Set[str] = set()
        re_font = re.compile(r'\bfont\s+["\']([^"\']+\.(?:ttf|otf|ttc))["\']', re.IGNORECASE)
        re_font_define = re.compile(r'define\s+gui\.[a-zA-Z0-9_]*font\s*=\s*["\']([^"\']+)["\']')

        if not game_dir.exists():
            return fonts

        for ext in ("*.rpy", "*.rpym"):
            for p in game_dir.rglob(ext):
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    for m in re_font.finditer(text):
                        fonts.add(m.group(1))
                    for m in re_font_define.finditer(text):
                        fonts.add(m.group(1))
                except Exception:
                    pass
        return fonts


class FontResolver:
    """
    Locates and reads font binary bytes from loose game files, VFS archives, or fallback system paths.
    """

    SYSTEM_FONT_DIRS = [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        Path.home() / ".local/share/fonts",
        Path("C:/Windows/Fonts"),
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
    ]

    @classmethod
    def resolve_font_bytes(
        cls,
        font_path_name: str,
        game_dir: Optional[Path] = None,
        vfs: Optional[Any] = None,
    ) -> Optional[Tuple[str, bytes]]:
        """
        Attempts to load font bytes for font_path_name.
        Returns tuple (resolved_name_or_path, font_bytes) or None if not found.
        """
        norm_name = font_path_name.replace("\\", "/").lstrip("/")

        # 1. Try VfsLayer if provided
        if vfs is not None:
            for candidate in (norm_name, f"fonts/{norm_name}", f"gui/{norm_name}"):
                try:
                    data = bytes(vfs.read_file(candidate))
                    if len(data) > 0:
                        return (candidate, data)
                except Exception:
                    pass

        # 2. Try direct path under game_dir
        if game_dir is not None:
            for candidate_rel in (norm_name, f"fonts/{norm_name}", f"gui/{norm_name}"):
                candidate_path = game_dir / candidate_rel
                if candidate_path.is_file():
                    try:
                        return (str(candidate_path), candidate_path.read_bytes())
                    except Exception:
                        pass

        # 3. If direct absolute or relative file path on disk
        direct_path = Path(font_path_name)
        if direct_path.is_file():
            try:
                return (str(direct_path), direct_path.read_bytes())
            except Exception:
                pass

        # 4. Search system font directories
        font_basename = Path(norm_name).name
        for sdir in cls.SYSTEM_FONT_DIRS:
            if sdir.exists():
                for found in sdir.rglob(font_basename):
                    if found.is_file():
                        try:
                            return (str(found), found.read_bytes())
                        except Exception:
                            pass

        return None


class CmapAuditor:
    """
    Audits font cmap table for all 134 Vietnamese NFC characters and produces FontGroup DSL.
    """

    @classmethod
    def audit_font(cls, font_name: str, font_data: bytes) -> Dict[str, Any]:
        """
        Audits font bytes using renloc_native.audit_font_bytes.
        """
        return renloc_native.audit_font_bytes(font_name, font_data)

    @classmethod
    def generate_fontgroup_dsl(
        cls,
        group_name: str,
        primary_font_path: str,
        fallback_font_path: str,
        audit_result: Dict[str, Any],
    ) -> str:
        """
        Generates Ren'Py FontGroup DSL configuration code to seamlessly fallback missing Vietnamese glyphs.
        """
        dsl_lines = [
            "init python:",
            f'    renpy.register_font_group("{group_name}",',
            f'        FontGroup().add("{primary_font_path}", 0x0020, 0x007e)',
        ]
        if not audit_result.get("is_fully_compatible", False):
            dsl_lines.append(f'        .add("{fallback_font_path}", 0x00c0, 0x1ef9) # Vietnamese Vowels & Tone Marks')
        dsl_lines.append(f'        .add("{primary_font_path}") # Default fallback')
        dsl_lines.append("    )")
        return "\n".join(dsl_lines) + "\n"


class TextShaper:
    """
    HarfBuzz OpenType layout engine and pixel bounding box overflow detector.
    """

    @classmethod
    def shape_and_measure(
        cls,
        font_data: bytes,
        font_size_pt: float,
        text: str,
        max_box_width_px: Optional[float] = None,
        max_box_height_px: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Shapes text using HarfBuzz and returns pixel dimensions and overflow flag.
        Strips Ren'Py styling tags before measuring.
        """
        clean_text = strip_renpy_tags(text)
        return renloc_native.shape_and_measure(
            font_data,
            font_size_pt,
            clean_text,
            max_box_width_px,
            max_box_height_px,
        )


class OverflowAuditor:
    """
    Audits dialogue lines for UI overflow against gui.rpy dimensions.
    """

    @classmethod
    def audit_dialogue_line(
        cls,
        dialogue_item: Dict[str, Any],
        font_data: bytes,
        font_size_pt: float,
        max_width_px: float,
        max_height_px: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        raw_text = dialogue_item.get("text", "") or dialogue_item.get("translation", "")
        if not raw_text:
            return None

        metrics = TextShaper.shape_and_measure(
            font_data=font_data,
            font_size_pt=font_size_pt,
            text=raw_text,
            max_box_width_px=max_width_px,
            max_box_height_px=max_height_px,
        )

        if metrics.get("is_overflowing", False):
            adv = metrics.get("total_advance_pixels", 0.0)
            diff = adv - max_width_px
            return {
                "id": dialogue_item.get("id", ""),
                "who": dialogue_item.get("who", ""),
                "text": raw_text,
                "clean_text": strip_renpy_tags(raw_text),
                "total_advance_pixels": adv,
                "max_height_pixels": metrics.get("max_height_pixels", 0.0),
                "max_allowed_width": max_width_px,
                "max_allowed_height": max_height_px,
                "overflow_pixels": round(diff, 2),
                "glyph_count": metrics.get("glyph_count", 0),
                "is_overflowing": True,
            }
        return None

    @classmethod
    def audit_dialogue_batch(
        cls,
        dialogues: List[Dict[str, Any]],
        font_data: bytes,
        font_size_pt: float,
        max_width_px: float,
        max_height_px: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        overflow_reports: List[Dict[str, Any]] = []
        total_checked = len(dialogues)

        for d in dialogues:
            report = cls.audit_dialogue_line(
                dialogue_item=d,
                font_data=font_data,
                font_size_pt=font_size_pt,
                max_width_px=max_width_px,
                max_height_px=max_height_px,
            )
            if report is not None:
                overflow_reports.append(report)

        summary = {
            "total_dialogues_checked": total_checked,
            "overflow_count": len(overflow_reports),
            "overflow_rate": round(len(overflow_reports) / max(total_checked, 1), 4),
            "max_width_limit": max_width_px,
            "max_height_limit": max_height_px,
        }
        return overflow_reports, summary


class FontShaperPipeline:
    """
    End-to-end 3-step pipeline:
      1. Collect fonts & parse GUI bounds
      2. Audit cmap & generate FontGroup DSL fallback
      3. Shape dialogue & detect UI overflow
    """

    def __init__(
        self,
        game_dir: Optional[Path] = None,
        vfs: Optional[Any] = None,
        default_fallback_font: str = "DejaVuSans.ttf",
    ):
        self.game_dir = game_dir
        self.vfs = vfs
        self.default_fallback_font = default_fallback_font

    def run_full_audit(
        self,
        gui_rpy_path: Optional[Path] = None,
        dialogues: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete font and dialogue shaping audit.
        """
        # Step 1: Parse gui.rpy metrics
        gui_metrics = GuiMetrics()
        gui_text = ""
        if gui_rpy_path and gui_rpy_path.is_file():
            gui_text = gui_rpy_path.read_text(encoding="utf-8", errors="ignore")
        elif self.game_dir:
            candidate = self.game_dir / "gui.rpy"
            if candidate.is_file():
                gui_text = candidate.read_text(encoding="utf-8", errors="ignore")

        if gui_text:
            gui_metrics = GuiParser.parse_gui_content(gui_text)

        # Collect additional fonts from game directory
        if self.game_dir:
            additional_fonts = GuiParser.scan_directory_for_fonts(self.game_dir)
            gui_metrics.all_declared_fonts.update(additional_fonts)

        # Step 2: Audit cmap for all discovered fonts
        font_audits: Dict[str, Any] = {}
        fontgroup_dsls: Dict[str, str] = {}
        font_bytes_cache: Dict[str, bytes] = {}

        for font_name in sorted(gui_metrics.all_declared_fonts):
            resolved = FontResolver.resolve_font_bytes(
                font_name,
                game_dir=self.game_dir,
                vfs=self.vfs,
            )
            if resolved:
                res_path, f_bytes = resolved
                font_bytes_cache[font_name] = f_bytes
                audit = CmapAuditor.audit_font(font_name, f_bytes)
                font_audits[font_name] = audit

                if not audit.get("is_fully_compatible", False):
                    dsl = CmapAuditor.generate_fontgroup_dsl(
                        group_name=f"{Path(font_name).stem}_viet",
                        primary_font_path=font_name,
                        fallback_font_path=self.default_fallback_font,
                        audit_result=audit,
                    )
                    fontgroup_dsls[font_name] = dsl

        # Step 3: Shape dialogue lines and audit overflow
        overflow_reports: List[Dict[str, Any]] = []
        overflow_summary: Dict[str, Any] = {}

        if dialogues:
            # Determine active dialogue font
            active_font = gui_metrics.text_font or "DejaVuSans.ttf"
            font_data = font_bytes_cache.get(active_font)

            # If not in cache, resolve directly
            if not font_data:
                resolved = FontResolver.resolve_font_bytes(active_font, self.game_dir, self.vfs)
                if resolved:
                    font_data = resolved[1]

            if font_data:
                overflow_reports, overflow_summary = OverflowAuditor.audit_dialogue_batch(
                    dialogues=dialogues,
                    font_data=font_data,
                    font_size_pt=gui_metrics.text_size,
                    max_width_px=gui_metrics.dialogue_width,
                    max_height_px=gui_metrics.dialogue_height,
                )

        return {
            "gui_metrics": {
                "dialogue_width": gui_metrics.dialogue_width,
                "dialogue_height": gui_metrics.dialogue_height,
                "text_size": gui_metrics.text_size,
                "name_text_size": gui_metrics.name_text_size,
                "text_font": gui_metrics.text_font,
                "name_text_font": gui_metrics.name_text_font,
                "all_fonts": list(gui_metrics.all_declared_fonts),
            },
            "font_audits": font_audits,
            "fontgroup_dsls": fontgroup_dsls,
            "overflow_reports": overflow_reports,
            "overflow_summary": overflow_summary,
        }
