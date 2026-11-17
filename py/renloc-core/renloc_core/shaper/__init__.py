"""
renloc_core.shaper
Native OpenType Text Shaper and 134-Glyph Vietnamese NFC Cmap Auditor Pipeline.
"""

from .font_shaper import (
    GuiMetrics,
    GuiParser,
    FontResolver,
    CmapAuditor,
    TextShaper,
    OverflowAuditor,
    FontShaperPipeline,
    strip_renpy_tags,
)

__all__ = [
    "GuiMetrics",
    "GuiParser",
    "FontResolver",
    "CmapAuditor",
    "TextShaper",
    "OverflowAuditor",
    "FontShaperPipeline",
    "strip_renpy_tags",
]
