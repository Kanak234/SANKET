"""theme.py — colours, fonts, text and settings shared by every screen.

WHY: two people read SANKET's screen from up to 1.5 m away, so text size and contrast matter more
     than decoration (UI/UX brief §1, §6). Keeping colours/fonts here makes every screen consistent
     and lets Settings switch Light / Dark / High-contrast in one place.
WHAT: Theme palettes (all text colours ≥ 4.5:1 contrast on their background), `qss()` stylesheet,
      `tr()` for UI strings (Hinglish, Roman script), and `Settings` saved as JSON in the user's
      app-data folder (never camera/audio data).
NEXT: imported by main_window.py and every view.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache

from sanket.paths import resource, user_dir

FONT_STACK = '"Noto Sans", "Nirmala UI", "Segoe UI", "Noto Sans Devanagari", sans-serif'  # Nirmala = Windows Hindi font


@dataclass(frozen=True)
class Palette:
    name: str
    bg: str  # window background
    panel: str  # card / panel background
    text: str
    muted: str  # secondary text (still ≥ 4.5:1)
    accent: str  # deep teal — buttons, focus ring
    accent_text: str
    ok: str  # confident chip (always paired with ✓ icon, never colour alone)
    warn: str  # low-confidence chip (paired with ?)
    bad: str  # error banner (paired with ⚠)
    avatar_body: str
    avatar_hand: str
    avatar_line: str


PALETTES = {
    "light": Palette("light", "#F6F4EF", "#FFFFFF", "#1C1B19", "#55524C", "#0F6E6E", "#FFFFFF",
                     "#1E6B3A", "#8A4B00", "#A61B1B", "#D9E7E6", "#0F6E6E", "#1C1B19"),
    "dark": Palette("dark", "#151A1A", "#1F2626", "#F2F0EA", "#B9C2C0", "#5FC4C4", "#0B1414",
                    "#7BD69A", "#F0B35A", "#FF8A8A", "#2F4A49", "#5FC4C4", "#F2F0EA"),
    "contrast": Palette("contrast", "#000000", "#000000", "#FFFFFF", "#FFFF00", "#FFFF00", "#000000",
                        "#00FF66", "#FFFF00", "#FF5555", "#333333", "#FFFF00", "#FFFFFF"),
}


@dataclass
class Settings:
    """User choices, saved to %LOCALAPPDATA%\\SANKET\\settings.json."""

    theme: str = "light"
    text_px: int = 28  # caption size (UI brief: default 28, range 20–48)
    camera_index: int = 0
    stt_language: str = "hi"
    tts_language: str = "hi"
    avatar_speed: float = 1.0
    threshold: float | None = None  # None = use the calibrated value from training
    contribute: bool = False

    @classmethod
    def load(cls) -> Settings:
        try:
            data = json.loads((user_dir() / "settings.json").read_text(encoding="utf-8"))
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self) -> None:
        (user_dir() / "settings.json").write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


@lru_cache(maxsize=1)
def _strings() -> dict[str, str]:
    return json.loads(resource("sanket", "i18n", "hinglish.json").read_text(encoding="utf-8"))


def tr(key: str) -> str:
    """UI text for `key` (falls back to the key itself so a missing string is visible, not a crash)."""
    return _strings().get(key, key)


def qss(p: Palette, text_px: int) -> str:
    """Qt stylesheet for the whole app. Big targets (≥ 44 px), visible focus ring, no tiny text."""
    small = max(15, int(text_px * 0.6))
    return f"""
    * {{ font-family: {FONT_STACK}; }}
    QMainWindow, QWidget#root {{ background: {p.bg}; color: {p.text}; font-size: {small}px; }}
    QLabel {{ color: {p.text}; }}
    QLabel[role="muted"] {{ color: {p.muted}; }}
    QLabel[role="caption"] {{ font-size: {text_px}px; font-weight: 600; }}
    QLabel[role="title"] {{ font-size: {small + 4}px; font-weight: 700; letter-spacing: 1px; }}
    QFrame[role="panel"] {{ background: {p.panel}; border-radius: 14px; }}
    QPushButton {{ background: {p.panel}; color: {p.text}; border: 2px solid {p.muted};
                   border-radius: 10px; padding: 8px 14px; min-height: 28px; font-size: {small}px; }}
    QPushButton:hover {{ border-color: {p.accent}; }}
    QPushButton:disabled {{ color: {p.muted}; border-style: dotted; background: {p.bg}; }}
    QPushButton:focus {{ border: 3px solid {p.accent}; }}
    QPushButton[role="primary"] {{ background: {p.accent}; color: {p.accent_text}; border-color: {p.accent};
                                   font-weight: 700; min-height: 44px; }}
    QPushButton[role="primary"]:pressed {{ background: {p.text}; }}
    QPushButton[role="nav"] {{ text-align: left; border: none; border-radius: 8px; padding: 10px 14px; }}
    QPushButton[role="nav"]:checked {{ background: {p.accent}; color: {p.accent_text}; }}
    QPushButton[chip="ok"] {{ border-color: {p.ok}; color: {p.ok}; font-weight: 700; }}
    QPushButton[chip="warn"] {{ border-color: {p.warn}; color: {p.warn}; font-weight: 700; }}
    QPushButton[chip="text"] {{ border-style: dashed; border-color: {p.warn}; color: {p.text}; }}
    QPushButton[chip="current"] {{ background: {p.accent}; color: {p.accent_text}; border-color: {p.accent}; }}
    QLineEdit, QComboBox, QListWidget, QTextBrowser, QSpinBox {{ background: {p.panel}; color: {p.text};
                   border: 2px solid {p.muted}; border-radius: 8px; padding: 6px; font-size: {small}px; }}
    QLineEdit:focus, QComboBox:focus, QListWidget:focus {{ border-color: {p.accent}; }}
    QListWidget::item:selected {{ background: {p.accent}; color: {p.accent_text}; }}
    QLabel#banner {{ background: {p.bad}; color: #FFFFFF; border-radius: 8px; padding: 8px 12px; }}
    QLabel#notice {{ background: {p.panel}; color: {p.warn}; border: 2px solid {p.warn};
                     border-radius: 8px; padding: 6px 10px; }}
    QSlider::groove:horizontal {{ height: 6px; background: {p.muted}; border-radius: 3px; }}
    QSlider::handle:horizontal {{ background: {p.accent}; width: 20px; margin: -8px 0; border-radius: 10px; }}
    QCheckBox {{ color: {p.text}; }}
    """
