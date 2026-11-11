"""other_views.py — Dictionary, Settings (+ Contribute) and About screens.

WHY one file: these screens are simple forms around core logic; keeping them together makes the
     UI easy to read for a reviewer. The heavy screen (conversation) has its own file.
WHAT:
  • DictionaryView — search all 263 signs in Hindi/English, filter by category, watch the avatar
    perform any sign at 1x or 0.5x, report "this sign looks wrong" (saved locally only).
  • SettingsView  — camera, speech language, voice language, avatar speed, confidence, text size,
    theme; and the opt-in Contribute recorder (consent first, landmarks only — never video).
  • AboutView     — what SANKET is, honest limits, credits required by CC BY 4.0, licence,
    third-party notices, KP watermark.
NEXT: main_window.py puts them in the left navigation rail.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSpinBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from sanket.core.lexicon import default_lexicon, norm_word
from sanket.core.signs import SignPlayer
from sanket.paths import resource, user_dir
from sanket.ui.avatar_widget import AvatarWidget
from sanket.ui.theme import PALETTES, Settings, tr


class DictionaryView(QWidget):
    def __init__(self, player: SignPlayer) -> None:
        super().__init__()
        self.setObjectName("root")
        self.lex = default_lexicon()
        self.player = player
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        self.search = QLineEdit(placeholderText=tr("search"))
        self.search.textChanged.connect(self._refresh)
        left.addWidget(self.search)
        self.category = QComboBox()
        self.category.addItem(tr("all_categories"), "")
        for c in sorted({e.category for e in self.lex.entries.values()}):
            self.category.addItem(c, c)
        self.category.currentIndexChanged.connect(self._refresh)
        left.addWidget(self.category)
        self.list = QListWidget()
        self.list.setAccessibleName("Sign list")
        self.list.currentItemChanged.connect(lambda cur, _p: cur and self.play(cur.data(Qt.UserRole)))
        left.addWidget(self.list, 1)
        lay.addLayout(left, 2)

        right = QVBoxLayout()
        self.title = QLabel("")
        self.title.setProperty("role", "caption")
        right.addWidget(self.title)
        self.avatar = AvatarWidget()
        right.addWidget(self.avatar, 1)
        row = QHBoxLayout()
        self.play_btn = QPushButton(tr("play"))
        self.play_btn.setProperty("role", "primary")
        self.play_btn.clicked.connect(lambda: self.play(self._gloss))
        row.addWidget(self.play_btn)
        self.slow = QCheckBox("0.5x")
        row.addWidget(self.slow)
        self.report = QPushButton(tr("wrong_sign"))
        self.report.clicked.connect(self._report)
        row.addWidget(self.report)
        right.addLayout(row)
        credit = QLabel(tr("credits"))
        credit.setWordWrap(True)
        credit.setProperty("role", "muted")
        right.addWidget(credit)
        lay.addLayout(right, 3)
        self._gloss = ""
        self._refresh()

    def _refresh(self) -> None:
        q = norm_word(self.search.text())
        cat = self.category.currentData()
        self.list.clear()
        for e in sorted(self.lex.entries.values(), key=lambda e: e.en.lower()):
            if cat and e.category != cat:
                continue
            words = " ".join([e.hi, e.en, *e.synonyms_hi, *e.synonyms_en])
            if q and q not in norm_word(words):
                continue
            item = QListWidgetItem(f"{e.hi}  —  {e.en}")
            item.setData(Qt.UserRole, e.gloss)
            self.list.addItem(item)

    def play(self, gloss: str) -> None:
        if not gloss:
            return
        self._gloss = gloss
        e = self.lex.get(gloss)
        missing = "" if self.player.dict.has(gloss) else "  (avatar data abhi nahi)"
        self.title.setText(f"{e.hi} · {e.en}{missing}")
        self.avatar.play(self.player.timeline([gloss], 0.5 if self.slow.isChecked() else 1.0))

    def _report(self) -> None:
        """Save a local note that this sign looks wrong (user can share it; nothing is uploaded)."""
        if not self._gloss:
            return
        with (user_dir() / "sign_reports.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"gloss": self._gloss, "time": dt.datetime.now().isoformat()}) + "\n")
        self.title.setText(tr("wrong_sign_saved"))


class SettingsView(QWidget):
    changed = Signal()  # main window re-applies theme / restarts camera when settings change
    contributeRequested = Signal(str)  # gloss to record

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.setObjectName("root")
        self.s = settings
        outer = QHBoxLayout(self)
        holder = QWidget()
        holder.setMaximumWidth(860)  # long forms are hard to read edge-to-edge on wide screens
        outer.addWidget(holder, 1)
        outer.addStretch(0)
        form = QFormLayout(holder)
        form.setLabelAlignment(Qt.AlignLeft)
        self.camera = QSpinBox(minimum=0, maximum=9, value=settings.camera_index)
        form.addRow(tr("settings_camera"), self.camera)
        self.stt = QComboBox()
        self.stt.addItem("हिन्दी", "hi")
        self.stt.addItem("English (India)", "en")
        self.stt.setCurrentIndex(0 if settings.stt_language == "hi" else 1)
        form.addRow(tr("settings_mic_lang"), self.stt)
        self.tts = QComboBox()
        self.tts.addItem("हिन्दी", "hi")
        self.tts.addItem("English", "en")
        self.tts.setCurrentIndex(0 if settings.tts_language == "hi" else 1)
        form.addRow(tr("settings_voice"), self.tts)
        self.speed = QComboBox()
        self.speed.addItem("1x", 1.0)
        self.speed.addItem("0.75x", 0.75)
        self.speed.addItem("0.5x", 0.5)
        self.speed.setCurrentIndex([1.0, 0.75, 0.5].index(settings.avatar_speed) if settings.avatar_speed in
                                   (1.0, 0.75, 0.5) else 0)
        form.addRow(tr("settings_avatar_speed"), self.speed)
        self.threshold = QSlider(Qt.Horizontal, minimum=30, maximum=95,
                                 value=int((settings.threshold or 0.6) * 100))
        self.threshold_auto = QCheckBox("Auto (training se)")
        self.threshold_auto.setChecked(settings.threshold is None)
        row = QHBoxLayout()
        row.addWidget(self.threshold, 1)
        row.addWidget(self.threshold_auto)
        form.addRow(tr("settings_threshold"), row)
        self.text_px = QSpinBox(minimum=20, maximum=48, value=settings.text_px, suffix=" px")
        form.addRow(tr("settings_text_size"), self.text_px)
        self.theme = QComboBox()
        for key in PALETTES:
            self.theme.addItem(tr(f"theme_{key}"), key)
        self.theme.setCurrentIndex(list(PALETTES).index(settings.theme) if settings.theme in PALETTES else 0)
        form.addRow(tr("settings_theme"), self.theme)
        save = QPushButton("✓ Save")
        save.setProperty("role", "primary")
        save.clicked.connect(self._save)
        form.addRow(save)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        form.addRow(line)
        self.contrib = QCheckBox(tr("settings_contribute"))
        self.contrib.setChecked(settings.contribute)
        form.addRow(self.contrib)
        self.consent = QCheckBox(tr("contribute_consent"))
        self.consent.setStyleSheet("QCheckBox { font-size: 14px; }")
        form.addRow(self.consent)
        self.word = QComboBox()
        for e in sorted(default_lexicon().entries.values(), key=lambda e: e.en.lower()):
            self.word.addItem(f"{e.hi} — {e.en}", e.gloss)
        self.record = QPushButton(tr("contribute_record"))
        self.record.clicked.connect(lambda: self.contributeRequested.emit(self.word.currentData()))
        row2 = QHBoxLayout()
        row2.addWidget(self.word, 1)
        row2.addWidget(self.record)
        form.addRow(row2)
        self.contrib_status = QLabel("")
        form.addRow(self.contrib_status)
        for w in (self.consent, self.word, self.record):
            w.setEnabled(self.contrib.isChecked())
        self.contrib.toggled.connect(lambda on: [w.setEnabled(on) for w in (self.consent, self.word, self.record)])
        self.record.setEnabled(False)
        self.consent.toggled.connect(lambda on: self.record.setEnabled(on and self.contrib.isChecked()))

    def _save(self) -> None:
        s = self.s
        s.camera_index = self.camera.value()
        s.stt_language = self.stt.currentData()
        s.tts_language = self.tts.currentData()
        s.avatar_speed = float(self.speed.currentData())
        s.threshold = None if self.threshold_auto.isChecked() else self.threshold.value() / 100
        s.text_px = self.text_px.value()
        s.theme = self.theme.currentData()
        s.contribute = self.contrib.isChecked()
        s.save()
        self.changed.emit()


def save_contribution(gloss: str, segment: np.ndarray, take: int) -> str:
    """Store one recorded take as landmarks only (no video) + consent record. Returns the path."""
    folder = user_dir() / "contrib" / gloss
    folder.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = folder / f"{stamp}_take{take}.npz"
    np.savez_compressed(path, frames=segment.astype(np.float32), fps=np.float32(30), gloss=gloss,
                        consent=f"consent given in app on {dt.date.today().isoformat()}")
    return str(path)


class AboutView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("root")
        lay = QVBoxLayout(self)
        title = QLabel("SANKET (संकेत)  ·  v1.0 Preview")
        title.setProperty("role", "caption")
        lay.addWidget(title)
        for key in ("about_text", "credits", "about_licence"):
            lbl = QLabel(tr(key))
            lbl.setWordWrap(True)
            lay.addWidget(lbl)
        notices = resource("THIRD_PARTY_NOTICES.md")
        box = QTextBrowser()
        box.setAccessibleName(tr("third_party"))
        box.setMarkdown(notices.read_text(encoding="utf-8") if notices.exists() else "THIRD_PARTY_NOTICES.md missing")
        lay.addWidget(QLabel(tr("third_party")))
        lay.addWidget(box, 1)
