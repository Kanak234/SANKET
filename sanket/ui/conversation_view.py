"""conversation_view.py — the main screen: two people, one device, both directions.

WHY: this is where SANKET is actually used — at a shop counter, a bank desk, a clinic. One side
     shows the camera for the Deaf/mute signer, the other side the avatar for the hearing speaker;
     the transcript at the bottom keeps both turns readable (UI/UX brief §3).
WHAT (flow):
  A) CameraWorker → Prediction → a chip appears (✓ confident / ? not sure, tap to fix) →
     1.2 s with no new sign (or Enter) → sentence closes → Hindi + English caption → spoken aloud.
  B) Space / button held → microphone → released → Vosk text → glosses → avatar signs them,
     caption chips underline the word being signed; words without a sign show as dashed chips.
  Keyboard: Space hold = speak, Enter = close sentence, Esc = clear, Ctrl+F = flip sides, F12 = debug.
NEXT: hosted by main_window.py; uses core/* for all logic (this file only wires and displays).
"""

from __future__ import annotations

import html

import numpy as np
from PySide6.QtCore import QPointF, Qt, QThreadPool, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from sanket.core.grammar import glosses_to_sentence, is_text_token
from sanket.core.lexicon import default_lexicon
from sanket.core.recognizer import IslRecognizer, Prediction, Recognizer
from sanket.core.signs import SignDictionary, SignPlayer
from sanket.core.stt import SpeechToText
from sanket.core.tts import Speaker
from sanket.ui.avatar_widget import AvatarWidget
from sanket.ui.theme import Palette, Settings, tr
from sanket.ui.workers import CameraWorker, SpeechJob

SENTENCE_GAP_MS = 1200  # PRD FR-A6


class CameraView(QWidget):
    """Shows the mirrored webcam image with the detected hand points drawn on top."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAccessibleName("Camera preview")
        self._img: QImage | None = None
        self._pts: np.ndarray | None = None
        self.signing = False
        self.message = tr("camera_hint")
        self.debug_text = ""
        self.accent = QColor("#0F6E6E")

    def set_frame(self, img: QImage, pts: np.ndarray) -> None:
        self._img, self._pts = img, pts
        self.update()

    def paintEvent(self, _e) -> None:  # noqa: N802
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.fillRect(self.rect(), QColor("#101414"))
        if self._img is None:
            qp.setPen(QColor("#E8E6E1"))
            qp.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self.message)
            return
        pm = QPixmap.fromImage(self._img).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x0, y0 = (self.width() - pm.width()) / 2, (self.height() - pm.height()) / 2
        qp.drawPixmap(int(x0), int(y0), pm)
        if self._pts is not None:  # overlay: hand points in 0-1 image units; mirror x like the preview
            qp.setPen(QPen(self.accent.lighter(150), 4))
            for p in self._pts:
                if np.isfinite(p[0]):
                    qp.drawPoint(QPointF(x0 + (1 - p[0]) * pm.width(), y0 + p[1] * pm.height()))
        if self.signing:  # red dot = "I am reading your sign now"
            qp.setBrush(QColor("#D32F2F"))
            qp.setPen(Qt.NoPen)
            qp.drawEllipse(QPointF(x0 + 22, y0 + 22), 10, 10)
        if self.debug_text:
            qp.setPen(QColor("#FFFF00"))
            qp.drawText(int(x0) + 40, int(y0) + 28, self.debug_text)


class ChipRow(QScrollArea):
    """Horizontal row of word chips (recognised signs, or caption words for the avatar)."""

    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setFixedHeight(64)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        self.box = QHBoxLayout(inner)
        self.box.setContentsMargins(0, 0, 0, 0)
        self.box.addStretch(1)
        self.setWidget(inner)
        self.chips: list[QPushButton] = []

    def clear(self) -> None:
        for c in self.chips:
            c.deleteLater()
        self.chips = []

    def add(self, text: str, kind: str, tooltip: str = "") -> QPushButton:
        b = QPushButton(text)
        b.setProperty("chip", kind)
        b.setToolTip(tooltip)
        b.setAccessibleName(f"{text} {tooltip}".strip())
        self.box.insertWidget(self.box.count() - 1, b)
        self.chips.append(b)
        self.ensureWidgetVisible(b)
        return b

    def highlight(self, index: int, kinds: list[str]) -> None:
        """Mark chip `index` as current (avatar is signing it); others back to their kind."""
        for i, c in enumerate(self.chips):
            c.setProperty("chip", "current" if i == index else kinds[i])
            c.style().unpolish(c)
            c.style().polish(c)


class ConversationView(QWidget):
    def __init__(self, settings: Settings, recognizer: Recognizer | IslRecognizer, signs: SignDictionary) -> None:
        super().__init__()
        self.setObjectName("root")
        self.settings = settings
        self.recognizer = recognizer
        self.lex = default_lexicon()
        self.player = SignPlayer(signs)
        self.stt = SpeechToText(settings.stt_language)
        self.speaker = Speaker(settings.tts_language)
        self.worker: CameraWorker | None = None
        self.pending: list[Prediction] = []  # signs of the sentence being built
        self.last_sentence = None
        self.listening = False
        self.debug = False
        self.flipped = False
        self._build()
        self._sentence_timer = QTimer(self, singleShot=True, interval=SENTENCE_GAP_MS)
        self._sentence_timer.timeout.connect(self.close_sentence)
        if recognizer.demo_mode:
            self.show_notice(tr("demo_mode"))

    # ------------------------------------------------------------------ layout
    def _panel(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        f = QFrame()
        f.setProperty("role", "panel")
        lay = QVBoxLayout(f)
        lay.setContentsMargins(14, 12, 14, 12)
        t = QLabel(title)
        t.setProperty("role", "title")
        lay.addWidget(t)
        return f, lay

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        self.notice = QLabel()
        self.notice.setObjectName("notice")
        self.notice.setWordWrap(True)
        self.notice.hide()
        root.addWidget(self.notice)

        self.panels = QHBoxLayout()
        root.addLayout(self.panels, 5)

        # --- signer side (direction A)
        self.sign_panel, sl = self._panel(tr("signer"))
        self.banner = QLabel()
        self.banner.setObjectName("banner")
        self.banner.hide()
        sl.addWidget(self.banner)
        self.retry_btn = QPushButton(tr("retry"))
        self.retry_btn.hide()
        self.retry_btn.clicked.connect(self.start_camera)
        sl.addWidget(self.retry_btn)
        self.camera = CameraView()
        sl.addWidget(self.camera, 1)
        self.sign_chips = ChipRow()
        sl.addWidget(self.sign_chips)
        row = QHBoxLayout()
        self.sentence_label = QLabel("")
        self.sentence_label.setProperty("role", "caption")
        self.sentence_label.setWordWrap(True)
        self.sentence_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row.addWidget(self.sentence_label, 1)
        self.replay_btn = QPushButton(tr("replay"))
        self.replay_btn.setEnabled(False)
        self.replay_btn.clicked.connect(self.replay)
        row.addWidget(self.replay_btn)
        sl.addLayout(row)

        # --- speaker side (direction B)
        self.speak_panel, kl = self._panel(tr("speaker"))
        self.avatar = AvatarWidget()
        self.avatar.tokenChanged.connect(self._on_token)
        kl.addWidget(self.avatar, 1)
        self.speak_chips = ChipRow()
        kl.addWidget(self.speak_chips)
        self.heard_label = QLabel(tr("speak_hint"))
        self.heard_label.setProperty("role", "caption")
        self.heard_label.setWordWrap(True)
        kl.addWidget(self.heard_label)
        self.speak_btn = QPushButton(tr("hold_to_speak"))
        self.speak_btn.setProperty("role", "primary")
        self.speak_btn.pressed.connect(self.start_listening)
        self.speak_btn.released.connect(self.stop_listening)
        if not self.stt.available:
            self.speak_btn.setEnabled(False)
            self.speak_btn.setToolTip(tr("no_mic"))
        kl.addWidget(self.speak_btn)
        self._speak_kinds: list[str] = []

        self.panels.addWidget(self.sign_panel, 1)
        self.panels.addWidget(self.speak_panel, 1)

        bottom = QHBoxLayout()
        t = QLabel(tr("transcript"))
        t.setProperty("role", "muted")
        bottom.addWidget(t, 1)
        flip = QPushButton(tr("flip"))
        flip.clicked.connect(self.flip)
        bottom.addWidget(flip)
        root.addLayout(bottom)
        self.transcript = QTextBrowser()
        self.transcript.setAccessibleName("Transcript")
        root.addWidget(self.transcript, 2)

    def apply_palette(self, p: Palette) -> None:
        self.avatar.set_palette(p)
        self.camera.accent = QColor(p.accent)

    # ------------------------------------------------------------------ camera / direction A
    def start_camera(self) -> None:
        self.stop_camera()
        self.banner.hide()
        self.retry_btn.hide()
        self.worker = CameraWorker(self.recognizer, self.settings.camera_index)
        self.worker.frameReady.connect(self.camera.set_frame)
        self.worker.signingChanged.connect(self._on_signing)
        self.worker.predicted.connect(self.on_prediction)
        self.worker.timings.connect(self._on_timings)
        self.worker.hint.connect(lambda k: self.show_notice(tr(k)))
        self.worker.failed.connect(self._on_camera_failed)
        self.worker.start()

    def stop_camera(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

    def _on_camera_failed(self, why: str) -> None:
        self.banner.setText("⚠ " + (tr("no_camera") if why == "no_camera" else why))
        self.banner.show()
        self.retry_btn.show()

    def _on_signing(self, active: bool) -> None:
        self.camera.signing = active
        if active:
            self._sentence_timer.stop()  # still signing → don't close the sentence yet
        self.camera.update()

    def _on_timings(self, t: dict) -> None:
        self.camera.debug_text = (f"FPS {t['fps']}  MediaPipe {t['mediapipe_ms']} ms  "
                                  f"model {t['recognizer_ms']} ms") if self.debug else ""

    def on_prediction(self, pred: Prediction) -> None:
        """A sign was read. Confident → ✓ chip; not confident → '?' chip with alternatives."""
        if pred.accepted:
            self.pending.append(pred)
            self._add_sign_chip(len(self.pending) - 1)
            self._sentence_timer.start()
        else:
            chip = self.sign_chips.add("? " + tr("not_understood"), "warn")
            alts = [(g, p) for g, p in pred.top if p > 0.05]
            if alts:  # let the user pick the right word if it is among the guesses
                chip.setToolTip(", ".join(f"{self.lex.get(g).hi} {p:.0%}" for g, p in alts))
                chip.clicked.connect(lambda _=False, pr=pred, c=chip: self._choose_alternative(pr, c))
            else:
                QTimer.singleShot(2500, chip.deleteLater)

    def _add_sign_chip(self, idx: int) -> None:
        pred = self.pending[idx]
        e = self.lex.get(pred.gloss)
        chip = self.sign_chips.add(f"✓ {e.hi}", "ok", f"{e.en} · {pred.prob:.0%}")
        chip.clicked.connect(lambda _=False, i=idx, c=chip: self._chip_menu(i, c))

    def _chip_menu(self, idx: int, chip: QPushButton) -> None:
        """Tap a chip → swap with an alternative or remove it (fixes a wrong sign quickly)."""
        menu = QMenu(self)
        for g, p in self.pending[idx].top[1:]:
            act = menu.addAction(f"{self.lex.get(g).hi} ({self.lex.get(g).en}) {p:.0%}")
            act.triggered.connect(lambda _=False, g=g: self._replace(idx, g, chip))
        menu.addAction("✕ Hataayein").triggered.connect(lambda: self._remove(idx, chip))
        menu.exec(chip.mapToGlobal(chip.rect().bottomLeft()))

    def _replace(self, idx: int, gloss: str, chip: QPushButton) -> None:
        old = self.pending[idx]
        self.pending[idx] = Prediction(gloss, 1.0, old.top, True)
        e = self.lex.get(gloss)
        chip.setText(f"✓ {e.hi}")

    def _remove(self, idx: int, chip: QPushButton) -> None:
        self.pending[idx] = Prediction("", 0.0, [], False)
        chip.hide()

    def _choose_alternative(self, pred: Prediction, chip: QPushButton) -> None:
        menu = QMenu(self)
        for g, p in pred.top:
            act = menu.addAction(f"{self.lex.get(g).hi} ({self.lex.get(g).en}) {p:.0%}")
            act.triggered.connect(lambda _=False, g=g: self._accept_alt(g, chip))
        menu.exec(chip.mapToGlobal(chip.rect().bottomLeft()))

    def _accept_alt(self, gloss: str, chip: QPushButton) -> None:
        chip.deleteLater()
        self.pending.append(Prediction(gloss, 1.0, [(gloss, 1.0)], True))
        self._add_sign_chip(len(self.pending) - 1)
        self._sentence_timer.start()

    def close_sentence(self) -> None:
        """Signer paused (or pressed Enter): turn the collected signs into a spoken sentence."""
        self._sentence_timer.stop()
        glosses = [p.gloss for p in self.pending if p.gloss]
        self.pending = []
        self.sign_chips.clear()
        if not glosses:
            return
        s = glosses_to_sentence(glosses, self.lex)
        self.last_sentence = s
        self.sentence_label.setText(f"{s.hi}\n{s.en}")
        self.replay_btn.setEnabled(True)
        lang = self.speaker.say(s.hi, s.en)
        if lang == "en" and self.speaker.english_fallback_used:
            self.show_notice(tr("hindi_voice_missing"))
        self._log("🤟", s.hi, s.en)

    def replay(self) -> None:
        if self.last_sentence:
            self.speaker.say(self.last_sentence.hi, self.last_sentence.en)

    # ------------------------------------------------------------------ microphone / direction B
    def start_listening(self) -> None:
        if self.listening or not self.stt.available:
            return
        try:
            self.stt.start()
        except Exception as exc:  # no microphone / device busy
            self.heard_label.setText(f"⚠ {tr('no_mic')} ({exc})")
            return
        self.listening = True
        self.heard_label.setText(tr("listening"))

    def stop_listening(self) -> None:
        if not self.listening:
            return
        self.listening = False
        self.heard_label.setText(tr("understanding"))
        job = SpeechJob(self.stt)
        job.signals.done.connect(self.on_heard)
        job.signals.failed.connect(lambda e: self.heard_label.setText(f"⚠ {e}"))
        QThreadPool.globalInstance().start(job)

    def on_heard(self, text: str, glosses: list[str]) -> None:
        """Speech recognised → caption chips + avatar plays the signs in ISL order."""
        if not text:
            self.heard_label.setText(tr("not_understood"))
            return
        self.heard_label.setText(text)
        self.speak_chips.clear()
        self._speak_kinds = []
        for g in glosses:
            if is_text_token(g):
                word = g.split(":", 1)[1]
                self.speak_chips.add(word, "text", "no sign yet")
                self._speak_kinds.append("text")
            else:
                e = self.lex.get(g)
                self.speak_chips.add(e.hi, "ok", e.en)
                self._speak_kinds.append("ok")
        self.avatar.play(self.player.timeline(glosses, self.settings.avatar_speed))
        isl = " ".join(f"[{g.split(':', 1)[1]}]" if is_text_token(g) else g.upper() for g in glosses)
        self._log("🗣", text, f"ISL: {isl}")

    def play_glosses(self, glosses: list[str]) -> None:
        """Public helper (Dictionary screen / tests): sign a gloss list on this avatar."""
        self.on_heard(" ".join(glosses), glosses)

    def _on_token(self, i: int) -> None:
        if self._speak_kinds:
            self.speak_chips.highlight(i, self._speak_kinds)

    # ------------------------------------------------------------------ misc
    def clear(self) -> None:
        self._sentence_timer.stop()
        self.pending = []
        self.sign_chips.clear()
        self.sentence_label.setText("")

    def flush(self) -> None:
        if self.worker:
            self.worker.flush_sign()
        QTimer.singleShot(150, self.close_sentence)  # give the worker a moment to emit the last sign

    def flip(self) -> None:
        """Swap sides so the avatar faces whoever sits on that side of the counter."""
        self.flipped = not self.flipped
        self.panels.removeWidget(self.sign_panel)
        self.panels.removeWidget(self.speak_panel)
        first, second = (self.speak_panel, self.sign_panel) if self.flipped else (self.sign_panel, self.speak_panel)
        self.panels.addWidget(first, 1)
        self.panels.addWidget(second, 1)

    def toggle_debug(self) -> None:
        self.debug = not self.debug
        if not self.debug:
            self.camera.debug_text = ""

    def show_notice(self, text: str) -> None:
        self.notice.setText(text)
        self.notice.show()
        QTimer.singleShot(8000, self.notice.hide)

    def _log(self, icon: str, main: str, sub: str) -> None:
        self.transcript.append(
            f"<p style='margin:6px 0'><b>{icon} {html.escape(main)}</b><br>"
            f"<span style='opacity:.75'>{html.escape(sub)}</span></p>")

    def shutdown(self) -> None:
        self.stop_camera()
        self.speaker.close()
