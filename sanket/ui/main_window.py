"""main_window.py — the SANKET window: header, left navigation, and the four screens.

WHY: one place that owns the shared objects (settings, recognizer, sign dictionary) and the
     global keyboard shortcuts, so each screen stays simple.
WHAT: header "SANKET ● Offline · Preview · 263 words [KP logo]", navigation rail
      (Baatcheet / Shabdkosh / Settings / About), a QStackedWidget with the screens, theme
      application, and the global keys (Space hold = speak, Enter, Esc, Ctrl+F, F12).
NEXT: `run()` is called by sanket/__main__.py.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from sanket.core.recognizer import load_recognizer
from sanket.core.signs import SignDictionary, SignPlayer
from sanket.paths import resource
from sanket.ui.conversation_view import ConversationView
from sanket.ui.other_views import AboutView, DictionaryView, SettingsView, save_contribution
from sanket.ui.theme import PALETTES, Settings, qss, tr

LOGO = resource("assets", "logo-mark.svg")
ICON = resource("assets", "icon.png")


class SpaceToTalk(QObject):
    """Global Space key = push-to-talk (ignored while typing in a text box; key auto-repeat ignored)."""

    def __init__(self, view: ConversationView, window: QMainWindow) -> None:
        super().__init__()
        self.view, self.window = view, window

    def eventFilter(self, obj, ev):  # noqa: N802
        if ev.type() in (QEvent.KeyPress, QEvent.KeyRelease) and ev.key() == Qt.Key_Space:
            if isinstance(QApplication.focusWidget(), QLineEdit) or ev.isAutoRepeat():
                return False
            if self.window.stack.currentWidget() is not self.view:
                return False
            (self.view.start_listening if ev.type() == QEvent.KeyPress else self.view.stop_listening)()
            return True
        return False


class MainWindow(QMainWindow):
    def __init__(self, start_camera: bool = True) -> None:
        super().__init__()
        self.setWindowTitle("SANKET — ISL ⇄ Speech (Preview)")
        self.resize(1360, 820)
        self.settings = Settings.load()
        self.recognizer = load_recognizer(threshold=self.settings.threshold)
        self.signs = SignDictionary()
        self.player = SignPlayer(self.signs)

        root = QWidget(objectName="root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._header())
        body = QHBoxLayout()
        outer.addLayout(body, 1)

        self.conversation = ConversationView(self.settings, self.recognizer, self.signs)
        self.dictionary = DictionaryView(self.player)
        self.settings_view = SettingsView(self.settings)
        self.about = AboutView()
        self.stack = QStackedWidget()
        rail = QVBoxLayout()
        rail.setContentsMargins(10, 10, 6, 10)
        group = QButtonGroup(self)
        for i, (key, view) in enumerate((("nav_conversation", self.conversation),
                                          ("nav_dictionary", self.dictionary),
                                          ("nav_settings", self.settings_view),
                                          ("nav_about", self.about))):
            self.stack.addWidget(view)
            b = QPushButton(tr(key))
            b.setProperty("role", "nav")
            b.setCheckable(True)
            b.setChecked(i == 0)
            group.addButton(b, i)
            rail.addWidget(b)
        group.idClicked.connect(self.stack.setCurrentIndex)
        rail.addStretch(1)
        body.addLayout(rail)
        body.addWidget(self.stack, 1)
        self.setCentralWidget(root)

        self.settings_view.changed.connect(self._settings_changed)
        self.settings_view.contributeRequested.connect(self._record_contribution)
        self._shortcuts()
        self._apply_theme()
        self._space = SpaceToTalk(self.conversation, self)
        QApplication.instance().installEventFilter(self._space)
        if start_camera:
            QTimer.singleShot(300, self.conversation.start_camera)  # let the window appear first

    def _header(self) -> QWidget:
        bar = QWidget()
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 10, 16, 6)
        t = QLabel("SANKET")
        t.setProperty("role", "title")
        lay.addWidget(t)
        badge = QLabel(f"●  {tr('offline')}   ·   {tr('preview')}")
        badge.setProperty("role", "muted")
        lay.addWidget(badge)
        lay.addStretch(1)
        if LOGO.exists():  # Kanak's KP orbit watermark (required on every project)
            from PySide6.QtSvgWidgets import QSvgWidget

            logo = QSvgWidget(str(LOGO))
            logo.setFixedSize(28, 28)
            logo.setToolTip("Kanak Prabhakar")
            lay.addWidget(logo)
        else:
            k = QLabel("KP")
            k.setProperty("role", "title")
            k.setToolTip("assets/logo-mark.svg missing")
            lay.addWidget(k)
        return bar

    def _shortcuts(self) -> None:
        conv = self.conversation
        for keys, fn in (("Return", conv.flush), ("Enter", conv.flush), ("Escape", conv.clear),
                         ("Ctrl+F", conv.flip), ("F12", conv.toggle_debug)):
            QShortcut(QKeySequence(keys), self, activated=fn)

    def _apply_theme(self) -> None:
        p = PALETTES.get(self.settings.theme, PALETTES["light"])
        self.setStyleSheet(qss(p, self.settings.text_px))
        self.conversation.apply_palette(p)
        self.dictionary.avatar.set_palette(p)

    def _settings_changed(self) -> None:
        self._apply_theme()
        if self.settings.threshold is not None:
            self.recognizer.threshold = self.settings.threshold
        self.conversation.stt.language = self.settings.stt_language
        self.conversation.start_camera()  # camera index may have changed

    def _record_contribution(self, gloss: str) -> None:
        """Contribute mode: go to the camera screen and save the next 5 signs as landmark files."""
        self.stack.setCurrentWidget(self.conversation)
        worker = self.conversation.worker
        if worker is None:
            return
        worker.recognize = False
        takes = {"n": 0}

        def on_segment(seg):
            takes["n"] += 1
            save_contribution(gloss, seg, takes["n"])
            self.conversation.show_notice(f"{tr('contribute_saved')} {takes['n']}/5")
            if takes["n"] >= 5:
                worker.segmentReady.disconnect(on_segment)
                worker.recognize = True

        worker.segmentReady.connect(on_segment)
        self.conversation.show_notice(f"● {gloss}: 5 baar sign karein")

    def closeEvent(self, ev) -> None:  # noqa: N802
        self.conversation.shutdown()
        super().closeEvent(ev)


def run() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("SANKET")
    app.setOrganizationName("Kanak Prabhakar")
    if ICON.exists():
        app.setWindowIcon(QIcon(str(ICON)))
    win = MainWindow()
    win.show()
    return app.exec()
