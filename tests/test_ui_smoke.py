"""UI smoke test (offscreen, no camera/mic): every screen builds, both conversation directions
render, keyboard helpers work. Catches wiring mistakes before a release build."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


@pytest.fixture()
def window(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from sanket.ui.main_window import MainWindow

    w = MainWindow(start_camera=False)
    w.show()
    yield app, w
    w.conversation.shutdown()
    w.close()


def test_conversation_both_directions(window):
    app, w = window
    from sanket.core.recognizer import Prediction

    c = w.conversation
    c.on_prediction(Prediction("i", 0.9, [("i", 0.9)], True))
    c.on_prediction(Prediction("happy", 0.8, [("happy", 0.8)], True))
    c.close_sentence()
    assert "मैं ख़ुश हूँ" in c.sentence_label.text()
    c.on_prediction(Prediction("hello", 0.2, [("hello", 0.2), ("good", 0.1)], False))  # low confidence
    assert any("?" in chip.text() for chip in c.sign_chips.chips)
    c.on_heard("नमस्ते रमेश", ["hello", "TEXT:रमेश"])
    assert [ch.text() for ch in c.speak_chips.chips] == ["नमस्ते", "रमेश"]
    c.flip()
    c.flip()
    c.clear()
    app.processEvents()
    assert not w.grab().isNull()


def test_all_screens_render(window):
    app, w = window
    for i in range(w.stack.count()):
        w.stack.setCurrentIndex(i)
        app.processEvents()
        assert not w.grab().isNull()
    w.dictionary.search.setText("doctor")
    assert w.dictionary.list.count() >= 1
    w.dictionary.search.setText("डॉक्टर")
    assert w.dictionary.list.count() >= 1


def test_settings_roundtrip(window, tmp_path):
    _app, w = window
    w.settings_view.theme.setCurrentIndex(1)
    w.settings_view._save()
    from sanket.ui.theme import Settings

    assert Settings.load().theme == "dark"
