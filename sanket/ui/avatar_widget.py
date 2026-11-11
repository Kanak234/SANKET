"""avatar_widget.py — draws the 2D signing avatar from landmark frames (direction B output).

WHY 2D drawn from points: the avatar's motion is copied from real INCLUDE recordings (signs.npz),
     so we only need to draw a clear figure at those points — no 3D engine, no rigging, ~5 ms per
     frame on a laptop, and the same drawing code can move to Android later.
WHAT: a QWidget that, 30 times a second, paints the current frame:
        torso + neck + head (from shoulders / face points), arms (shoulder → elbow → wrist as thick
        rounded strokes), hands (palm polygon + 5 finger chains — the most important part, drawn
        largest), eyebrows and lips (ISL uses them for questions and mouthing).
      The figure faces the viewer exactly like a person standing opposite (the INCLUDE videos
      are not mirrored), so left/right hands are correct for the person watching.
NEXT: ConversationView/DictionaryView call play(frames) with SignPlayer.timeline(...) output and
      listen to `tokenChanged` to underline the caption word currently being signed.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from sanket.core.normalize import FACE, LEFT_HAND, RIGHT_HAND
from sanket.core.signs import Frame, default_rest_pose
from sanket.ui.theme import PALETTES, Palette

P = 42  # start of pose block in the 79-point array
FINGERS = [(0, 1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
PALM = (0, 1, 5, 9, 13, 17)
LIPS = (10, 14, 12, 15, 11, 17, 13, 16)  # outer-lip points in drawing order (see normalize.FACE_IDX)
VIEW = QRectF(-2.3, -1.9, 4.6, 4.2)  # region of normalised space shown (shoulder width = 1 unit)


def _ok(p: np.ndarray) -> bool:
    return bool(np.isfinite(p[:2]).all())


class AvatarWidget(QWidget):
    tokenChanged = Signal(int)  # index of the word being signed (for caption highlighting)
    finished = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 300)
        self.setAccessibleName("Signing avatar")
        self.palette_: Palette = PALETTES["light"]
        self._frames: list[Frame] = []
        self._i = 0
        self._current = default_rest_pose()
        self._token = -1
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    # ---------------------------------------------------------------- playback control
    def set_palette(self, palette: Palette) -> None:
        self.palette_ = palette
        self.update()

    def play(self, frames: list[Frame], fps: float = 30.0) -> None:
        """Start playing a timeline from SignPlayer (replaces anything playing)."""
        self._frames, self._i, self._token = frames, 0, -1
        if frames:
            self._timer.start(int(1000 / fps))

    def stop(self) -> None:
        self._timer.stop()
        self._frames = []

    def show_pose(self, points: np.ndarray) -> None:
        """Draw a single still pose (used for the rest pose and in tests)."""
        self._current = points
        self.update()

    def _tick(self) -> None:
        if self._i >= len(self._frames):
            self._timer.stop()
            self.finished.emit()
            return
        f = self._frames[self._i]
        self._current = f.points
        if f.token_index != self._token:
            self._token = f.token_index
            self.tokenChanged.emit(f.token_index)
        self._i += 1
        self.update()

    # ---------------------------------------------------------------- drawing
    def _mapper(self):
        """Function mapping normalised (x, y) → widget pixels, keeping aspect ratio, centred."""
        w, h = self.width(), self.height()
        s = min(w / VIEW.width(), h / VIEW.height())
        ox = (w - VIEW.width() * s) / 2 - VIEW.left() * s
        oy = (h - VIEW.height() * s) / 2 - VIEW.top() * s
        return (lambda p: QPointF(ox + float(p[0]) * s, oy + float(p[1]) * s)), s

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        pal = self.palette_
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.fillRect(self.rect(), QColor(pal.panel))
        pts = self._current
        m, s = self._mapper()
        body = QColor(pal.avatar_body)
        line = QColor(pal.avatar_line)
        hand_col = QColor(pal.avatar_hand)

        ls, rs = pts[P + 11], pts[P + 12]
        if _ok(ls) and _ok(rs):
            # Torso: a rounded trapezoid hanging from the shoulders.
            width = abs(ls[0] - rs[0])
            path = QPainterPath()
            path.moveTo(m((rs[0] - 0.08 * width, rs[1] - 0.05)))
            path.lineTo(m((ls[0] + 0.08 * width, ls[1] - 0.05)))
            path.lineTo(m((ls[0] - 0.05 * width, ls[1] + 2.2)))
            path.lineTo(m((rs[0] + 0.05 * width, rs[1] + 2.2)))
            path.closeSubpath()
            qp.setPen(Qt.NoPen)
            qp.setBrush(QBrush(body))
            qp.drawPath(path)
            # Neck
            mid = (ls[:2] + rs[:2]) / 2
            qp.setPen(QPen(body, 0.22 * s, Qt.SolidLine, Qt.RoundCap))
            qp.drawLine(m(mid), m((mid[0], mid[1] - 0.45)))
            self._head(qp, m, s, pts, mid, body, line)

        # Arms: shoulder → elbow → wrist (hand wrist preferred: it is more precise than pose wrist)
        for sh, el, wr, hand in ((11, 13, 15, LEFT_HAND), (12, 14, 16, RIGHT_HAND)):
            chain = [pts[P + sh], pts[P + el]]
            hw = pts[hand][0]
            chain.append(hw if _ok(hw) else pts[P + wr])
            chain = [c for c in chain if _ok(c)]
            if len(chain) >= 2:
                qp.setPen(QPen(body.darker(115), 0.2 * s, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                qp.drawPolyline(QPolygonF([m(c) for c in chain]))

        for hand in (LEFT_HAND, RIGHT_HAND):
            self._hand(qp, m, s, pts[hand], hand_col, line)
        qp.end()

    def _head(self, qp, m, s, pts, mid, body, line) -> None:
        """Head ellipse centred between nose and shoulders' line + brows, eyes and lips."""
        face = pts[FACE]
        nose = face[18] if _ok(face[18]) else pts[P + 0]
        centre = nose[:2] if _ok(nose) else np.array([mid[0], mid[1] - 0.95])
        qp.setPen(QPen(line, 0.03 * s))
        qp.setBrush(QBrush(body.lighter(108)))
        qp.drawEllipse(m(centre), 0.34 * s, 0.44 * s)
        qp.setPen(QPen(line, 0.045 * s, Qt.SolidLine, Qt.RoundCap))
        for brow in (face[0:5], face[5:10]):  # eyebrows (raised = question)
            ok = [b for b in brow if _ok(b)]
            if len(ok) >= 2:
                qp.drawPolyline(QPolygonF([m(b) for b in ok]))
                eye = np.mean([b[:2] for b in ok], axis=0) + (0, 0.09)
                qp.setBrush(QBrush(line))
                qp.drawEllipse(m(eye), 0.025 * s, 0.025 * s)
        lips = [face[i] for i in LIPS if _ok(face[i])]
        if len(lips) >= 4:
            qp.setBrush(QBrush(QColor("#B4534B")))
            qp.setPen(QPen(line, 0.02 * s))
            qp.drawPolygon(QPolygonF([m(p_) for p_ in lips]))

    def _hand(self, qp, m, s, h: np.ndarray, col: QColor, line: QColor) -> None:
        """Palm polygon + finger chains; hands are drawn biggest/brightest — they carry the sign."""
        if np.isfinite(h[:, 0]).sum() < 10:  # hand not visible → nothing to draw
            return
        palm = [h[i] for i in PALM if _ok(h[i])]
        qp.setPen(QPen(col.darker(130), 0.02 * s))
        qp.setBrush(QBrush(col))
        if len(palm) >= 3:
            qp.drawPolygon(QPolygonF([m(p_) for p_ in palm]))
        qp.setPen(QPen(col, 0.07 * s, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        for finger in FINGERS:
            chain = [h[i] for i in finger if _ok(h[i])]
            if len(chain) >= 2:
                qp.drawPolyline(QPolygonF([m(c) for c in chain]))
        qp.setPen(Qt.NoPen)
        qp.setBrush(QBrush(line))
        for tip in (4, 8, 12, 16, 20):  # dark fingertip dots make handshapes easy to read
            if _ok(h[tip]):
                qp.drawEllipse(m(h[tip]), 0.03 * s, 0.03 * s)
