"""make_icons.py — draws SANKET's app icon (open palm + speech arcs) at every size the Store needs.

WHY: Microsoft Store / MSIX requires square logos (44, 50, 150 px) and a wide tile; drawing them
     from code keeps them sharp at each size and avoids any copyrighted artwork.
WHAT: assets/icon.png (256) and assets/store/*.png. Teal (#0F6E6E) palm on a warm white tile,
      with two arcs = "sign becomes voice".
NEXT: packaging/AppxManifest.xml references assets/store/*; the app window uses assets/icon.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPainterPath, QPen

REPO = Path(__file__).resolve().parent.parent
TEAL, BG, WHITE = QColor("#0F6E6E"), QColor("#F6F4EF"), QColor("#FFFFFF")


def draw(size_w: int, size_h: int, tile: bool = True) -> QImage:
    img = QImage(size_w, size_h, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    qp = QPainter(img)
    qp.setRenderHint(QPainter.Antialiasing)
    s = min(size_w, size_h)
    ox, oy = (size_w - s) / 2, (size_h - s) / 2
    if tile:
        qp.setPen(Qt.NoPen)
        qp.setBrush(BG)
        qp.drawRoundedRect(QRectF(0, 0, size_w, size_h), s * 0.18, s * 0.18)

    def p(x, y):  # icon drawn in a 100x100 box
        return QPointF(ox + x * s / 100, oy + y * s / 100)

    # palm
    palm = QPainterPath()
    palm.moveTo(p(26, 88))
    palm.cubicTo(p(18, 72), p(18, 56), p(24, 50))
    palm.lineTo(p(60, 50))
    palm.cubicTo(p(64, 62), p(62, 78), p(54, 88))
    palm.closeSubpath()
    qp.setPen(Qt.NoPen)
    qp.setBrush(TEAL)
    qp.drawPath(palm)
    pen = QPen(TEAL, s * 0.085, Qt.SolidLine, Qt.RoundCap)
    qp.setPen(pen)
    for x, top in ((29, 22), (38, 14), (47, 16), (56, 26)):  # four fingers, spread
        qp.drawLine(p(x, 52), p(x - 1, top))
    qp.drawLine(p(24, 62), p(12, 44))  # thumb
    # speech arcs (sign → voice)
    arc = QPen(TEAL, s * 0.05, Qt.SolidLine, Qt.RoundCap)
    qp.setPen(arc)
    for r in (14, 24):
        qp.drawArc(QRectF(p(66 - r, 40 - r), p(66 + r, 40 + r)), -50 * 16, 100 * 16)
    qp.end()
    return img


def main() -> None:
    _app = QGuiApplication.instance() or QGuiApplication(sys.argv)
    store = REPO / "assets" / "store"
    store.mkdir(parents=True, exist_ok=True)
    draw(256, 256).save(str(REPO / "assets" / "icon.png"))
    for name, (w, h) in {"Square44x44Logo": (44, 44), "Square150x150Logo": (150, 150), "StoreLogo": (50, 50),
                         "Wide310x150Logo": (310, 150), "Square310x310Logo": (310, 310),
                         "StoreListing300": (300, 300)}.items():
        draw(w, h).save(str(store / f"{name}.png"))
    # .ico for the PyInstaller exe
    draw(256, 256).save(str(REPO / "assets" / "icon.ico"))
    print("icons written to assets/ and assets/store/")


if __name__ == "__main__":
    main()
