# sanket.spec — PyInstaller recipe that turns SANKET into SANKET\SANKET.exe (one folder, offline).
#
# WHY "onedir" (a folder, not one exe): starts faster (nothing to unpack each launch), and keeps
#      the Qt libraries as separate files, which the LGPL licence of Qt expects.
# WHAT: bundles the runtime only — sanket/ code, models/ (recognizer.onnx, signs.npz, lexicon.json,
#      MediaPipe .task, Vosk Hindi/English), assets/, i18n strings, LICENSE and notices.
#      PyTorch and the training code are excluded (they are never needed by the app).
# NEXT: packaging/build_msix.ps1 wraps dist/SANKET into an MSIX for the Microsoft Store.
# Run (Windows):  uv run pyinstaller packaging/sanket.spec --noconfirm

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).parent  # noqa: F821 (SPECPATH is provided by PyInstaller)

datas = [
    (str(ROOT / "models"), "models"),
    (str(ROOT / "assets"), "assets"),
    (str(ROOT / "sanket" / "i18n"), "sanket/i18n"),
    (str(ROOT / "LICENSE"), "."),
    (str(ROOT / "THIRD_PARTY_NOTICES.md"), "."),
]
if (ROOT / "third_party").exists():
    datas.append((str(ROOT / "third_party"), "third_party"))
datas += collect_data_files("mediapipe")

binaries = collect_dynamic_libs("vosk") + collect_dynamic_libs("onnxruntime") + collect_dynamic_libs("sounddevice")

a = Analysis(  # noqa: F821
    [str(ROOT / "sanket" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=["pyttsx3.drivers", "pyttsx3.drivers.sapi5", "comtypes.stream", "PySide6.QtSvgWidgets"],
    excludes=["torch", "torchvision", "training", "tests", "matplotlib", "tkinter", "scipy", "sklearn"],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SANKET",
    icon=str(ROOT / "assets" / "icon.ico"),
    console=False,  # windowed app; use `SANKET.exe --selftest` from a terminal to see self-test output
    version=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="SANKET")  # noqa: F821
