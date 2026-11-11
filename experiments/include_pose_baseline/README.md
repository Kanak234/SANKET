# Experiment: INCLUDE pose baseline + live webcam demo

Ye **main SANKET app (`sanket/`) se alag** ek tez experiment hai. Main app apna 56.8 GB video pipeline aur Transformer model use karta hai. Ye experiment yeh dekhne ke liye tha ki INCLUDE ke **pehle se nikle pose keypoints** (632 MB, Zenodo) pe ek chhota model kitna accha seekh sakta hai.

| | INCLUDE-50 | INCLUDE (262 words) |
|---|---|---|
| Test top-1 | **97.9%** | **93.8%** |
| Test top-5 | 100% | 99.1% |
| Test videos | 192 | 816 |
| Training time (Colab T4) | ~2 min | ~5 min |

Model: BiGRU (2 layer, bidirectional) + attention pooling, 2.14M params, **random init, from scratch**. ONNX size 8.6 MB.
Trained model: Hugging Face `KANAKq/sanket-isl-include` (GitHub pe model files nahi rakhe).

## Files
| File | Kya hai |
|---|---|
| `train_colab.py` | Colab ka poora training: download → matching → normalize → train → ONNX export + verify |
| `sanket_core.py` | Live logic: 75-point frame, hand A/B assignment, normalization (training ki hubahu copy), segmenter, ONNX predictor |
| `sanket_live.py` | Webcam app: MediaPipe Pose + Hand (Tasks API), OpenCV UI, offline aawaz (pyttsx3) |
| `tests/test_core.py` | Webcam ke bina tests |
| `run.bat` | Windows pe ek click: venv + packages + app |
| `model/` | Colab zip ya HF se aaye model files (gitignored) |
| `HF_MODEL_CARD.md` | Hugging Face model page ka README |

## Chalana (Windows)
1. Model ko `model\` me rakho. Ya to Colab wali `sanket_include_model.zip` yahan unzip karo, ya HF se download karo:
   `hf download KANAKq/sanket-isl-include --local-dir model`
2. `run.bat` chalao. Pehli baar ye packages install karega aur MediaPipe ke 2 `.task` files (~15 MB) download karega. Uske baad poora offline chalega.

Keys: `SPACE` manual record · `M` auto/manual · `S` aawaz · `C` sentence saaf · `Q` band.
Auto mode: haath neeche rakho, phir haath utha ke sign karo, phir haath gira do. Prediction tab aayegi.
Options: `--manual`, `--camera 1`, `--threshold 0.45`, `--raise-k 1.2`, `--no-tts`.

## Verification
**Verified:**
- `python tests/test_core.py` → ALL TESTS PASSED. Live normalization training code se bit-for-bit same hai (20 random videos), hand assignment, segmenter, aur ONNX predictor sab check hue.
- Colab: ONNX vs PyTorch 20/20 same prediction, max diff 3.8e-06.
- MediaPipe 1.2.2 ka Tasks API installed package se inspect karke likha.

**NOT verified:**
- Asli webcam pe accuracy. INCLUDE ke videos alag camera aur alag logon ke hain, isliye domain gap ki wajah se accuracy kam aa sakti hai.
- `--raise-k` ka default, aur laptop pe FPS.

## Main app se farak (abhi merge nahi)
| | Ye experiment | Main app (`sanket/`) |
|---|---|---|
| Landmarks | Pose 33 + Hand 21×2 = 75 (INCLUDE pose layout) | MediaPipe Holistic, 79 points |
| Features | `sanket_core.normalize` | `sanket/core/normalize.py` |
| Model | BiGRU 2.1M | Transformer ~1.9M |
| UI | OpenCV window | PySide6 |

`AGENTS.md` rule 3 ke hisaab se main app me sirf `sanket/core/normalize.py` hi feature implementation hai. Isliye ye experiment main app me **merge nahi kiya gaya**. Ise main pipeline me laana hai ya nahi, ye `docs/OPEN_QUESTIONS.md` me likha hai.

## Data credit
INCLUDE dataset: Sridhar, Ganesan, Kumar, Khapra, ACM MM 2020. AI4Bharat / IIT Madras, CC BY 4.0. Pose keypoints Zenodo record 6674324 se liye gaye hain.
