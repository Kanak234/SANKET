"""stt.py — offline speech-to-text for the hearing person (direction B, first step).

WHY: the hearing person just talks. SANKET must turn that speech into text WITHOUT internet.
     Vosk (by Alpha Cephei) is a small offline recogniser with a Hindi model of ~40 MB and an
     Android SDK, so the same approach works on the phone version later.
     It is used exactly as shipped — SANKET does not train or fine-tune it.
WHAT: push-to-talk. start() opens the microphone, stop() closes it and returns the text.
      Recording is capped at 15 s (PRD FR-B1). Audio is only held in memory, never saved.
NEXT: the text goes to grammar.text_to_glosses() and then to the avatar.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path

from sanket.paths import resource

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # Vosk small models are trained on 16 kHz mono audio
MAX_SECONDS = 15
MODEL_DIRS = {"hi": resource("models", "vosk", "hi"), "en": resource("models", "vosk", "en")}


class SpeechToText:
    """One microphone session at a time. Thread-safe enough for a push-to-talk button."""

    def __init__(self, language: str = "hi", model_dir: Path | str | None = None) -> None:
        self.language = language
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIRS[language]
        self._model = None  # loaded lazily on first use (takes ~1 s), not at app start
        self._audio: queue.Queue[bytes] = queue.Queue()
        self._stream = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        """False if the Vosk model folder is missing (UI then disables the Speak button)."""
        return (self.model_dir / "am").exists() or (self.model_dir / "conf").exists()

    def _ensure_model(self):
        if self._model is None:
            from vosk import Model, SetLogLevel

            SetLogLevel(-1)  # silence Kaldi's console spam
            self._model = Model(str(self.model_dir))
        return self._model

    def start(self) -> None:
        """Open the microphone and start buffering audio (button pressed / Space held)."""
        import sounddevice as sd

        with self._lock:
            if self._stream is not None:
                return
            self._ensure_model()
            self._audio = queue.Queue(maxsize=int(MAX_SECONDS * SAMPLE_RATE / 4000) + 4)

            def on_audio(indata, frames, time_info, status):  # runs on the audio thread
                try:
                    self._audio.put_nowait(bytes(indata))
                except queue.Full:
                    pass  # 15 s cap reached — extra audio is ignored (PRD FR-B1)

            self._stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE, blocksize=4000, dtype="int16", channels=1, callback=on_audio
            )
            self._stream.start()

    def stop(self) -> str:
        """Close the microphone and return the recognised text ("" if nothing understood)."""
        with self._lock:
            if self._stream is None:
                return ""
            self._stream.stop()
            self._stream.close()
            self._stream = None
        chunks = []
        while not self._audio.empty():
            chunks.append(self._audio.get_nowait())
        return self.transcribe_pcm(b"".join(chunks))

    def transcribe_pcm(self, pcm16: bytes) -> str:
        """Recognise raw 16 kHz mono int16 audio. Also used by the self-test with silence."""
        from vosk import KaldiRecognizer

        rec = KaldiRecognizer(self._ensure_model(), SAMPLE_RATE)
        rec.AcceptWaveform(pcm16)
        return json.loads(rec.FinalResult()).get("text", "").strip()
