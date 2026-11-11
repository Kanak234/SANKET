"""tts.py — speaks the recognised sentence aloud (direction A, last step).

WHY: the hearing person should HEAR what the Deaf/mute person signed, not only read it.
     Windows already ships offline voices (SAPI). Using them costs 0 MB and needs no internet.
WHAT: Speaker.say(hindi, english) speaks Hindi if a Hindi voice is installed (e.g. "Microsoft
      Kalpana/Hemant"), otherwise English, and reports the fallback once so the UI can show
      how to add the Hindi speech pack (PRD FR-A8). Speech runs on a background thread so the
      camera view never freezes.
NEXT: called by the conversation screen after grammar.glosses_to_sentence().
"""

from __future__ import annotations

import logging
import queue
import threading

log = logging.getLogger(__name__)

HINDI_HINTS = ("hindi", "hi-in", "hi_in", "kalpana", "hemant", "swara", "madhur")


class Speaker:
    def __init__(self, prefer: str = "hi") -> None:
        self.prefer = prefer
        self.hindi_available = False
        self.english_fallback_used = False
        self._jobs: queue.Queue[str | None] = queue.Queue()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sanket-tts", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)

    def _run(self) -> None:
        """Owns the pyttsx3 engine for its whole life (SAPI must stay on one thread)."""
        try:
            import pyttsx3

            engine = pyttsx3.init()
        except Exception as exc:  # no speech engine at all (rare; e.g. stripped-down Windows)
            log.warning("text-to-speech unavailable: %s", exc)
            self._engine_ok = False
            self._ready.set()
            return
        self._engine_ok = True
        hindi_id = None
        for v in engine.getProperty("voices"):
            text = f"{v.id} {v.name} {getattr(v, 'languages', '')}".lower()
            if any(h in text for h in HINDI_HINTS):
                hindi_id = v.id
                break
        self.hindi_available = hindi_id is not None
        if hindi_id and self.prefer == "hi":
            engine.setProperty("voice", hindi_id)
        engine.setProperty("rate", 165)  # a little slower than default — clearer for listeners
        self._ready.set()
        while True:
            text = self._jobs.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()

    def say(self, hindi: str, english: str) -> str:
        """Queue a sentence. Returns which language will be spoken ("hi", "en" or "")."""
        if not getattr(self, "_engine_ok", False):
            return ""
        if self.prefer == "hi" and self.hindi_available and hindi:
            self._jobs.put(hindi)
            return "hi"
        if self.prefer == "hi" and not self.hindi_available:
            self.english_fallback_used = True  # UI shows the one-time "add Hindi voice" notice
        self._jobs.put(english)
        return "en"

    def close(self) -> None:
        self._jobs.put(None)
