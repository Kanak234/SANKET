"""lexicon.py — the word list that connects ISL signs to Hindi and English.

WHY: every part of SANKET talks about the same 263 signs (the INCLUDE vocabulary). The model
     outputs a class number, the avatar needs a sign name, captions need Hindi and English.
     models/lexicon.json is the single table that links all of these.
WHAT: loads lexicon.json into `Lexicon`, giving: class index → gloss, gloss → Hindi/English,
      and a search index from Hindi/English words (incl. synonyms) back to glosses.
NEXT: recognizer.py uses `labels` to name its output; grammar.py uses the word index to turn
      speech text into glosses; the Dictionary screen lists `entries`.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sanket.paths import resource

NUKTA = "़"  # the dot under ज़/फ़/ड़ — speech-to-text sometimes drops it, so we ignore it


def norm_word(text: str) -> str:
    """Normalise a word/phrase for matching: lowercase, no nukta, chandrabindu→anusvara, single spaces.

    WHY: Vosk may write "जमीन" where the lexicon says "ज़मीन". Matching on a normalised form
         makes both spellings find the same sign.
    """
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace(NUKTA, "").replace("ँ", "ं")  # ँ → ं
    text = text.replace("।", " ").replace("॥", " ")  # Hindi full stops (danda) are punctuation too
    text = re.sub(r"[^\wऀ-ॿ\s-]", " ", text)  # drop punctuation, keep Devanagari
    return re.sub(r"\s+", " ", text.replace("-", " ")).strip()


# When two signs share a Hindi word, this decides which sign the word maps to by default.
# (grammar.py adds context rules on top, e.g. "कल" + past tense → YESTERDAY.)
PREFERRED = {
    "कल": "tomorrow",
    "छोटा": "smalllittle",
    "लंबा": "long",
    "वह": "he",
    "वो": "he",
    "बच्चा": "child",
    "प्रसन्न": "happy",
    "महिला": "woman",
    "पुरुष": "man",
    "बढ़िया": "good",
    "बढिया": "good",
    "स्त्री": "woman",
    "ठंडा": "cold",
    "you": "you",
}


@dataclass(frozen=True)
class Entry:
    """One sign: its gloss name plus how to say it in Hindi and English."""

    gloss: str
    index: int  # class number in the recognizer's output
    hi: str
    en: str
    pos: str  # noun / adj / pronoun / greeting / time / verb — used by grammar rules
    category: str
    synonyms_hi: tuple[str, ...]
    synonyms_en: tuple[str, ...]


class Lexicon:
    """In-memory view of models/lexicon.json."""

    def __init__(self, path: Path | str | None = None) -> None:
        path = Path(path) if path else resource("models", "lexicon.json")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.entries: dict[str, Entry] = {}
        for gloss, e in data["entries"].items():
            self.entries[gloss] = Entry(
                gloss=gloss,
                index=int(e["index"]),
                hi=e["hi"],
                en=e["en"],
                pos=e["pos"],
                category=e["category"],
                synonyms_hi=tuple(e.get("synonyms_hi", [])),
                synonyms_en=tuple(e.get("synonyms_en", [])),
            )
        # labels[i] = gloss of model output class i (order MUST match training's label map)
        self.labels: list[str] = [g for g, _ in sorted(self.entries.items(), key=lambda kv: kv[1].index)]
        self.word_index: dict[str, str] = self._build_index()
        self.max_phrase_words = max(len(k.split()) for k in self.word_index)

    def _build_index(self) -> dict[str, str]:
        """Map every normalised Hindi/English word or phrase → gloss.

        Order of trust: PREFERRED overrides > primary words > synonyms. First writer wins
        among equals, so the table is deterministic.
        """
        index: dict[str, str] = {}
        for e in self.entries.values():  # primary renderings first
            for w in (e.hi, e.en, e.gloss):
                index.setdefault(norm_word(w), e.gloss)
        for e in self.entries.values():  # then synonyms
            for w in (*e.synonyms_hi, *e.synonyms_en):
                index.setdefault(norm_word(w), e.gloss)
        for w, g in PREFERRED.items():
            index[norm_word(w)] = g
        return index

    def get(self, gloss: str) -> Entry | None:
        return self.entries.get(gloss)

    def __len__(self) -> int:
        return len(self.entries)


@lru_cache(maxsize=1)
def default_lexicon() -> Lexicon:
    """Shared instance for the app (the file never changes while SANKET runs)."""
    return Lexicon()
