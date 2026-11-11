"""grammar.py — converts between ISL sign order (glosses) and Hindi/English sentences.

WHY: ISL is its own language. It has no words for "is/the/to", keeps the verb/adjective at the
     end (Subject-Object-Verb, like Hindi), and puts question words last. So we cannot just
     map word-for-word; a few rules make the output read naturally in both directions.
WHAT (v1, rule-based — no AI model, fully predictable):
  A) glosses_to_sentence(["i", "happy"])  → ("मैं ख़ुश हूँ", "I am happy.")
  B) text_to_glosses("मैं कल बाज़ार गया था") → ["i", "yesterday", "market", "TEXT:गया"]
     Tokens that have no sign come back as "TEXT:<word>" so the UI shows them as captions
     instead of silently dropping them (PRD FR-B5).
NEXT: (A) feeds captions + text-to-speech after the signer finishes a sentence;
      (B) feeds the avatar player after the hearing person speaks.
"""

from __future__ import annotations

from dataclasses import dataclass

from sanket.core.lexicon import Lexicon, default_lexicon, norm_word

TEXT_PREFIX = "TEXT:"  # marks a word that has no sign yet

# Words ISL does not sign (helping verbs, case markers, articles). Dropped in direction B.
STOPWORDS = {
    norm_word(w)
    for w in (
        "है हैं हूँ हूं हो था थी थे का की के को ने से में पर तो भी ही और या एक "
        "a an the is am are was were be been to of in on at and or do does did please "
        "जी हाँ हां ना न"
    ).split()
}
# Question words: ISL places them at the END of the sentence.
WH_WORDS = {norm_word(w) for w in "क्या कहाँ कहां कब कौन क्यों कैसे कितना कितने कितनी what where when who why how".split()}
PAST_MARKERS = {norm_word(w) for w in "था थी थे गया गई गए गये किया की आया आई आए yesterday was were did".split()}
TIME_DEPENDENT = {"tomorrow": "yesterday"}  # "कल" means both — past tense decides

# Copula ("is/am/are") chosen from the subject pronoun, Hindi and English.
COPULA = {
    "i": ("हूँ", "am"),
    "you": ("हैं", "are"),
    "youplural": ("हैं", "are"),
    "we": ("हैं", "are"),
    "they": ("हैं", "are"),
    "he": ("है", "is"),
    "she": ("है", "is"),
    "it": ("है", "is"),
}


@dataclass
class Sentence:
    """Result of direction A: what the captions show and what TTS speaks."""

    hi: str
    en: str
    glosses: list[str]


def glosses_to_sentence(glosses: list[str], lex: Lexicon | None = None) -> Sentence:
    """Direction A — recognised signs → Hindi + English sentence.

    RULES:
      1. greetings alone are said as-is ("नमस्ते", "Thank you.");
      2. a pronoun subject + adjective at the end gets "हूँ/है/हैं" and "am/is/are";
      3. otherwise words are joined in ISL order, which is already Hindi order (SOV);
      4. "how are you" ends with "?".
    """
    lex = lex or default_lexicon()
    words = [g for g in glosses if lex.get(g)]
    if not words:
        return Sentence("", "", [])
    entries = [lex.get(g) for g in words]
    hi_parts = [e.hi for e in entries]
    en_parts = [e.en for e in entries]

    subject = words[0] if entries[0].pos == "pronoun" else None
    last = entries[-1]
    if len(words) >= 2 and last.pos == "adj":  # rule 2: "I happy" → "I am happy"
        hi_cop, en_cop = COPULA.get(subject, ("है", "is"))
        hi_parts.append(hi_cop)
        en_parts.insert(len(en_parts) - 1, en_cop)

    is_question = "howareyou" in words
    hi = " ".join(hi_parts) + ("?" if is_question else "")
    en = " ".join(en_parts)
    en = (en[:1].upper() + en[1:]) + ("?" if is_question else ".")
    if all(e.pos == "greeting" for e in entries):  # rule 1
        hi = " ".join(e.hi for e in entries) + ("?" if is_question else "")
    return Sentence(hi=hi, en=en, glosses=words)


def text_to_glosses(text: str, lex: Lexicon | None = None) -> list[str]:
    """Direction B — spoken Hindi/English text → ISL gloss sequence.

    STEPS:
      1. normalise and split into words;
      2. greedily match the longest known phrase first ("आप कैसे हैं", "train ticket");
      3. drop words ISL does not sign (STOPWORDS);
      4. resolve "कल" to YESTERDAY when the sentence is in past tense;
      5. move question words to the end (ISL grammar);
      6. unknown words → "TEXT:<word>" so they still appear as captions.
    """
    lex = lex or default_lexicon()
    tokens = norm_word(text).split()
    past = any(t in PAST_MARKERS for t in tokens)
    out: list[str] = []
    wh: list[str] = []
    i = 0
    while i < len(tokens):
        matched = False
        for n in range(min(lex.max_phrase_words, len(tokens) - i), 0, -1):  # step 2: longest first
            phrase = " ".join(tokens[i : i + n])
            gloss = lex.word_index.get(phrase)
            if gloss and not (n == 1 and phrase in STOPWORDS):
                if past and gloss in TIME_DEPENDENT and phrase == norm_word("कल"):
                    gloss = TIME_DEPENDENT[gloss]  # step 4
                out.append(gloss)
                i += n
                matched = True
                break
        if matched:
            continue
        word = tokens[i]
        i += 1
        if word in STOPWORDS:  # step 3
            continue
        if word in WH_WORDS:  # step 5 (collected, appended at the end)
            wh.append(TEXT_PREFIX + word)
            continue
        out.append(TEXT_PREFIX + word)  # step 6
    return out + wh


def is_text_token(token: str) -> bool:
    """True for "TEXT:<word>" tokens (shown as caption chips, not signed)."""
    return token.startswith(TEXT_PREFIX)
