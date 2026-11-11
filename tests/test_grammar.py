"""Both translation directions, checked on 30+ real-life sentences each (PRD TRD §6)."""

import pytest

from sanket.core.grammar import glosses_to_sentence, is_text_token, text_to_glosses

# ---------- Direction A: signs → Hindi + English -------------------------------------------
A = [
    (["hello"], "नमस्ते", "Hello."),
    (["thankyou"], "धन्यवाद", "Thank you."),
    (["goodmorning"], "सुप्रभात", "Good morning."),
    (["goodnight"], "शुभ रात्रि", "Good night."),
    (["howareyou"], "आप कैसे हैं?", "How are you?"),
    (["hello", "howareyou"], "नमस्ते आप कैसे हैं?", "Hello how are you?"),
    (["i", "happy"], "मैं ख़ुश हूँ", "I am happy."),
    (["i", "sad"], "मैं दुखी हूँ", "I am sad."),
    (["i", "sick"], "मैं बीमार हूँ", "I am sick."),
    (["i", "deaf"], "मैं बहरा हूँ", "I am deaf."),
    (["you", "beautiful"], "आप सुंदर हैं", "You are beautiful."),
    (["he", "healthy"], "वह स्वस्थ है", "He is healthy."),
    (["she", "happy"], "वह ख़ुश है", "She is happy."),
    (["we", "happy"], "हम ख़ुश हैं", "We are happy."),
    (["they", "poor"], "वे ग़रीब हैं", "They are poor."),
    (["it", "expensive"], "यह महँगा है", "It is expensive."),
    (["it", "cheap"], "यह सस्ता है", "It is cheap."),
    (["price", "high"], "क़ीमत ऊँचा है", "Price is high."),
    (["house", "biglarge"], "घर बड़ा है", "House is big."),
    (["shirt", "red"], "कमीज़ लाल है", "Shirt is red."),
    (["medicine", "bad"], "दवाई बुरा है", "Medicine is bad."),
    (["i", "doctor"], "मैं डॉक्टर", "I doctor."),
    (["i", "student"], "मैं विद्यार्थी", "I student."),
    (["mother", "hospital"], "माँ अस्पताल", "Mother hospital."),
    (["today", "monday"], "आज सोमवार", "Today Monday."),
    (["tomorrow", "market"], "कल बाज़ार", "Tomorrow market."),
    (["i", "bank"], "मैं बैंक", "I bank."),
    (["you", "friend"], "आप दोस्त", "You friend."),
    (["train", "slow"], "रेलगाड़ी धीमा है", "Train is slow."),
    (["kitchen", "dirty"], "रसोई गंदा है", "Kitchen is dirty."),
    (["alright"], "ठीक है", "Alright."),
]


@pytest.mark.parametrize("glosses,hi,en", A)
def test_signs_to_sentence(glosses, hi, en):
    s = glosses_to_sentence(glosses)
    assert s.hi == hi
    assert s.en == en


def test_unknown_gloss_ignored():
    assert glosses_to_sentence(["notaword"]).hi == ""


# ---------- Direction B: speech text → glosses ----------------------------------------------
B = [
    ("नमस्ते", ["hello"]),
    ("धन्यवाद", ["thankyou"]),
    ("शुक्रिया", ["thankyou"]),
    ("आप कैसे हैं", ["howareyou"]),
    ("सुप्रभात", ["goodmorning"]),
    ("मैं ख़ुश हूँ", ["i", "happy"]),
    ("मैं खुश हूं", ["i", "happy"]),  # no nukta, anusvara instead of chandrabindu
    ("मैं कल बाज़ार गया था", ["i", "yesterday", "market", "TEXT:गया"]),
    ("कल बाज़ार चलेंगे", ["tomorrow", "market", "TEXT:चलेंगे"]),
    ("यह बहुत महंगा है", ["it", "TEXT:बहुत", "expensive"]),
    ("यह सस्ता है", ["it", "cheap"]),
    ("दवाई की क़ीमत", ["medicine", "price"]),
    ("डॉक्टर अस्पताल में है", ["doctor", "hospital"]),
    ("मेरी माँ बीमार है", ["i", "mother", "sick"]),
    ("आज सोमवार है", ["today", "monday"]),
    ("ट्रेन टिकट", ["trainticket"]),
    ("रेलवे स्टेशन", ["trainstation"]),
    ("दुकान बंद है", ["storeorshop", "TEXT:बंद"]),
    ("पैसे दो", ["money", "TEXT:दो"]),
    ("लाल कमीज़", ["red", "shirt"]),
    ("नई किताब", ["new", "book"]),
    ("मोबाइल फ़ोन", ["cellphone", "cellphone"]),
    ("पुलिस", ["police"]),
    ("hello", ["hello"]),
    ("thank you", ["thankyou"]),
    ("good morning", ["goodmorning"]),
    ("I am happy", ["i", "happy"]),
    ("the price is high", ["price", "high"]),
    ("train ticket please", ["trainticket"]),
    ("where is the hospital", ["hospital", "TEXT:where"]),  # question word moves to the end
    ("क्या आप डॉक्टर हैं", ["you", "doctor", "TEXT:क्या"]),
    ("रमेश स्कूल गया", ["TEXT:रमेश", "school", "TEXT:गया"]),  # names have no sign → caption
]


@pytest.mark.parametrize("text,expected", B)
def test_text_to_glosses(text, expected):
    assert text_to_glosses(text) == expected


def test_empty_text():
    assert text_to_glosses("") == []
    assert text_to_glosses("   ।  ") == []


def test_text_token_helper():
    assert is_text_token("TEXT:राम") and not is_text_token("hello")
