"""The lexicon's class order must equal the training label map, or every prediction is mislabelled."""

import json
from pathlib import Path

from sanket.core.lexicon import Lexicon, norm_word

REPO = Path(__file__).resolve().parent.parent


def test_263_entries_match_label_map():
    lex = Lexicon()
    lm = json.loads((REPO / "training/splits/label_map_include.json").read_text())
    assert len(lex) == 263
    assert {g: e.index for g, e in lex.entries.items()} == lm
    assert lex.labels[lm["hello"]] == "hello"


def test_every_entry_has_hindi_and_english():
    for e in Lexicon().entries.values():
        assert e.hi.strip() and e.en.strip() and e.pos in {"noun", "adj", "pronoun", "greeting", "time", "verb"}


def test_norm_word():
    assert norm_word("ज़मीन") == norm_word("जमीन")
    assert norm_word("हूँ") == norm_word("हूं")
    assert norm_word("Hello, World!") == "hello world"


def test_training_label_rule_matches_all_split_files():
    from training.common import label_from_path, label_map, split_paths

    lm = label_map()
    for split in ("train", "val", "test"):
        for rel in split_paths(split):
            assert label_from_path(rel) in lm, rel
