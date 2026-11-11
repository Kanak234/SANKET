"""PRD FR-C1: SANKET's logic must never touch the network. The self-test blocks sockets and runs
every stage; here we additionally scan the runtime package for network libraries."""

import re
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent / "sanket"


def test_no_network_imports_in_runtime():
    banned = re.compile(r"^\s*(import|from)\s+(requests|urllib|http\.client|httpx|aiohttp|socket)\b", re.M)
    offenders = [p.name for p in PKG.rglob("*.py") if banned.search(p.read_text(encoding="utf-8"))]
    # __main__ imports socket only to BLOCK it during the self-test
    assert offenders in ([], ["__main__.py"]), offenders


def test_runtime_never_imports_torch():
    assert not any("import torch" in p.read_text(encoding="utf-8") for p in PKG.rglob("*.py"))


def test_pipeline_with_socket_blocked(monkeypatch):
    import socket

    def boom(*a, **k):
        raise AssertionError("network used")

    monkeypatch.setattr(socket, "socket", boom)
    from sanket.core.grammar import glosses_to_sentence, text_to_glosses
    from sanket.core.recognizer import Recognizer
    from sanket.core.signs import SignDictionary, SignPlayer
    from tests.synthetic import clip

    Recognizer().predict(clip(0, np.random.default_rng(0)))
    assert glosses_to_sentence(["i", "happy"]).en == "I am happy."
    assert SignPlayer(SignDictionary()).timeline(text_to_glosses("नमस्ते"))
