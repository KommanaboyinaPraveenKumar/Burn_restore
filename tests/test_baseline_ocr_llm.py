"""The OCR + LLM baseline must take only what OCR reads past the known prefix."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_align_read():
    from baseline_ocr_llm import align_read
    assert align_read("thinking about his debts, an", "about his debts, and") == "d"
    assert align_read("thinking about his debts, an", "ABOUT HIS DEBTS, AND") == "D"   # caseless match
    assert align_read("who were no beauties, po", "were no beauties, poor things") == "or things"
    assert align_read("anything", "") == ""
    assert align_read("he said", "he said") == ""
