"""The upload receipt must detect stale or unexpected material."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "receipt", ROOT / "docs/paper/build_submission_receipt.py"
)
RECEIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECEIPT)


@pytest.mark.parametrize("mutation", ["replace", "delete", "extra", "symlink", "duplicate"])
def test_receipt_rejects_changed_upload_tree(tmp_path, mutation):
    file = tmp_path / "article.txt"
    file.write_text("reviewed")
    row = {
        "filename": file.name,
        "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "size": file.stat().st_size,
    }
    receipt = {"files": [row]}
    RECEIPT.verify(tmp_path, receipt)
    if mutation == "replace":
        file.write_text("modified")
    elif mutation == "delete":
        file.unlink()
    elif mutation == "extra":
        (tmp_path / "scratch.txt").write_text("unreviewed")
    elif mutation == "symlink":
        (tmp_path / "alias").symlink_to(file)
    else:
        receipt["files"].append(row)
    with pytest.raises(ValueError):
        RECEIPT.verify(tmp_path, receipt)


def test_word_declaration_contains_the_canonical_confirmed_statement():
    from xml.etree import ElementTree
    from zipfile import ZipFile

    source = ROOT / "docs/paper/submission"
    expected = (source / "declaration_of_competing_interest.txt").read_text().strip()
    expected = expected.removeprefix("Declaration of competing interest").strip()
    with ZipFile(source / "declaration_of_competing_interest.docx") as archive:
        xml = ElementTree.fromstring(archive.read("word/document.xml"))
    text = " ".join(xml.itertext())
    assert expected in text
