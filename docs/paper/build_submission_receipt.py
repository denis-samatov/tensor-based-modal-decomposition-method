"""Record or verify the exact local upload files; never imply author approval."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

PAPER = Path(__file__).resolve().parent
PURPOSES = {
    "manuscript.pdf": ("Compiled manuscript reference", "manuscript/main.tex"),
    "supplementary_material.pdf": (
        "Publication supplement",
        "manuscript/supplementary/supplementary.tex",
    ),
    "cover_letter.pdf": ("Editor cover letter", "submission/cover_letter.tex"),
    "highlights.docx": ("Editable highlights", "submission/highlights.txt"),
    "graphical_abstract.pdf": (
        "Graphical abstract",
        "../../studies/brugge_sparse_sensing/scripts/make_graphical_abstract.py",
    ),
    "data_availability_statement.txt": (
        "Data availability field",
        "submission/data_availability_statement.txt",
    ),
    "code_availability_statement.txt": (
        "Code availability field",
        "submission/code_availability_statement.txt",
    ),
    "ai_use_statement.txt": ("AI disclosure field", "submission/ai_use_statement.txt"),
    "declaration_of_competing_interest.docx": (
        "Competing interests field",
        "submission/declaration_of_competing_interest.txt",
    ),
    "manuscript_source/manuscript.tex": ("Editable manuscript source", "manuscript/main.tex"),
    "manuscript_source/references.bib": ("Bibliography source", "manuscript/references.bib"),
    "manuscript_source/manuscript.bbl": (
        "Compiled bibliography companion",
        "manuscript/references.bib",
    ),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files_in(root: Path) -> dict[str, Path]:
    entries = list(root.rglob("*"))
    if any(p.is_symlink() for p in entries):
        raise ValueError("Upload tree must not contain symlinks")
    return {p.relative_to(root).as_posix(): p for p in entries if p.is_file()}


def verify(root: Path, receipt: dict) -> None:
    actual = files_in(root)
    rows = receipt["files"]
    expected = {row["filename"]: row for row in rows}
    if len(expected) != len(rows) or set(actual) != set(expected):
        raise ValueError("Upload file inventory differs from receipt")
    for name, path in actual.items():
        row = expected[name]
        if path.stat().st_size != row["size"] or digest(path) != row["sha256"]:
            raise ValueError(f"Upload file changed: {name}")


def build(root: Path, paper: Path, commit: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Code revision must be a full Git commit SHA")
    rows = []
    files = files_in(root)
    tex = files["manuscript_source/manuscript.tex"].read_text()
    figures = set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex))
    expected = set(PURPOSES) | {f"manuscript_source/{name}" for name in figures}
    if set(files) != expected:
        raise ValueError("Upload tree differs from the source-derived allowlist")
    for name, path in sorted(files.items()):
        if name in PURPOSES:
            purpose, source = PURPOSES[name]
        else:
            purpose = "Manuscript figure companion"
            source = "../../studies/brugge_sparse_sensing/figures/" + path.name
        source_path = paper / source
        if not source_path.is_file():
            raise ValueError(f"Missing canonical source for {name}")
        rows.append(
            {
                "filename": name,
                "sha256": digest(path),
                "size": path.stat().st_size,
                "purpose": purpose,
                "source": source,
                "source_sha256": digest(source_path),
                "build_date": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
            }
        )
    return {
        "schema_version": 1,
        "status": "BLOCKED",
        "upload_root": "submission_ready",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "proposed_tag": "v2.2.0",
        "archival_doi": None,
        "author_confirmations": "Received in this task; ORCIDs supplied and check digits validated",
        "limitations": [
            "Archival publication and version-specific DOI pending",
            "Three transferred-EM interface checks remain",
            "Official Declaration Tool DOCX requires author login if required by the portal",
        ],
        "build_date_definition": "UTC artifact modification time; copied source assets may retain their generation time",
        "validation_scope": "Local file identity and readability; not an Editorial Manager submission receipt",
        "files": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--code-commit")
    args = parser.parse_args()
    output = PAPER / "SUBMISSION_RECEIPT.json"
    root = PAPER / "submission_ready"
    if args.check:
        verify(root, json.loads(output.read_text()))
        print("SUBMISSION RECEIPT VERIFIED: exact file set, sizes and SHA-256 hashes")
    else:
        if not args.code_commit:
            parser.error("--code-commit is required when creating the receipt")
        receipt = build(root, PAPER, args.code_commit)
        verify(root, receipt)
        output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"Receipt written for {len(receipt['files'])} files; status remains BLOCKED")


if __name__ == "__main__":
    main()
