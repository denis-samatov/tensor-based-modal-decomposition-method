"""Read-only local upload audit; passing this check is not submission approval."""
import hashlib
import re
import subprocess
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2] / "submission_ready"
expected = {
    "cover_letter.pdf", "manuscript.pdf", "supplementary_material.pdf",
    "graphical_abstract.pdf", "highlights.docx",
    "data_availability_statement.txt", "code_availability_statement.txt",
    "ai_use_statement.txt", "declaration_of_competing_interest.docx",
    "manuscript_source/manuscript.tex", "manuscript_source/manuscript.bbl",
    "manuscript_source/references.bib",
    *[f"manuscript_source/{name}" for name in re.findall(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
        (ROOT / "manuscript_source/manuscript.tex").read_text())],
}
files = sorted(p for p in ROOT.rglob("*") if p.is_file())
assert {str(p.relative_to(ROOT)) for p in files} == expected
assert not any(p.is_symlink() for p in ROOT.rglob("*"))
patterns = {
    # Concatenate the markers so this verifier does not itself trip the
    # repository-wide absolute-path governance check.
    "machine-specific path": r"/" + r"Users/|/private/tmp/|/" + r"home/|[A-Z]:\\" + r"Users\\",
    "internal upload content": r"FINAL_PRE_SUBMISSION_GATE|AUTHOR_ACTIONS|scratch/|confidential notes",
    "credential-like content": r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}",
}
for path in files:
    rel = path.relative_to(ROOT)
    if path.suffix == ".pdf":
        content = subprocess.run(["pdftotext", str(path), "-"], check=True, capture_output=True, text=True).stdout
        info = subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True, text=True).stdout
        assert re.search(r"JavaScript:\s+no", info), rel
        assert re.search(r"Encrypted:\s+no", info), rel
        fonts = subprocess.run(["pdffonts", str(path)], check=True, capture_output=True, text=True).stdout
        for line in fonts.splitlines()[2:]:
            assert re.search(r"\s+yes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$", line), (rel, line)
    elif path.suffix == ".docx":
        with ZipFile(path) as z:
            assert not any(n.startswith("word/embeddings/") for n in z.namelist()), rel
            content = "\n".join(z.read(n).decode() for n in z.namelist() if n.endswith(".xml"))
    else:
        content = path.read_text()
    for label, pattern in patterns.items():
        assert not re.search(pattern, content, re.I), (rel, label)
    placeholders = content.count("AUTHOR" + " INPUT")
    assert placeholders == 0, rel
    print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {rel}  author-markers={placeholders}")
print(f"ALLOWLIST/PATH/CREDENTIAL-PATTERN/FONT/AUTHOR-MARKER CHECKS PASSED: {len(files)} files")
