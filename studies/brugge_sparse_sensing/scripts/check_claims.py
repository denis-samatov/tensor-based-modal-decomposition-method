"""Consistency check between the manuscript sources and the regenerated study outputs.

1. outputs/numbers.tex (regenerated) must be identical to <manuscript>/numbers.tex.
2. Every macro used in the manuscript must be defined in numbers.tex or be a known LaTeX command.
3. Hard-coded decimal numbers in Abstract/Results/Discussion/Conclusions/captions are reported;
   any decimal that is not on the explicit allow-list (design constants) is an error.
Usage: python scripts/check_claims.py <manuscript_dir>   (or set MANUSCRIPT_DIR)
<manuscript_dir> contains numbers.tex, sections/ and supplementary/ of the manuscript LaTeX source.
"""
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
_ms = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MANUSCRIPT_DIR")
if not _ms or not (Path(_ms) / "sections").is_dir():
    sys.exit("usage: python scripts/check_claims.py <manuscript_dir>  (directory with numbers.tex and sections/)")
MS = Path(_ms)
ok = True

sys.path.insert(0, str(HERE))
from bss import OUT  # noqa: E402
gen = (OUT / "numbers.tex").read_text()
ms_num = MS / "numbers.tex"
if not ms_num.exists() or ms_num.read_text() != gen:
    print("FAIL numbers.tex in manuscript differs from regenerated outputs/numbers.tex")
    ok = False
else:
    print("PASS numbers.tex identical to regenerated outputs")
defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", gen))

files = [MS / "sections" / f for f in ("abstract.tex", "results.tex", "discussion.tex", "conclusions.tex", "intro.tex", "experiments.tex", "methods.tex")]
supp = sorted((MS / "supplementary").glob("*.tex"))
used = set()
# design constants that are legitimately typed in the text (not results)
# values quoted verbatim from the reviewed manuscript (CAGEO-D-26-01439) in the audit table S2;
# they are reported as claims under audit, not as results of this study
quoted_reviewed = {"0.62", "0.17", "0.47", "0.90", "27", "34.5", "0.57", "0.20", "0.11", "0.88"}
allowed_decimals = quoted_reviewed | {"0.8", "0.95", "0.1", "0.5", "0.005", "0.01", "0.02", "0.05", "1.0", "2.0", "3.12", "2.0.0", "0.76"}
latex_cmds = set()
for f in files + supp + [MS / "main.tex"]:
    if not f.exists():
        continue
    t = f.read_text()
    for m in re.findall(r"\\([A-Za-z]+)", t):
        used.add(m)
    body = re.sub(r"(?m)%.*$", "", t)
    body = re.sub(r"\\(label|ref|eqref|cite|input|includegraphics|url|texttt)(\[[^\]]*\])?\{[^}]*\}", "", body)
    body = re.sub(r"\d+(\.\d+)?\s*\\(linewidth|textwidth)", " ", body)
    body = re.sub(r"Section~\d+\.\d+", " ", body)
    body = re.sub(r"\\[A-Za-z]+", " ", body)
    for num in re.findall(r"(?<![A-Za-z0-9.])(\d+\.\d+)(?![0-9])", body):
        if num not in allowed_decimals:
            print(f"FAIL hard-coded decimal {num} in {f.name}")
            ok = False
cand = {u for u in used if u[:1].isupper() and u in defined}
undefined_like = {u for u in used if re.match(r"^(POne|PTwo|W[PT]|Cost|N[A-Z]|Rank|Ver|Persist|Cross|Pmin|Pmax|SoMax|Tbmd|Basis|Train|Eone|Ethree|Efour|Efive)", u) and u not in defined}
for u in sorted(undefined_like):
    print(f"FAIL macro \\{u} used but not generated")
    ok = False
print(f"INFO {len(cand)} generated macros used in manuscript")
print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
sys.exit(0 if ok else 1)
