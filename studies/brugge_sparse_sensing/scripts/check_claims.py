"""Consistency check between the manuscript sources and the regenerated study outputs.

1. Every non-timing macro in regenerated outputs/numbers.tex must be identical to the manuscript;
   Cost* wall-clock macros may vary between machines/runs and are reported separately.
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
    sys.exit(
        "usage: python scripts/check_claims.py <manuscript_dir>  (directory with numbers.tex and sections/)"
    )
MS = Path(_ms)
ok = True

sys.path.insert(0, str(HERE))
from bss import OUT  # noqa: E402

gen = (OUT / "numbers.tex").read_text()
optimization_numbers_path = OUT / "optimization" / "optimization_numbers.tex"
optimization_numbers = (
    optimization_numbers_path.read_text() if optimization_numbers_path.exists() else ""
)
e9_numbers_path = OUT / "e9_property_mode" / "e9_numbers.tex"
e9_numbers = e9_numbers_path.read_text() if e9_numbers_path.exists() else ""
ms_num = MS / "numbers.tex"
if not ms_num.exists():
    inputs = re.findall(r"\\input\{([^}]+)\}", (MS / "main.tex").read_text())
    candidates = [MS / name for name in inputs if Path(name).name == "numbers.tex"]
    if len(candidates) == 1:
        ms_num = candidates[0]
main_inputs = re.findall(r"\\input\{([^}]+)\}", (MS / "main.tex").read_text())
optimization_candidates = [
    MS / name for name in main_inputs if Path(name).name == "optimization_numbers.tex"
]
e9_candidates = [MS / name for name in main_inputs if Path(name).name == "e9_numbers.tex"]
macro_re = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}")


def deterministic_numbers(text: str) -> str:
    """Return the generated file with explicitly run-dependent timing macros removed."""
    return "\n".join(
        line
        for line in text.splitlines()
        if not ((match := macro_re.fullmatch(line)) and match.group(1).startswith("Cost"))
    )


if not ms_num.exists():
    print("FAIL manuscript numbers.tex not found")
    ok = False
else:
    manuscript_numbers = ms_num.read_text()
    generated_macros = dict(macro_re.findall(gen))
    manuscript_macros = dict(macro_re.findall(manuscript_numbers))
    missing_definitions = set(generated_macros) ^ set(manuscript_macros)
    if missing_definitions:
        print(
            f"FAIL macro definitions differ (including timing macros): {sorted(missing_definitions)}"
        )
        ok = False
    if deterministic_numbers(manuscript_numbers) != deterministic_numbers(gen):
        print("FAIL non-timing macros in manuscript differ from regenerated outputs/numbers.tex")
        ok = False
    else:
        timing_drift = {
            name: (manuscript_macros.get(name), value)
            for name, value in generated_macros.items()
            if name.startswith("Cost") and manuscript_macros.get(name) != value
        }
        print("PASS all non-timing manuscript macros match regenerated outputs")
        if timing_drift:
            print(
                f"INFO {len(timing_drift)} wall-clock Cost* macros differ as expected between runs"
            )
if optimization_numbers:
    if len(optimization_candidates) != 1 or not optimization_candidates[0].exists():
        print("FAIL manuscript optimization_numbers.tex not found")
        ok = False
    elif optimization_candidates[0].read_text() != optimization_numbers:
        print("FAIL manuscript optimization macros differ from canonical generated output")
        ok = False
    else:
        print("PASS optimization manuscript macros match generated outputs")
if e9_numbers:
    if len(e9_candidates) != 1 or not e9_candidates[0].exists():
        print("FAIL manuscript E9 numbers.tex not found")
        ok = False
    elif e9_candidates[0].read_text() != e9_numbers:
        print("FAIL manuscript E9 macros differ from canonical generated output")
        ok = False
    else:
        print("PASS E9 manuscript macros match generated outputs")
combined_generated = gen + "\n" + optimization_numbers + "\n" + e9_numbers
defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", combined_generated))

files = [
    MS / "sections" / f
    for f in (
        "abstract.tex",
        "results.tex",
        "discussion.tex",
        "conclusions.tex",
        "intro.tex",
        "experiments.tex",
        "methods.tex",
        "optimization.tex",
        "e9_property_mode.tex",
    )
]
supp = sorted((MS / "supplementary").glob("*.tex"))
used = set()
# design constants that are legitimately typed in the text (not results)
# values quoted verbatim from the reviewed manuscript (CAGEO-D-26-01439) in the audit table S2;
# they are reported as claims under audit, not as results of this study
quoted_reviewed = {"0.62", "0.17", "0.47", "0.90", "27", "34.5", "0.57", "0.20", "0.11", "0.88"}
allowed_decimals = quoted_reviewed | {
    "0.8",
    "0.95",
    "0.1",
    "0.5",
    "0.005",
    "0.01",
    "0.02",
    "0.05",
    "1.0",
    "2.0",
    "3.12",
    "2.0.0",
    "0.76",
    "0.03",
    "1.5",
}
for f in files + supp + [MS / "main.tex"]:
    if not f.exists():
        continue
    t = f.read_text()
    for m in re.findall(r"\\([A-Za-z]+)", t):
        used.add(m)
    body = re.sub(r"(?m)%.*$", "", t)
    body = re.sub(
        r"\\(label|ref|eqref|cite|input|includegraphics|url|texttt)(\[[^\]]*\])?\{[^}]*\}", "", body
    )
    body = re.sub(r"\d+(\.\d+)?\s*\\(linewidth|textwidth)", " ", body)
    body = re.sub(r"Section~\d+\.\d+", " ", body)
    body = re.sub(r"\\[A-Za-z]+", " ", body)
    for num in re.findall(r"(?<![A-Za-z0-9.])(\d+\.\d+)(?![0-9])", body):
        if num not in allowed_decimals:
            print(f"FAIL hard-coded decimal {num} in {f.name}")
            ok = False
cand = {u for u in used if u[:1].isupper() and u in defined}
undefined_like = {
    u
    for u in used
    if re.match(
        r"^(POne|PTwo|W[PT]|Cost|N[A-Z]|Rank|Ver|Persist|Cross|Pmin|Pmax|SoMax|Tbmd|Basis|Train|Eone|Ethree|Efour|Efive|Opt|ENine)",
        u,
    )
    and u not in defined
}
for u in sorted(undefined_like):
    print(f"FAIL macro \\{u} used but not generated")
    ok = False
print(f"INFO {len(cand)} generated macros used in manuscript")
print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
sys.exit(0 if ok else 1)
