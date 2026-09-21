"""Build submission_ready/: the files to upload to Editorial Manager (EM), and nothing else.

EM places all LaTeX source files in one directory, so the manuscript and the supplement are
flattened: every \\input is inlined and figure paths lose their directory. The flattened sources
are compiled (Tectonic, XeTeX engine + BibTeX) to check that exactly what is uploaded builds; the
.bbl files are uploaded with the sources.

Usage: python build_submission_ready.py [--arxiv]   (needs `tectonic` and `pdftotext` on PATH)
       --arxiv also writes arxiv_v2/ and arxiv_v2.tar.gz: the manuscript source without line numbers,
       with the supplementary PDF as an ancillary file, for replacing arXiv:2607.09687v1.
Checks run: claim checker on the master sources, unresolved LaTeX references/citations,
and unresolved author placeholders in everything that is uploaded.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MS = HERE / "manuscript"
SUB = HERE / "submission"
OUT = HERE / "submission_ready"
BUILD_LOGS = HERE / "audit" / "final_submission_2026-09-21" / "build_logs"
STUDY = HERE.parents[1] / "studies" / "brugge_sparse_sensing"
INPUT = re.compile(r"\\input\{([^}]+)\}")
GRAPHIC = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)\}")


def flatten(tex: Path, base: Path | None = None) -> str:
    """Inline \\input files; like LaTeX, resolve paths relative to the root document's directory."""
    base = base or tex.parent

    def inline(m: re.Match) -> str:
        name = m.group(1)
        path = base / name
        if path.suffix != ".tex":
            path = path.with_suffix(".tex")
        return f"% ---- begin {name}\n{flatten(path, base).rstrip()}\n% ---- end {name}\n"

    text = tex.read_text()
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("\\graphicspath"))
    text = INPUT.sub(inline, text)
    return GRAPHIC.sub(lambda m: f"{m.group(1)}{Path(m.group(2)).name}}}", text)


def compile_tex(directory: Path, stem: str) -> list[str]:
    # The submission artefact must be reproducible from the locally pinned
    # Tectonic bundle.  A live bundle refresh can change the toolchain or hang
    # an otherwise offline build, so fail explicitly when the cache is absent.
    run = subprocess.run(["tectonic", "-X", "compile", "--only-cached", "--keep-logs", "--keep-intermediates", f"{stem}.tex"],
                         cwd=directory, capture_output=True, text=True)
    if run.returncode != 0:
        sys.exit(f"LaTeX build of {stem}.tex failed:\n{run.stderr[-3000:]}")
    log = (directory / f"{stem}.log").read_text(errors="replace")
    BUILD_LOGS.mkdir(parents=True, exist_ok=True)
    log_stem = f"arxiv_{stem}" if directory.name == "arxiv_build_check" else stem
    (BUILD_LOGS / f"{log_stem}.log").write_text(log)
    (BUILD_LOGS / f"{log_stem}.console.txt").write_text(run.stdout + run.stderr)
    problems = [line for line in log.splitlines()
                if "LaTeX Font Warning: Font shape" not in line
                and re.search(r"undefined|multiply defined|^!", line, re.I)]
    problems += [m.group(0) for m in re.finditer(r"Overfull \\[hv]box \(([0-9.]+)pt", log)
                 if float(m.group(1)) > 3.0]
    for ext in (".log", ".aux", ".blg", ".out", ".toc", ".spl"):
        (directory / f"{stem}{ext}").unlink(missing_ok=True)
    if problems:
        raise RuntimeError(f"LaTeX validation failed for {stem}: " + "\n".join(problems))
    return problems


def main() -> None:
    claims = subprocess.run([sys.executable, str(STUDY / "scripts" / "check_claims.py"), str(MS)], capture_output=True, text=True)
    print(claims.stdout.strip().splitlines()[-1])
    if claims.returncode != 0:
        sys.exit(claims.stdout)

    shutil.rmtree(OUT, ignore_errors=True)
    src = OUT / "manuscript_source"
    supp = HERE / "supplementary_build"  # the supplement is uploaded as PDF only
    shutil.rmtree(supp, ignore_errors=True)
    for d in (src, supp):
        d.mkdir(parents=True)

    (src / "manuscript.tex").write_text(flatten(MS / "main.tex"))
    shutil.copy(MS / "references.bib", src)
    figures = sorted({Path(m.group(2)).name for m in GRAPHIC.finditer((src / "manuscript.tex").read_text())})
    for f in figures:
        shutil.copy(STUDY / "figures" / f, src)

    s_text = flatten(MS / "supplementary" / "supplementary.tex").replace("\\bibliography{../references}", "\\bibliography{references}")
    (supp / "supplementary.tex").write_text(s_text)
    shutil.copy(MS / "references.bib", supp)
    for f in sorted((STUDY / "figures").glob("fig*.pdf")):
        shutil.copy(f, supp)

    problems = compile_tex(src, "manuscript") + compile_tex(supp, "supplementary")
    shutil.move(src / "manuscript.pdf", OUT / "manuscript.pdf")
    shutil.move(supp / "supplementary.pdf", OUT / "supplementary_material.pdf")
    shutil.copy(OUT / "manuscript.pdf", MS / "main.pdf")
    shutil.copy(OUT / "supplementary_material.pdf", MS / "supplementary" / "supplementary.pdf")

    cover = OUT / "cover_letter_build"
    cover.mkdir()
    (cover / "cover_letter.tex").write_text(flatten(SUB / "cover_letter.tex"))
    problems += compile_tex(cover, "cover_letter")
    shutil.move(cover / "cover_letter.pdf", OUT / "cover_letter.pdf")
    shutil.rmtree(cover)

    shutil.copy(SUB / "highlights.docx", OUT / "highlights.docx")
    shutil.copy(STUDY / "figures" / "graphical_abstract.pdf", OUT / "graphical_abstract.pdf")
    shutil.copy(SUB / "data_availability_statement.txt", OUT)
    shutil.copy(SUB / "code_availability_statement.txt", OUT)
    shutil.copy(SUB / "ai_use_statement.txt", OUT)
    shutil.copy(SUB / "declaration_of_competing_interest.docx", OUT)
    shutil.rmtree(supp)

    print("\n".join(f"LATEX: {p}" for p in problems) or "LaTeX: no undefined references/citations, no overfull boxes > 3 pt")
    placeholders = []
    author_marker = "AUTHOR" + " INPUT REQUIRED"
    for f in sorted(OUT.rglob("*")):
        if f.suffix in {".tex", ".txt", ".bbl"}:
            placeholders += [f"{f.relative_to(OUT)}: {l.strip()[:140]}" for l in f.read_text().splitlines() if author_marker in l]
    for pdf in ("manuscript.pdf", "supplementary_material.pdf", "cover_letter.pdf"):
        text = subprocess.run(["pdftotext", str(OUT / pdf), "-"], capture_output=True, text=True, check=True).stdout.replace("-\n", "").replace("\n", " ")
        for bad in ("AUTHOR INPUT", "INPUT REQUIRED", "??", "[?]"):
            if bad in text:
                placeholders.append(f"{pdf} text contains {bad!r}")
    print("Unresolved placeholders:" if placeholders else "No unresolved placeholders.")
    print("\n".join(f"  {p}" for p in placeholders))
    if placeholders:
        sys.exit("Submission package contains unresolved placeholders")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}{'/' if f.is_dir() else ''}")
    if "--arxiv" in sys.argv:
        build_arxiv(src)


def build_arxiv(src: Path) -> None:
    arx = HERE / "arxiv_v2"
    shutil.rmtree(arx, ignore_errors=True)
    (arx / "anc").mkdir(parents=True)
    tex = (src / "manuscript.tex").read_text()
    tex = tex.replace("\\linenumbers\n", "").replace("\\journal{Applied Computing and Geosciences}", "\\journal{arXiv}")
    (arx / "manuscript.tex").write_text(tex)
    for f in src.iterdir():
        if f.suffix == ".pdf" or f.name in ("manuscript.bbl", "references.bib"):
            shutil.copy(f, arx)
    shutil.copy(OUT / "supplementary_material.pdf", arx / "anc" / "supplementary_material.pdf")
    check = HERE / "arxiv_build_check"
    shutil.rmtree(check, ignore_errors=True)
    shutil.copytree(arx, check)
    problems = compile_tex(check, "manuscript")
    shutil.rmtree(check)
    subprocess.run(["tar", "-czf", str(HERE / "arxiv_v2.tar.gz"), "-C", str(arx), "."], check=True)
    print(f"arXiv package: arxiv_v2.tar.gz ({'builds' if not problems else problems})")


if __name__ == "__main__":
    main()
