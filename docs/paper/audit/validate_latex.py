"""Structural validation of the LaTeX sources without a TeX engine:
braces/environments balance, labels vs refs, cite keys vs bibliography, macro definitions,
figure files, and a word count of the main text after macro expansion."""
import re, sys
from pathlib import Path
M = Path(__file__).resolve().parents[1] / "manuscript"
ok = True
def expand_inputs(path, base):
    t = path.read_text()
    def rep(m):
        p = base / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
        return expand_inputs(p, base) if p.exists() else f"%MISSING INPUT {m.group(1)}\n"
    return re.sub(r"\\input\{([^}]*)\}", rep, t)
for doc, base in ((M / "main.tex", M), (M / "supplementary" / "supplementary.tex", M / "supplementary")):
    t = expand_inputs(doc, base)
    body = re.sub(r"(?<!\\)%.*", "", t)
    if "MISSING INPUT" in t:
        print("FAIL missing input in", doc.name, re.findall(r"MISSING INPUT (\S+)", t)); ok = False
    depth = 0
    for ch in re.sub(r"\\[{}]", "", body):
        depth += (ch == "{") - (ch == "}")
        if depth < 0: break
    if depth != 0:
        print(f"FAIL brace imbalance {depth} in {doc.name}"); ok = False
    envs = re.findall(r"\\(begin|end)\{([^}]*)\}", body)
    stack = []
    for kind, name in envs:
        if kind == "begin": stack.append(name)
        elif not stack or stack.pop() != name:
            print("FAIL environment mismatch at", name, "in", doc.name); ok = False; break
    label_list = re.findall(r"\\label\{([^}]*)\}", body)
    duplicates = sorted({x for x in label_list if label_list.count(x) > 1})
    if duplicates:
        print("FAIL duplicate labels", duplicates); ok = False
    labels = set(re.findall(r"\\label\{([^}]*)\}", body))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]*)\}", body))
    for r in sorted(refs - labels):
        print(f"FAIL undefined ref {r} in {doc.name}"); ok = False
    bib = set(re.findall(r"@\w+\{([^,]+),", (M / "references.bib").read_text()))
    cites = set(k.strip() for c in re.findall(r"\\cite\{([^}]*)\}", body) for k in c.split(","))
    for c in sorted(cites - bib):
        print(f"FAIL citation {c} not in bib ({doc.name})"); ok = False
    if doc.name == "main.tex":
        unused = bib - cites
        print("INFO bib entries not cited in main:", sorted(unused))
    defs = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", body))
    used = set(re.findall(r"\\([A-Z][A-Za-z]+)", body))
    known = {"LaTeX", "Sigma", "Require", "State", "If", "EndIf", "For", "EndFor", "Return", "Comment", "Else"}
    for u in sorted(used - set(defs) - known):
        if not re.search(r"\\\\" + u, body):
            print(f"FAIL undefined macro \\{u} in {doc.name}"); ok = False
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", body):
        cand = [M / g, base / g, M / "supplementary" / ".." / g]
        if not any(c.exists() for c in cand):
            print("FAIL missing figure", g); ok = False
    if doc.name == "main.tex":
        def words(section_text):
            s = section_text
            for k, v in defs.items():
                s = s.replace("\\" + k + "{}", v).replace("\\" + k, v)
            s = re.sub(r"\\begin\{(figure|table|algorithm)\*?\}.*?\\end\{\1\*?\}", " ", s, flags=re.S)
            s = re.sub(r"\$[^$]*\$", " X ", s)
            s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", s)
            return len(re.findall(r"[A-Za-z0-9][\w\-.,]*", s))
        abstract = expand_inputs(M / "sections" / "abstract.tex", M)
        print("INFO abstract words ~", words(abstract))
        mainbody = body[body.index("\\end{frontmatter}"):body.index("\\section*{Computer code availability}")]
        print("INFO main text words (Introduction-Conclusions, excl. figures/tables) ~", words(mainbody))
print("LATEX STRUCTURE OK" if ok else "LATEX STRUCTURE PROBLEMS")
sys.exit(0 if ok else 1)
