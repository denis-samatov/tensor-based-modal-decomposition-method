"""Generate editable highlights from the single canonical text source."""

from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor

root = Path(__file__).resolve().parent
lines = [
    line.strip() for line in (root / "highlights.txt").read_text().splitlines() if line.strip()
]
assert 3 <= len(lines) <= 5
assert all(len(line) <= 85 for line in lines), [(len(line), line) for line in lines]
doc = Document()
style = doc.styles["Normal"]
style.font.name = "Arial"
style.font.size = Pt(11)
style.font.color.rgb = RGBColor(0, 0, 0)
for line in lines:
    doc.add_paragraph(line, style="List Bullet")
doc.core_properties.title = "Brugge sparse reconstruction highlights"
doc.core_properties.author = ""
doc.core_properties.comments = ""
doc.save(root / "highlights.docx")
print("Highlights generated:", [len(line) for line in lines])
