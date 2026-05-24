"""Extract docx to plain text — UTF-8 safe."""
import sys, io, zipfile
from xml.etree import ElementTree as ET

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
path = r"C:\Users\viptd\OneDrive\Рабочий стол\files8\Zadanie_v8,1.docx"
out_path = r"Z:\birza_digital\Zadanie_v8_1.md"

with zipfile.ZipFile(path) as z:
    xml = z.read("word/document.xml").decode("utf-8")
root = ET.fromstring(xml)
lines = []
for p in root.iter(W + "p"):
    # detect heading style
    style = None
    pPr = p.find(W + "pPr")
    if pPr is not None:
        pStyle = pPr.find(W + "pStyle")
        if pStyle is not None:
            style = pStyle.get(W + "val")
    text = "".join(t.text or "" for t in p.iter(W + "t"))
    if style and style.startswith("Heading"):
        try:
            lvl = int(style.replace("Heading", "") or "1")
        except Exception:
            lvl = 1
        prefix = "#" * lvl + " "
        lines.append(prefix + text)
    else:
        lines.append(text)

content = "\n\n".join(lines)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(content)
print(f"OK → {out_path}, {len(content)} chars, {len(lines)} paragraphs")
