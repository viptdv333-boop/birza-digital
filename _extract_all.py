"""Extract multiple docx → md, UTF-8 safe."""
import sys, io, zipfile, os
from xml.etree import ElementTree as ET

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
SRC = r"C:\Users\viptd\OneDrive\Рабочий стол\files8"
DST = r"Z:\birza_digital\_v81_docs"
os.makedirs(DST, exist_ok=True)

files = [
    "Zadanie_v8,1.docx",
    "Etapy_v8,1.docx",
    "Reglament_v8_updated.docx",
    "Sborka.docx",
    "Zadanie_v9.docx",
]

for fname in files:
    src = os.path.join(SRC, fname)
    if not os.path.exists(src):
        print(f"[SKIP] {fname} not found")
        continue
    with zipfile.ZipFile(src) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    root = ET.fromstring(xml)
    lines = []
    for p in root.iter(W + "p"):
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
            lines.append("#" * lvl + " " + text)
        else:
            lines.append(text)
    out = os.path.join(DST, fname.replace(".docx", ".md").replace(",", "_"))
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(lines))
    print(f"[OK] {fname} → {out} ({sum(len(l) for l in lines)} chars)")
