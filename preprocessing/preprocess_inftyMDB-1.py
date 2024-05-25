import csv
import subprocess
from pathlib import Path

folder = Path("data/inftyMDB-1")

for file in folder.glob("*0.xhtml"):
    # pandoc -f html -t latex -o 0001-0100.tex 0001-0100.xhtml
    if Path(str(file).replace(".xhtml", ".tex")).exists(): continue
    convert = subprocess.Popen(["pandoc", "-f", "html", "-t", "latex", "-o", str(file).replace(".xhtml", ".tex"), str(file)])
    convert.wait(200)

formulas = {}
for file in sorted(folder.glob("*.tex")):
    with open(file, "r") as f:
        number = None
        formula = None
        nextLine = False
        for line in f.readlines():
            if line.startswith("No:"):
                number = int(line.split(" ")[0][3:])
            if "Recognition Result:" in line:
                formula = line.replace("Recognition Result:", "").replace("Corrected Result:", "")
                if "\\)" not in formula:
                    nextLine = True
                else:
                    formulas[number] = formula
            if nextLine:
                formula += line
                if "\\)" in line:
                    nextLine = False
                    formulas[number] = formula

with open(folder / "formulas.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "formula"])
    for number, formula in formulas.items():
        l = formula.replace("Recognition Result:", "").replace("Corrected Result:", "").replace("\n", "").replace("\\(", "")
        l = l[:l.find("\\)")]
        writer.writerow([number, l])