import csv
from pathlib import Path

files = ["data/CROHME23/train_katex.txt", "data/CROHME23/train_OffHME_katex.txt", "data/CROHME23/val_katex.txt", "data/CROHME19/test2019_katex.txt"]

for file in files:
    with open(file) as f:
        katexFormulas = f.readlines()

    with open(file.replace("_katex.txt", ".csv")) as fOld:
        reader = csv.reader(fOld)
        with open(file.replace(".txt", ".csv"), "w") as fNew:
            writer = csv.writer(fNew)
            writer.writerow(['formula', 'image'])
            for i, line in enumerate(reader):
                writer.writerow([katexFormulas[i][:-1], line[1]])