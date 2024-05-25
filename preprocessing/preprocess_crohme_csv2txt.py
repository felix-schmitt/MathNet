import csv
from pathlib import Path

files = ["data/CROHME23/train.csv", "data/CROHME23/train_OffHME.csv", "data/CROHME23/val.csv", "data/CROHME19/test2019.csv"]

for file in files:
    with open(file) as f:
        reader = csv.reader(f)
        with open(file.replace(".csv", ".txt"), "w") as f_txt:
            for formula, image in reader:
                f_txt.write(formula + "\n")


