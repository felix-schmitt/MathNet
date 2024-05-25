import csv
from pathlib import Path

folder = "data/CROHME23/"

with open(folder + "bimodal.csv") as f:
    reader = list(csv.reader(f))
    with open(folder + "train.csv", "w") as f:
        writer = csv.writer(f)
        for id, flag, inkml, img, lg, symlg in reader:
            if flag == "train":
                formula = ""
                if not Path(folder + "WebData_CROHME23" + inkml).exists():
                    continue
                with open(folder + "WebData_CROHME23" + inkml) as temp_f:
                    for line in temp_f.readlines():
                        if """<annotation type="truth">""" in line:
                            formula = line.replace("""<annotation type="truth">""", ""
                                                   ).replace("""</annotation>\n""", ""
                                                             ).replace("$", "")
                            break
                if formula[-1] == " ":
                    formula = formula[:-1]
                if formula[-1] == "\n":
                    formula = formula[:-1]
                writer.writerow([formula.replace("\n ", ""), "train_img/" + Path(img).name.replace("_", "-")])
    with open(folder + "val.csv", "w") as f:
        writer = csv.writer(f)
        for id, flag, inkml, img, lg, symlg in reader:
            if flag == "val":
                formula = ""
                if not Path(folder + "WebData_CROHME23" + inkml).exists():
                    continue
                with open(folder + "WebData_CROHME23" + inkml) as temp_f:
                    for line in temp_f.readlines():
                        if """<annotation type="truth">""" in line:
                            formula = line.replace("""<annotation type="truth">""", ""
                                                   ).replace("""</annotation>\n""", ""
                                                             ).replace("$", "")
                            break
                if formula[-1] == " ":
                    formula = formula[:-1]
                if formula[-1] == "\n":
                    formula = formula[:-1]
                writer.writerow([formula.replace("\n ", ""), "val_img/" + Path(img).name.replace("_", "-")])

folder = Path("data/CROHME23/OffHME/img")
csvFile = Path("data/CROHME23/train_OffHME.csv")
with open(csvFile, "w") as f:
    writer = csv.writer(f)
    for img in sorted(folder.glob("*.png")):
        formulaFile = folder.parent / "label" / img.name.replace(".png", ".txt")
        formula = ""
        emptyRow = False
        with open(formulaFile) as tempF:
            for line in tempF.readlines():
                if emptyRow:
                    formula = line[:-1]
                    break
                if line == "\n":
                    emptyRow = True
        if formula[-1] == " ":
            formula = formula[:-1]
        writer.writerow([formula.replace("\n ", ""), "img/" + img.name])
