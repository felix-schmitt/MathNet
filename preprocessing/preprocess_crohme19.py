import csv
from pathlib import Path

folder_images = Path("data/CROHME19/CROHME2019_data/Task1_onlineRec/MainTask_formula/Test/Test2019_GT")
with open("data/CROHME19/test2019.csv", "w") as f:
    writer = csv.writer(f)
    for image in sorted(folder_images.glob("*.inkml")):
        formula = ""
        with open(image) as temp_f:
            for line in temp_f.readlines():
                if """<annotation type="truth">""" in line:
                    formula = line.replace("""<annotation type="truth">""", ""
                                           ).replace("""</annotation>\n""", ""
                                                     ).replace("$", "")
                    break
        writer.writerow([formula.replace("\n ", ""), "test2019_img/" + image.name.replace(".inkml", ".png").replace("_", "-")])