import numpy
from pathlib import Path
import cv2

class formula_cutter():
    def __init__(self, white_threshold, y_threshold):
        self.white_threshold = white_threshold
        self.y_threshold = y_threshold

    def cut(self, image):
        white_lines = [all(image_row >= 255 - self.white_threshold) for image_row in image[0, :]]
        cuts = [int(white_line_i + self.y_threshold/2) for white_line_i, _ in enumerate(white_lines) if all(white_lines[white_line_i:white_line_i+self.y_threshold]) and not all(white_lines[:white_line_i])]
        while image.shape[1] > 300 and not cuts:
            modified_y_threshold = self.y_threshold
            modified_y_threshold -= 1
            cuts = [int(white_line_i + modified_y_threshold / 2) for white_line_i, _ in enumerate(white_lines) if
                    all(white_lines[white_line_i:white_line_i + modified_y_threshold]) and not all(
                        white_lines[:white_line_i])]
            if modified_y_threshold < 2/3*self.y_threshold:
                break
        cuts.insert(0, 0)
        cuts.append(len(white_lines))
        formula_lines = []
        for i, _ in enumerate(cuts):
            if cuts[i] >= len(white_lines):
                break
            if cuts[i] + 1 == cuts[i+1]:
                continue
            formula_lines.append(image[:, cuts[i]:cuts[i+1]])
        return formula_lines



if __name__ == '__main__':
    fc = formula_cutter(0, 30)
    image_folder = Path("data/realFormula/img-corrected")
    cutted_image_folder = Path("data/realFormula/cuttedImages")
    cutted_image_folder.mkdir(parents=True, exist_ok=True)
    for image_name in image_folder.glob("*.png"):
        img = cv2.imread(str(image_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape([img.shape[0], img.shape[1], 1])
        img = img.transpose(2, 0, 1)
        formula_lines = fc.cut(img)
        for formula_line_i, formula_line in enumerate(formula_lines):
            cv2.imwrite(str(cutted_image_folder / f"{image_name.name[:-4]}_{formula_line_i}.png"), formula_line.transpose(1, 2, 0))
