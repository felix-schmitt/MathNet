from pathlib import Path
import shutil
import os
import subprocess
import pdf2image
from preprocessing.formula2image import main_parallel
from tqdm import tqdm

basic = r"""\documentclass[12pt]{article}
\pagestyle{empty}
\usepackage{amsmath}
\begin{document}

$$ %s $$

\end{document}
"""
class render_prediction():
    def __init__(self, trainer, output_folder):
        self.trainer = trainer
        self.output_folder = output_folder
        self.failed_rendering = 0

    def render(self):
        temp_folder = Path("temp")
        if temp_folder.exists():
            shutil.rmtree(temp_folder)
        temp_folder.mkdir(parents=True)
        self.failed_rendering = 0
        pbar = tqdm(self.load_predictions(), postfix=f"render formulae, failed {self.failed_rendering}")
        for prediction in pbar:
            rend = self.formula_to_image(formula=prediction[1], name=prediction[0], folder=temp_folder,
                             rend_setup=[basic, {'fmt': 'png', 'dpi': 200, 'grayscale': False}])
            if rend:
                self.crop_formula(self.output_folder, Path("temp"))
            pbar.postfix = f"render formulae, failed {self.failed_rendering}"

    def formula_to_image(self, formula, name, folder, rend_setup):
        """ Turns given formula into images based on RENDERING_SETUPS
        returns list of lists [[image_name, rendering_setup], ...], one list for
        each rendering.
        Return None if couldn't render the formula"""
        DEVNULL = open(os.devnull, "w")
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        formula = formula.strip("%")
        name = str(folder / name[:-4])
        full_path = name
        # Create latex source
        latex = rend_setup[0] % formula
        # Write latex source
        with open(full_path + ".tex", "w") as f:
            f.write(latex)

        # Call pdflatex to turn .tex into .pdf
        code = subprocess.Popen(
                ["pdflatex", '-interaction=nonstopmode', f'-output-directory={folder}', full_path + ".tex"],
                stdout=DEVNULL, stderr=DEVNULL)
        try:
            code.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.failed_rendering += 1
            return False
        # Turn .pdf to .png
        try:
            pdf2image.convert_from_path(full_path + ".pdf", output_folder=".", fmt=rend_setup[1]['fmt'],
                                                 dpi=rend_setup[1]['dpi'], output_file=full_path,
                                                 grayscale=rend_setup[1]['grayscale'])
        except:
            self.failed_rendering += 1
            return False
        os.rename(next(Path(folder).glob("*.png")), full_path + ".png")
        return True

    def crop_formula(self, image_processed_path, image_raw_path):
        processed_image_dir = Path(image_processed_path)
        postfix = ".png"
        crop_blank_default_size = [600, 60]
        pad_size = [8, 8, 8, 8]
        buckets = [[240, 100], [320, 80], [400, 80], [400, 100], [480, 80], [480, 100], [560, 80], [560, 100], [640, 80],
                   [640, 100], [720, 80], [720, 100], [720, 120], [720, 200], [800, 100], [800, 320], [1000, 200],
                   [1000, 400], [1200, 200], [1600, 200], [1600, 1600]]
        downsample_ratio = 1.0
        images = sorted(Path(image_raw_path).glob("*.png"))
        main_parallel(filename=images[0], postfix=postfix, output_folder=processed_image_dir,
                       crop_blank_default_size=crop_blank_default_size, pad_size=pad_size, buckets=buckets,
                       downsample_ratio=downsample_ratio)

    def load_predictions(self):
        predictions = []
        with open(self.trainer.config['model']['model_save_path'] + f"/results/test_{self.trainer.ckpt['epoch']}_predictions.txt") as f:
            line = f.readline()
            while line:
                predictions.append(line.split(": "))
                line = f.readline()
        return predictions