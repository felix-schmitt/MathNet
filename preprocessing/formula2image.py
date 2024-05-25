#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  formula2image.py
#  Turns bunch of formulas into images and dataset listing
#
#  © Copyright 2016, Anssi "Miffyli" Kanervisto
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#  

"""
Purpose of this script is to turn list of tex formulas into images
and a dataset list for OpenAI im2latex task.
Script outputs two lists:
    - im2latex.lst
        - Each row is: [idx of formula] [image name] [render type]
            - idx of formula is the line number in im2latex_formulas.lst
            - image name is name of the image (without filetype) 
            - render type is name of the method used to draw the picture
              (See RENDERING_SETUPS)
    - im2latex_formulas.lst
        - List of formulas, one per line
            -> No \n characters in formulas (doesn't affect math tex)
"""
import csv
import glob
import os
from tqdm import tqdm
from p_tqdm import p_umap
from functools import partial
import subprocess
from pathlib import Path
import shutil
from preprocessing.preprocess_images import main_parallel
import pdf2image
from preprocessing.templates.templates import *
import argparse

# Running a thread pool masks debug output. Set DEBUG to 1 to run
# formulas over images sequentially to see debug errors more clearly
DEVNULL = open(os.devnull, "w")



# Different settings used to render images
# in format key: [skeleton, rendering_call]
#   - skeleton is the LaTeX code in which formula is inserted 
#     (see BASIC_SKELETON)
#   - rendering_call is the system call made to turn .tex into .png
# Each rendering setup is done for each formula.
# key/name is used to identify different renderings in dataset file

#RENDERING_SETUPS = {"basic": [BASIC_SKELETON, "./textogif -png -dpi 200 %s"]}
RENDERING_SETUPS = {"basic": [basic,
                              {'fmt': 'png', 'dpi': 200, 'grayscale': False}],
                    "template_1": [template_1,
                              {'fmt': 'png', 'dpi': 200, 'grayscale': False}],
                    "template_2": [template_2,
                              {'fmt': 'png', 'dpi': 200, 'grayscale': False}],
                    "template_3": [template_3,
                              {'fmt': 'png', 'dpi': 200, 'grayscale': False}],
                    "template_4": [template_4,
                              {'fmt': 'png', 'dpi': 200, 'grayscale': False}],
                    "template_5": [template_5,
                              {'fmt': 'png', 'dpi': 200, 'grayscale': False}],
                    "template_6": [template_6,
                              {'fmt': 'png', 'dpi': 200, 'grayscale': False}]
                   }

def parser():
    p = argparse.ArgumentParser(description="Tool to render latex formulas")
    p.add_argument("--task", "-t", required=True, choices=['render', 'crop', 'csv'], help="render: rendering the latex formulae\ncrop: crop the images\ncsv: create the csv file", nargs='+')
    p.add_argument("--csv_file", "-c", help="input csv file with the latex formula")
    p.add_argument("--start", default=0, help="start point for rendering")
    p.add_argument("--stop", default=9999999, help="stop point for rendering")
    p.add_argument("--image-raw-path", "-ir", default=None, help="image raw path")
    p.add_argument("--image-processed-path", "-ip", default=None, help="image processed path")
    return p.parse_args()

class create_im2latex_more_templates():
    def __init__(self, task, csv_file, start=0, stop=9999999, image_raw_path=None, image_processed_path=None,
                 debug=False, num_cpus=14, rendering_setups=RENDERING_SETUPS):
        self.task = task
        self.csv_file = csv_file
        self.start = start
        self.stop = stop
        self.image_raw_path = image_raw_path
        self.image_processed_path = image_processed_path
        self.rendering_setups = rendering_setups
        self.debug = debug
        self.num_cpus = num_cpus
        # find already done formulas:
        self.already_rendered = []
        for file in Path(self.image_raw_path).glob("*.png"):
            self.already_rendered.append(file.name)
        self.already_cropped = []
        for file in Path(self.image_processed_path).glob("*.png"):
            self.already_cropped.append(file.name)

        if 'render' in self.task:
            print("start render formula images")
            self.render_formula()
        if 'crop' in self.task:
            print("start crop formula images")
            self.crop_formula()
        if 'csv' in self.task:
            print("start creating csv")
            self.create_csv()
    def remove_temp_files(self, name):
        """ Removes .aux, .log, .pdf and .tex files for name """
        os.remove(name+".aux")
        os.remove(name+".log")
        os.remove(name+".pdf")
        os.remove(name+".tex")
        if os.path.exists(name + ".out"):
            os.remove(name + ".out")
        if os.path.exists(name + "Notes.bib"):
            os.remove(name + "Notes.bib")

    def formula_to_image(self, input, rendering_setups):
        """ Turns given formula into images based on RENDERING_SETUPS
        returns list of lists [[image_name, rendering_setup], ...], one list for
        each rendering.
        Return None if couldn't render the formula"""
        formula = input[0]
        formula = formula.strip("%")
        name = input[1][:-4]
        done_images = ["_".join(image.split("_")[1:])[:-4] for image in self.already_rendered if name + "_" in image] + \
                      ["_".join(image.split("_")[1:])[:-4] for image in self.already_cropped if name + "_" in image]
        new_rendering_setups = {}
        for rend_name, rend_setup in rendering_setups.items():
            if not rend_name in done_images:
                new_rendering_setups[rend_name] = rend_setup

        for rend_name, rend_setup in new_rendering_setups.items():
            full_path = name+"_"+rend_name
            if full_path + ".png" in self.already_rendered or full_path + ".png" in self.already_cropped:
                continue
            # Create latex source
            latex = rend_setup[0] % formula.replace(" ", "")
            # Write latex source
            with open(full_path+".tex", "w") as f:
                f.write(latex)

            # Call pdflatex to turn .tex into .pdf
            code = subprocess.Popen(["pdflatex", '-interaction=nonstopmode', full_path+".tex"],
                        stdout=DEVNULL, stderr=DEVNULL)
            try:
                code.wait(timeout=300)
            except subprocess.TimeoutExpired:
                code.kill()
            if not Path(full_path + ".pdf").exists():
                os.system("rm -rf "+full_path+"*")
                self.failed += 1
                continue
            # Turn .pdf to .png
            try:
                output = pdf2image.convert_from_path(full_path + ".pdf", output_folder=".", fmt=rend_setup[1]['fmt'],
                                                     dpi=rend_setup[1]['dpi'], output_file=full_path,
                                                     grayscale=rend_setup[1]['grayscale'])
            except pdf2image.exceptions.PDFPageCountError:
                os.system("rm -rf " + full_path + "*")
                self.failed += 1
                continue
            os.rename(output[0].filename[2:], full_path + ".png")

            #Remove files
            try:
                self.remove_temp_files(full_path)
            except Exception as e:
                # try-except in case one of the previous scripts removes these files
                # already
                print("Could not remove files")

            # Detect of convert created multiple images -> multi-page PDF
            resulted_images = glob.glob(full_path+"-*")

            if len(resulted_images) > 1:
                # We have multiple images for same formula
                # Discard result and remove files
                for filename in resulted_images:
                    os.system("rm -rf "+filename+"*")
                self.failed += 1
                continue
        return

    def formula_to_image_single(self, input, rendering_setups):
        """ Turns given formula into images based on RENDERING_SETUPS
        returns list of lists [[image_name, rendering_setup], ...], one list for
        each rendering.
        Return None if couldn't render the formula"""
        formula = input[0]
        formula = formula.strip("%")
        name = input[1][:-4]
        rend_name = list(rendering_setups.keys())[0]
        rend_setup = rendering_setups[rend_name]
        full_path = name+"_"+rend_name
        # Create latex source
        latex = rend_setup[0] % formula
        # Write latex source
        with open(full_path+".tex", "w") as f:
            f.write(latex)

        # Call pdflatex to turn .tex into .pdf
        code = subprocess.Popen(["pdflatex", '-interaction=nonstopmode', full_path+".tex"],
                    stdout=DEVNULL, stderr=DEVNULL)
        try:
            code.wait(timeout=60)
        except subprocess.TimeoutExpired:
            code.kill()
        if not Path(full_path + ".pdf").exists():
            os.system("rm -rf "+full_path+"*")
            self.failed += 1
            return
        # Turn .pdf to .png
        try:
            output = pdf2image.convert_from_path(full_path + ".pdf", output_folder=".", fmt=rend_setup[1]['fmt'],
                                                 dpi=rend_setup[1]['dpi'], output_file=full_path,
                                                 grayscale=rend_setup[1]['grayscale'])
        except pdf2image.exceptions.PDFPageCountError:
            os.system("rm -rf " + full_path + "*")
            self.failed += 1
            return
        os.rename(output[0].filename[2:], full_path + ".png")

        #Remove files
        try:
            self.remove_temp_files(full_path)
        except Exception as e:
            # try-except in case one of the previous scripts removes these files
            # already
            print("Could not remove files")

        # Detect of convert created multiple images -> multi-page PDF
        resulted_images = glob.glob(full_path+"-*")

        if len(resulted_images) > 1:
            # We have multiple images for same formula
            # Discard result and remove files
            for filename in resulted_images:
                os.system("rm -rf "+filename+"*")
            self.failed += 1
        return

    def create_csv(self):
        formula_file = Path(self.csv_file)
        with open(formula_file, "r") as f:
            reader = csv.reader(f)
            formulas = {rows[1]: rows[0] for rows in reader}
        new_formula_file = formula_file.parent / formula_file.name.replace(".csv", "_new.csv")
        with open(new_formula_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(['formula', 'image'])
            not_formula = 0
            not_template = 0
            pbar = tqdm(formulas.items(), postfix=f"csv, no formula {not_formula}, no image {not_template}")
            for image_name, formula in pbar:
                if image_name == 'image':
                    continue
                writer.writerow([formula, image_name])
                images = sorted(Path(self.image_processed_path).glob(f"{image_name[:-4]}_*.png"))
                if len(images) == 0:
                    not_formula += 1
                    print(image_name)
                not_template += (len(RENDERING_SETUPS.items()) - len(images))
                for image in images:
                    writer.writerow([formula, image.name])
                pbar.postfix = f"csv, no formula {not_formula}, no image {not_template}"

    def crop_formula(self):
        processed_image_dir = Path(self.image_processed_path)
        processed_image_dir.mkdir(exist_ok=True, parents=True)
        postfix = ".png"
        crop_blank_default_size = [600, 60]
        pad_size = [8, 8, 8, 8]
        buckets = [[240, 100], [320, 80], [400, 80], [400, 100], [480, 80], [480, 100], [560, 80], [560, 100], [640, 80],
                   [640, 100], [720, 80], [720, 100], [720, 120], [720, 200], [800, 100], [800, 320], [1000, 200],
                   [1000, 400], [1200, 200], [1600, 200], [1600, 1600]]
        downsample_ratio = 1.0
        images = sorted(Path(self.image_raw_path).glob("*.png"))
        images_new = []
        pbar = tqdm(images, postfix="checking already cropped images")
        for image in pbar:
            if image.name not in self.already_cropped and image.name.split('.')[0][-6:] != '0001-1':
                images_new.append(image)
        print(f"croping {len(images_new)} images")
        if self.debug:
            pbar = tqdm(images_new, postfix=f'Process Images')
            for image in pbar:
                main_parallel(image, postfix=postfix, output_folder=processed_image_dir,
                              crop_blank_default_size=crop_blank_default_size, pad_size=pad_size, buckets=buckets,
                              downsample_ratio=downsample_ratio)
        else:
            p_umap(partial(main_parallel, postfix=postfix, output_folder=processed_image_dir,
                           crop_blank_default_size=crop_blank_default_size, pad_size=pad_size, buckets=buckets,
                           downsample_ratio=downsample_ratio), images_new,
                   **{'postfix': f'Process Images', 'num_cpus': self.num_cpus})

    def render_formula(self):
        formulas = []
        with open(self.csv_file) as f:
            reader = csv.reader(f)
            pbar = tqdm(list(reader), postfix="load formula")
            for line in pbar:
                formulas.append(line)
        formulas.pop(0)
        print(f'formulas from {self.start} to {min(int(self.stop), len(formulas))}')
        formulas = formulas[int(self.start):min(int(self.stop), len(formulas))]

        image_dir = Path(self.image_raw_path)
        image_dir.mkdir(exist_ok=True, parents=True)
        for file in Path("preprocessing/templates/resources").iterdir():
            if "DS_Store" in file.name:
                continue
            shutil.copy(file, image_dir / file.name)



        print("Turning formulas into images...")

        # Change to image dir because textogif doesn't seem to work otherwise...
        oldcwd = os.getcwd()
        # Check we are not in image dir yet (avoid exceptions)
        if not str(image_dir) in os.getcwd():
            os.chdir(image_dir)

        self.failed = 0

        if len(list(self.rendering_setups)) > 1:
            if self.debug:
                pbar = tqdm(formulas, postfix=f'Compile formulae, failed {self.failed}')
                for formula in pbar:
                    self.formula_to_image(formula, rendering_setups=self.rendering_setups)
                    pbar.postfix = f'Compile formulae, failed {self.failed}'
            else:
                p_umap(partial(self.formula_to_image, rendering_setups=self.rendering_setups), formulas,**{'postfix': f'Compile formulae, failed {self.failed}', 'num_cpus': 14})
                print(f'Compile formulae, failed {self.failed}')
        else:
            done_images = [image.split("_")[0] for image in self.already_rendered] + [image.split("_")[0] for image in self.already_cropped]
            new_formulas = []
            for formula in formulas:
                if not formula[1][:-4] in done_images:
                    new_formulas.append(formula)
            if self.debug:
                pbar = tqdm(new_formulas, postfix=f'Compile formulae, failed {self.failed}')
                for formula in pbar:
                    self.formula_to_image_single(formula, rendering_setups=self.rendering_setups)
                    pbar.postfix = f'Compile formulae, failed {self.failed}'
            else:
                p_umap(partial(self.formula_to_image_single, rendering_setups=self.rendering_setups), new_formulas,
                       **{'postfix': f'Compile formulae, failed {self.failed}', 'num_cpus': 14})
                print(f'Compile formulae, failed {self.failed}')

        os.chdir(oldcwd)

if __name__ == '__main__':
    args = parser()
    if not args.image_raw_path:
        args.image_raw_path = (Path(args.csv_file).parent/"formula_images")
    if not args.image_processed_path:
        args.image_processed_path = (Path(args.csv_file).parent/"images_processed")
    dataset = create_im2latex_more_templates(task=args.task, csv_file=args.csv_file, start=args.start, stop=args.stop,
                                             image_raw_path=args.image_raw_path,
                                             image_processed_path=args.image_processed_path, debug=False)

    """
    from preprocessing.formula2image import create_im2latex_more_templates
    from pathlib import Path
    csv_file = "preprocessing/data/im2latex-kaggle-more-templates/im2latex_test.csv"
    create_im2latex_more_templates(["render", "crop"], csv_file , image_raw_path=Path(csv_file).parent/"formula_images", image_processed_path=Path(csv_file).parent/"images_processed")
    """



