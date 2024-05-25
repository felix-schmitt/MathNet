import csv
import shutil

import tabulate
from preprocessing.formula2image import create_im2latex_more_templates
from preprocessing.templates import templates2
from pathlib import Path
import argparse
import tarfile
import os
import cv2
from p_tqdm import p_umap
from tqdm import tqdm


def crop_better(img):
    if (new_img_folder / img.name).exists():
        return
    image = cv2.imread(str(img))
    temp_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Top
    height, width = temp_img.shape[:2]
    top = 0
    bottom = height
    right = width
    left = 0
    for i in range(0, height - 1):
        temp = temp_img[i:i + 1, :]
        if all((temp > 250)[0]):
            top = i + 1
        else:
            break
    for i in range(0, height - 1):
        temp = temp_img[height - i - 1:height - i, :]
        if all((temp > 250)[0]):
            bottom = height - i - 1
        else:
            break
    for i in range(0, width - 1):
        temp = temp_img[:, i:i + 1]
        if all((temp > 250)):
            left = i + 1
        else:
            break
    for i in range(0, width - 1):
        temp = temp_img[:, width - i - 1:width - i]
        if all((temp > 250)):
            right = width - i - 1
        else:
            break
    top = max(top - 2, 0)
    bottom = min(bottom + 2, height)
    left = max(left - 2, 0)
    right = min(right + 2, width)
    if top >= bottom:
        return
    if left >= right:
        return
    img_new = image[top:bottom, left:right, :]
    cv2.imwrite(str(new_img_folder / img.name), img_new)


def convert_into_grey(img):
    """if (new_img_folder / img.name).exists():
        return"""
    image = cv2.imread(str(img))
    cv2.imwrite(str(new_img_folder / img.name), cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Tool to create contrastive dataset")
    p.add_argument("--start", default=0, help="start rendering setup for rendering", type=int)
    p.add_argument("--stop", default=9999999, help="stop rendering setup", type=int)
    p.add_argument("--num-cpus", default=14, help="number of cpus", type=int)
    p.add_argument("--tasks", default=['render', 'crop', 'cropBetter', 'convertIntoGrey',  'tar', 'delete'], help="tasks to do", nargs='+')
    p.add_argument("--debug", default=False, help="start debug mode")
    p.add_argument("--styles", default=[], help='styles to do', nargs='+')
    p.add_argument("--folder", default="data/new-61styles", help="folder")
    p.add_argument("--file", default="im2latex_train.csv", help='filename')
    p.add_argument("--output-folder", default="train", help="output folder name")
    config = p.parse_args()
    folder = Path(config.folder)
    file = config.file
    options = templates2.__dict__
    options = {option: options[option] for option in options.keys() if not option[:2] == "__"}
    option_keys = list(options.keys())[config.start:config.stop]
    new_options = {}
    if config.styles:
        for style in config.styles:
            new_options[style] = options[style]
    else:
        for option in option_keys:
            new_options[option] = options[option]
    if 'render' in config.tasks or 'crop' in config.tasks:
        for option in new_options.keys():
            print(f"doing rendering setup {option}")
            rendering_setup = {option: [options[option], {'fmt': 'png', 'dpi': 600, 'grayscale': False}]}

            create_im2latex_more_templates(task=config.tasks, csv_file=folder/file,
                                           image_raw_path=folder/f"{option}/{config.output_folder}/formula_image",
                                           image_processed_path=folder/f"{option}/{config.output_folder}/img",
                                           num_cpus=config.num_cpus, rendering_setups=rendering_setup, debug=config.debug)

    if 'cropBetter' in config.tasks:
        folders = list(folder.glob("*"))
        folders_name = {folder.name: folder_i for folder_i, folder in enumerate(folders)}
        for folder in folders:
            if folder.name in new_options.keys():
                img_folder = folder / f"{config.output_folder}/img"
                new_img_folder = img_folder.parent / "img2"
                new_img_folder.mkdir(exist_ok=True, parents=True)
                if config.debug:
                    for image in tqdm(list(img_folder.glob("*.png")), postfix=f"crop images {folder.name}"):
                        crop_better(image)
                else:
                    p_umap(crop_better, list(img_folder.glob("*.png")), **{'postfix': f'Crop images {folder.name}', 'num_cpus': config.num_cpus})

    if 'convertIntoGrey' in config.tasks:
        folders = list(folder.glob("*"))
        folders_name = {folder.name: folder_i for folder_i, folder in enumerate(folders)}
        for folder in folders:
            if folder.name in new_options.keys():
                img_folder = folder / f"{config.output_folder}/img2"
                new_img_folder = img_folder.parent / "img2-grey"
                new_img_folder.mkdir(exist_ok=True, parents=True)
                if config.debug:
                    for image in tqdm(list(img_folder.glob("*.png")), postfix=f'Convert into grey {folder.name}'):
                        convert_into_grey(image)
                else:
                    p_umap(convert_into_grey, list(img_folder.glob("*.png")), **{'postfix': f'Convert into grey {folder.name}', 'num_cpus': config.num_cpus})

    if 'analyze' in config.tasks:
        formulas = {}
        with open(folder/file) as f:
            reader = csv.reader(f)
            for line in reader:
                formulas[line[1].replace(".png", "")] = line[0]
        formulas.pop("image")
        results = []
        for style in folder.glob("*/"):
            if style.name not in new_options:
                continue
            images = list((style / f"{config.output_folder}/img2-grey").glob("*.png"))
            with open(style / f"{config.output_folder}/train2_modified_tokens.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(['formula', 'image'])
                for image in tqdm(images, postfix=f"list images of {style}"):
                    writer.writerow([formulas[image.name.replace(".png", "").split("_")[0]], image.name])
            results.append([style, len(images)])
        print("results")
        print(tabulate.tabulate(results))
        with open(folder / "results.csv", "w") as f:
            writer = csv.writer(f)
            for line in results:
                writer.writerow(line)

    if 'analyze2' in config.tasks:
        image_folder = 'img2-grey'
        formulas = {}
        with open(folder / file) as f:
            reader = csv.reader(f)
            for line in reader:
                formulas[line[1].replace(".png", "")] = line[0]
        if 'image' in formulas.keys():
            formulas.pop("image")
        results = []
        grouped_images = {}
        for style in tqdm(folder.glob("*/"), postfix=f"load style"):
            if style.name not in new_options:
                continue
            images = list((style / f"{config.output_folder}/{image_folder}").glob("*.png"))
            for image in images:
                image_number = image.name.split("_")[0]
                if image_number in grouped_images.keys():
                    grouped_images[image_number].append(style.name + f"/{config.output_folder}/{image_folder}/{image.name}")
                else:
                    grouped_images[image_number] = [style.name + f"/{config.output_folder}/{image_folder}/{image.name}"]
        print("create csv file")
        with open(folder / "im2latexv2_validate_normalized2.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["formula", "images"])
            for key in grouped_images.keys():
                if key not in formulas.keys():
                    continue
                if len(grouped_images[key]) == 59:
                    writer.writerow([formulas[key]]+grouped_images[key])
        print("csv file created")

    def tarfolders(option):
        if option in folders_name:
            image_folder_path = folders[folders_name[option]] / f"{config.output_folder}/{image_folder}"
            if not image_folder_path.exists():
                return
            """if Path(f"{image_folder_path}.tar.gz").exists():
                return"""
            with tarfile.open(f"{image_folder_path}.tar.gz", "w:gz") as tar:
                tar.add(image_folder_path, arcname=os.path.basename(image_folder_path))

    if 'tar' in config.tasks:
        image_folder = "img2-grey"
        folders = list(folder.glob("*"))
        folders_name = {folder.name: folder_i for folder_i, folder in enumerate(folders)}
        if config.debug:
            for option in tqdm(new_options.keys(), postfix=f'tar folder {folder.name}'):
                tarfolders(option)
        else:
            p_umap(tarfolders, new_options.keys(),**{'postfix': f'tar folders', 'num_cpus': config.num_cpus})

    if 'untar' in config.tasks:
        image_folder = "img2-grey"
        folders = list(folder.glob("*"))
        folders_name = {folder.name: folder_i for folder_i, folder in enumerate(folders)}
        for option in new_options.keys():
            if option in folders_name:
                image_folder_path = folders[folders_name[option]]/f"{config.output_folder}/{image_folder}"
                print(f"untar {image_folder_path.parent.parent.name}")
                with tarfile.open(f"{image_folder_path}.tar.gz", "r:gz") as tar:
                    tar.extractall(image_folder_path.parent)
                print(f"{image_folder_path.parent.parent.name} extracted")
                continue

    if 'delete' in config.tasks:
        folders = list(folder.glob("*"))
        folders_name = {folder.name: folder_i for folder_i, folder in enumerate(folders)}
        for option in new_options.keys():
            if option in folders_name:
                image_folders = list((folders[folders_name[option]]/f"{config.output_folder}").glob("*"))
                for image_folder in image_folders:
                    if "tar.gz" not in image_folder.name and ".DS_Store" not in image_folder.name:
                        shutil.rmtree(image_folder)

    if 'add_style_grouped' in config.tasks:
        group_file = "grouped-grey.csv"
        grouped_images = {}
        image_folder = "img2-grey"
        with open(folder/group_file) as f:
            reader = csv.reader(f)
            for line in reader:
                image_number = line[1].split("/")[-1].split("_")[0]
                if image_number == "images":
                    continue
                grouped_images[image_number] = line
        formulas = {}
        with open(folder / file) as f:
            reader = csv.reader(f)
            for line in reader:
                formulas[line[1].replace(".png", "")] = line[0]
        for style in tqdm(folder.glob("*/"), postfix=f"load style"):
            if style.name not in new_options:
                continue
            images = list((style / f"{config.output_folder}/{image_folder}").glob("*.png"))
            for image in images:
                image_number = image.name.split("_")[0]
                if image_number in grouped_images.keys():
                    grouped_images[image_number].append(style.name + f"{config.output_folder}/{image_folder}/{image.name}")
                else:
                    grouped_images[image_number] = [formulas[image_number], style.name + f"{config.output_folder}/{image_folder}/{image.name}"]
        print("create csv file")
        with open(folder / group_file.replace(".csv", "_new.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(["formula", "images"])
            for key in grouped_images.keys():
                writer.writerow(grouped_images[key])
        print("csv file created")

    if 'update_formulas' in config.tasks:
        new_formula_file = "data/im2latexv2/im2latexv2_train_formulae_normalized2.csv"
        grouped_file = "data/im2latexv2/im2latexv2_train_normalized.csv"
        new_ending = "2.csv"
        new_formula = {}
        with open(new_formula_file) as f:
            reader = csv.reader(f)
            for formula, image in reader:
                new_formula[image] = formula

        with open(grouped_file) as f:
            reader = csv.reader(f)
            with open(grouped_file.replace(".csv", new_ending), "w") as ff:
                writer = csv.writer(ff)
                for line in reader:
                    formula, image = line[0], line[1].split("/")[-1].split("_")[0] + ".png"
                    if image in new_formula:
                        writer.writerow([new_formula[image]] + line[1:])

    if 'copyNewFolder' in config.tasks:
        new_folder = Path("data/61styles-v4-array")
        for folder in Path(config.folder).glob("*/"):
            for subfolder in folder.glob("*/"):
                if '-array' in subfolder.name:
                    new_subfolder = new_folder / f"{folder.name}"
                    new_subfolder.mkdir(exist_ok=True, parents=True)
                    shutil.move(str(subfolder), str(new_subfolder))
