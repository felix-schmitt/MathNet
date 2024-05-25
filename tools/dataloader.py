import torch
from pathlib import Path
import csv
import yaml
from tools.data_utils import LatexVocab
import cv2
import numpy as np
import torchvision
from tqdm import tqdm
from preprocessing.improve_tokens import improve_tokens
import PIL

class Im2LatexDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_config_files, vocab_path=None, image_size=None, mode='train', cache=False, max_size=150,
                 transforms=[], no_sampling=False, no_arrays=False, only_basic=False, normalize=True, dpi=200):
        super(Im2LatexDataset, self).__init__()
        self.data_list = []
        self.formulas = []
        self.vocab = None
        self.max_size = max_size
        self.drop_styles = []
        self.styles = []
        self.normalize = normalize
        self.dpi = dpi
        self.pil_transform = torchvision.transforms.ToPILImage()
        for dataset_config_file in dataset_config_files:
            with open(f"configs/datasets/{dataset_config_file}") as f:
                self.dataset_config = yaml.safe_load(f)
                if 'drop_styles' in self.dataset_config:
                    self.drop_styles = self.dataset_config['drop_styles']
            datasets = self.dataset_config[mode]
            for dataset in datasets:
                if self.normalize:
                    improve_tokens([str(Path(self.dataset_config['dataset_path']) / dataset[1])], file_ending="_normalized.csv", use_only_c=True)
                    dataset[1] = dataset[1].replace(".csv", "_normalized.csv")
                with open(Path(self.dataset_config['dataset_path']) / dataset[1], 'r') as f:
                    data = csv.reader(f)
                    for i, row in tqdm(enumerate(data), postfix="load dataset"):
                        if i == 0 and row[0].replace(" ", "") == "formula":
                            continue
                        if no_arrays and "\\begin{array}" in row[0]:
                            continue
                        if only_basic:
                            row = [row[0]] + [image for image in row[1:] if (image.split("/")[0] == 'basic')]
                        row = [row[0]] + [image for image in row[1:] if not (image.split("/")[0] in self.drop_styles)]
                        styles = [image.split("/")[0] for image in row[1:] if not (image.split("/")[0] in self.drop_styles)]
                        for style in styles:
                            if style not in self.styles:
                                self.styles.append(style)
                        if len(row) == 1:
                            continue
                        if no_sampling:
                            self.formulas += [[row[0], [str(Path(self.dataset_config['dataset_path']) / dataset[0] / image)]] for image in row[1:]]
                        else:
                            self.formulas.append([row[0], [str(Path(self.dataset_config['dataset_path']) / dataset[0] / image) for image in row[1:]]])
        if vocab_path is not None:
            self.vocab = LatexVocab(vocab_path)
        self.use_cache = cache
        self.transforms = []
        if self.use_cache:
            self.cached = [None] * len(self.formulas)
            self.transforms = [self.random_selection]
        self.resize_factor = self.dpi / self.dataset_config['dpi']
        self.transforms.append(self.resize)
        if "AddWhitespace" in transforms:
            self.transforms.append(self.add_whitespace)
        if "RandomResize" in transforms:
            self.transforms.append(self.random_resize)
        if "WhiteBorder" in transforms:
            self.transforms.append(self.white_border)
        if "DownUpResize" in transforms:
            self.transforms.append(self.randomDownUpResize)
        if "GaussianBlur" in transforms:
            self.transforms.append(torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)))
        if "ColorJitter" in transforms:
            self.transforms.append(torchvision.transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=0))
        if "AdjustSharpness" in transforms:
            self.transforms.append(self.adjust_sharpness)
        if "White2Black" in transforms:
            self.transforms.append(self.white2black)
        self.height = image_size['height']
        self.width = image_size['width']
        self.channels = 1
        self.min = 0.8
        self.max = 2.0
        self.number = 1
        self.whitespace_samples = 10
        self.whitespace_max_size = 300

    def cache_all_files(self):
        for idx in tqdm(range(0, len(self.formulas)), postfix="cache dataset"):
            if not self.cached[idx]:
                formula_seq, image, image_name = self._load_all_images(idx)
                self.cached[idx] = [formula_seq, image, image_name]

    def _load_all_images(self, idx):
        formula, image_names = self.formulas[idx]
        if self.vocab:
            formula_seq = np.array(self.vocab.text2seq(formula, self.max_size))
        else:
            formula_seq = np.array([])
        image = []
        for image_name in image_names:
            img = cv2.imread(image_name)
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img.reshape([img.shape[0], img.shape[1], 1])
            img = img.transpose(2, 0, 1)
            img = img / 255.0  # normalize
            img = np.ascontiguousarray(img, dtype=np.float32)
            img = torch.from_numpy(img).to(torch.float32)
            image.append(img)
        image_name = image_names[0].split("/")[-1].split("_")[0] + ".png"
        style = [image_name.split("/")[-1] for image_name in image_names]
        return formula_seq, image, image_name, style

    def _load_one_image(self, idx):
        formula, image_names = self.formulas[idx]
        if self.vocab:
            formula_seq = np.array(self.vocab.text2seq(formula, self.max_size))
        else:
            formula_seq = np.array([])
        image_name = np.random.choice(image_names)
        image = cv2.imread(image_name)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape([image.shape[0], image.shape[1], 1])
        image = image.transpose(2, 0, 1)
        image = image / 255.0  # normalize
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).to(torch.float32)
        image_name = image_names[0].split("/")[-1].split("_")[0] + ".png"
        return formula_seq, image, image_name

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx):
        if self.use_cache:
            if self.cached[idx]:
                formula_seq, image, image_name = self.cached[idx]
            else:
                formula_seq, image, image_name = self._load_all_images(idx)
                self.cached[idx] = [formula_seq, image, image_name]
        else:
            formula_seq, image, image_name = self._load_one_image(idx)
        for transform in self.transforms:
            image = transform(image)
        formula_seq = torch.from_numpy(formula_seq)
        return image, ([image_name, idx], formula_seq)

    def random_selection(self, x):
        indice = torch.randint(0, len(x), (self.number, 1))
        if len(indice) == 1:
            x = x[indice[0]]
        else:
            x = x[indice]
        return x

    def add_whitespace(self, x):
        white = torch.mean(x, axis=2)
        c, h, _ = x.size()
        for i in torch.randint(0, white.size()[1], [self.whitespace_samples]):
            if white[0, i] == 1:
                x = torch.cat((x[:, :, :i], torch.ones(c, h, torch.randint(1, self.whitespace_max_size, [1])[0]), x[:, :, i:]), dim=-1)
        return x

    def random_resize(self, x):
        channel, height, width = x.shape
        if height > self.height or width > self.width:
            ratio = min(self.height/height, self.width/width)
        else:
            ratio = self.min + torch.rand(1)*(min(self.max, self.height/height, self.width/width)-self.min)
        x = torchvision.transforms.functional.resize(x, (int(ratio*height), int(ratio*width)),
                                                     torchvision.transforms.functional.InterpolationMode.BICUBIC,
                                                     antialias=False)
        return x

    def white_border(self, x):
        image = torch.ones((self.channels, self.height, self.width))
        channel, height, width = x.shape
        if width > self.width or height > self.height:
            ratio = min(self.height/height, self.width/width)
            x = torchvision.transforms.functional.resize(x, (int(ratio * height), int(ratio * width)), antialias=False)
            channel, height, width = x.shape
        if width < self.width:
            x_indice = torch.randint(0, self.width-width, (1, 1))[0]
        else:
            x_indice = 0
        if height < self.height:
            y_indice = torch.randint(0, self.height - height, (1, 1))[0]
        else:
            y_indice = 0
        image[:, y_indice:y_indice+height, x_indice:x_indice+width] = x
        return image

    def white2black(self, x):
        return 1 - x

    def resize(self, x):
        channel, height, width = x.shape
        return torchvision.transforms.functional.resize(x, (int(self.resize_factor * height), int(self.resize_factor * width)),
                                                        torchvision.transforms.functional.InterpolationMode.BICUBIC,
                                                        antialias=False)

    def adjust_sharpness(self, x):
        return torchvision.transforms.functional.adjust_sharpness(x, 2)

    def randomDownUpResize(self, x):
        if self.dpi > 100:
            resize_factor = torch.randint(100, self.dpi, (1, 1))[0] / self.dpi
        else:
            resize_factor = 1
        channel, height, width = x.shape
        x = torchvision.transforms.functional.resize(x, (int(resize_factor * height), int(resize_factor * width)),
                                                     torchvision.transforms.functional.InterpolationMode.BICUBIC,
                                                     antialias=False)
        return torchvision.transforms.functional.resize(x, (int(height), int(width)),
                                                        torchvision.transforms.functional.InterpolationMode.BICUBIC,
                                                        antialias=False)
