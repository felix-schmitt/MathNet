import numpy as np

import torch
import pathlib
import yaml
import os
import cv2
import pynvml
import csv
from pathlib import Path


def draw_grid(img, line_color=(0, 255, 0), thickness=1, type_=cv2.LINE_AA, pxstep=40):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    img = img.copy()
    x = pxstep
    y = pxstep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep
    return img


class LatexVocab(object):
    def __init__(self, vocab_path):
        token2id = {"_START_": 0, "_PAD_": 1, "_END_": 2, "_UNK_": 3}
        id2token = {0: "_START_", 1: "_PAD_", 2: "_END_", 3: "_UNK_"}
        k = len(id2token)
        with open(vocab_path, 'r') as f:
            line = f.readline()
            while line:
                token = line.split('\n')[0]
                if token in token2id.keys():
                    line = f.readline()
                    continue
                token2id[token] = k
                id2token[k] = token
                line = f.readline()
                k += 1
        self.id2token = id2token
        self.token2id = token2id

    def text2seq(self, text, max_size=None):
        text = text.split(' ')
        seq = []
        for ch in text:
            try:
                seq.append(self.token2id[ch])
            except:
                seq.append(self.token2id['_UNK_'])
        seq = [self.token2id['_START_']] + seq + [self.token2id['_END_']]
        if max_size is not None:
            temp = [self.token2id['_PAD_']] * max_size
            temp[:len(seq)] = seq
            seq = temp
            seq = seq[:max_size]
        return seq

    def seq2text(self, seq):
        seq = [self.id2token[id] for id in seq]
        text = ''
        for s in seq:
            if s == '_START_':
                continue
            if s == '_END_' or s == '_PAD_':
                break
            text += s + ' '
        return text[:-1]

    def __len__(self):
        return len(self.id2token)


def bind_objects(frame, thresh_img, minArea, plot=False):
    '''Draws bounding boxes and detects when cars are crossing the line
    frame: numpy image where boxes will be drawn onto
    thresh_img: numpy image after subtracting the background and all thresholds and noise reduction operations are applied
    '''
    cnts, _ = cv2.findContours(thresh_img, 1,
                               2)  # this line is for opencv 2.4, and also now for OpenCV 4.4, so this is the current one
    # cnts = sorted(cnts,key = cv2.contourArea,reverse=False)
    # frame = cv2.drawContours(frame, cnts, -1, (0,255,0), 3)
    cnt_id = 1
    cur_centroids = []
    boxes = []
    for c in cnts:
        if cv2.contourArea(c) < minArea:  # ignore contours that are smaller than this area
            continue

        x, y, w, h = cv2.boundingRect(c)

        box = np.array([x, y, x + w, y + h], int)
        boxes.append(box)
        if plot:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)
            cv2.rectangle(thresh_img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)
    boxes = np.array(boxes)
    return boxes


class Resize(object):
    def __init__(self, size, jitter=False, random_resize=False, prob=0.5):
        self.jitter = jitter
        self.random_resize = random_resize
        self.p = prob
        self.size = size if isinstance(size, tuple) else tuple(size)

    def __call__(self, x):
        temp = 255 * np.ones((self.size[0], self.size[1], 3), dtype=x.dtype)
        h, w, c = x.shape
        r = max([w / self.size[1], h / self.size[0]])
        if np.random.uniform(0., 1.) < self.p and r < 1 and self.random_resize:
            r = 1
        x = cv2.resize(x, (int(w / r), int(h / r)))
        px = self.size[1] - int(w / r)
        py = self.size[0] - int(h / r)

        jx = np.random.uniform(-px, px)
        jy = np.random.uniform(-py, py)
        if not self.jitter:
            jx = 0
            jy = 0
        jx = int(jx // 2)
        jy = int(jy // 2)
        temp[py // 2 - jy:py // 2 - jy + x.shape[0], px // 2 - jx:px // 2 - jx + x.shape[1], :] = x
        return temp


class Im2LatexDataset_csv(torch.utils.data.Dataset):
    def __init__(self, dataset_config_files, vocab_path=None, max_size=None, transforms=None,
                 num_caches=1000, mode='train'):
        super(Im2LatexDataset_csv, self).__init__()
        self.data_list = []
        self.formulas = []
        self.vocab = None
        self.max_len = 0
        for dataset_config_file in dataset_config_files:
            with open(f"configs/datasets/{dataset_config_file}") as f:
                dataset_config = yaml.safe_load(f)
            datasets = dataset_config[mode]
            for dataset in datasets:
                with open(Path(dataset_config['dataset_path']) / dataset[1], 'r') as f:
                    data = csv.reader(f)
                    for i, [formula, image] in enumerate(data):
                        if i == 0:
                            continue
                        self.formulas.append([formula, str(Path(dataset_config['dataset_path']) / dataset[0] / image)])
        if vocab_path is not None:
            self.vocab = LatexVocab(vocab_path)
        self.max_size = max_size if max_size > 0 else self.max_len

        self.num_caches_limits = num_caches
        if self.num_caches_limits:
            self.cached_images = [None] * len(self.formulas)
            self.cached_labels = [None] * len(self.formulas)
        self.cached = 0
        self.transforms = transforms

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx):
        if self.num_caches_limits and self.cached_images[idx] is not None:
            img = self.cached_images[idx]
            formula_txt, formula_seq = self.cached_labels[idx]
        else:
            formula, image_name = self.formulas[idx]
            formula_seq = np.array(self.vocab.text2seq(formula, self.max_size))
            img = cv2.imread(image_name)

            if self.cached < self.num_caches_limits:
                self.cached_images[idx] = img
                self.cached_labels[idx] = (formula, formula_seq)
                self.cached += 1
        if self.transforms is not None:
            for trf in self.transforms:
                img = trf(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape([img.shape[0], img.shape[1], 1])
        img = img.transpose(2, 0, 1)  #
        img = np.ascontiguousarray(img)
        img = np.float32(img)
        img = torch.from_numpy(img).to(torch.float32)
        formula, image_name = self.formulas[idx]
        formula_seq = torch.from_numpy(formula_seq)
        return img, ([image_name, idx], formula_seq)


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], scale=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.scale = np.array(scale)

    def __call__(self, x):
        x = np.float32(x) / 255.0
        return x


class RandomBGR2GRAY(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if np.random.uniform(0., 1.0) < self.p:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        return x


class RandomCrop(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if np.random.uniform(0, 1) > self.p:
            return img
        h = img.shape[0]
        w = img.shape[1]
        xmin = w
        ymin = h
        xmax = 0
        ymax = 0
        for x in range(w):
            if np.min(img[:, w - x - 1]):
                xmin = x
                break
        for x in range(w):
            if np.min(img[:, w - x - 1]) < 100:
                xmax = w - x - 1
                break
        for y in range(h):
            if np.min(img[y, :]) < 100:
                ymin = y
                break
        for y in range(h):
            if np.min(img[h - y - 1, :]) < 100:
                ymax = h - y - 1
                break
        xmin -= 4
        xmax += 4
        ymax += 4
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > w:
            xmax = w - 1
        if ymax > h:
            ymax = h - 1
        try:
            cropped = img[ymin:ymax, xmin:xmax]
        except:
            cropped = img

        return cropped
