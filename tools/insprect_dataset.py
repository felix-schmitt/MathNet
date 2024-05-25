import numpy as np

import torchvision
import matplotlib.pyplot as plt

from model import *
from tools.data_utils import Im2LatexDataset, Resize, Normalize, draw_grid, RandomCrop

DATASET_PATH = 'data/im2latex-100k-processed'
FORMULA_PATH = DATASET_PATH + '/im2latex_formulas.norm.lst'
IMAGES_PATH = DATASET_PATH + '/formula_images_processed'
VOCAB_PATH = '../latex.vocab'
TRAIN_PATH = DATASET_PATH + '/im2latex_train_filter.lst'
VAL_PATH = DATASET_PATH + '/im2latex_validate_filter.lst'
TEST_PATH = DATASET_PATH + '/im2latex_test_filter.lst'

IMG_HSIZE = 128
IMG_WSIZE = 256
BATCH_SIZE = 5
formulas = []
lens = []
with open(FORMULA_PATH, 'r', encoding="ISO-8859-1") as f:
    line = f.readline()
    while line:
        formulas.append(line.split('\n')[0])
        lens.append(len(line.split('\n')[0].split(' ')))

        line = f.readline()
formulas = np.array(formulas)
# %%
plt.hist(lens, bins=20)

formulas = []
with open(FORMULA_PATH, 'r', encoding="ISO-8859-1") as f:
    line = f.readline()
    while line:
        formulas.append(line.split('\n')[0])

        line = f.readline()
formulas = np.array(formulas)

train_data = []

with open(TRAIN_PATH, 'r', encoding="ISO-8859-1") as f:
    line = f.readline()
    while line:
        train_data.append(line.split('\n')[0])

        line = f.readline()
train_data = np.array(train_data)
print(train_data.shape)
train_data[1]

ds_train = Im2LatexDataset(IMAGES_PATH, TRAIN_PATH, FORMULA_PATH, VOCAB_PATH, MAX_LEN,
                           transforms=[RandomCrop(1.0), Resize([IMG_HSIZE, IMG_WSIZE], False, True, 1),
                                       Normalize()])  # create dataset
train_loader = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                                           num_workers=1)  # creat dataloader

images, labels = next(iter(train_loader))
grid_imgs = torchvision.utils.make_grid(images, nrow=4)
grid_imgs.size()
np_grid = grid_imgs.permute([1, 2, 0]).numpy()
plt.figure(figsize=(20, 20))
plt.axis('off')
plt.imshow(np_grid)
plt.show()

img = images[3].permute([1, 2, 0]).numpy()
plt.imshow(img, cmap='gray')

img2 = draw_grid(img, (1, 1, 1), pxstep=8)
plt.imshow(img2, cmap='gray')
