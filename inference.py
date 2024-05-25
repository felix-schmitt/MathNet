import torchvision

import model
from tools.data_utils import LatexVocab
import torch
import numpy as np
import cv2
from tools.utils import load_config

class Inference:
    def __init__(self, config, model_path):
        self.config = config
        self.vocab = LatexVocab(self.config['dataset']['vocab_file'])
        self.config['model']['vocab_size'] = len(self.vocab.id2token)
        self.model = model.model(self.config, self.vocab, None)
        self.model.eval()
        try:
            self.ckpt = torch.load(model_path)
            self.model.decoder.load_state_dict(self.ckpt['decoder_model'], strict=True)
            self.model.encoder.load_state_dict(self.ckpt['encoder_model'], strict=True)
            print("load model from checkpoint")
        except Exception as e:
            print('Error occurred during loading ', self.config['arguments']['resume_from'], ' (', e, ')')
        self.width = config['model']['image']['width']
        self.height = config['model']['image']['height']
        self.channels = 1

    def forward(self, image):
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape([image.shape[0], image.shape[1], 1])
        image = image.transpose(2, 0, 1)
        image = image / 255.0  # normalize
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).to(torch.float32).to(self.config['device'])
        image = self.white_border(image)
        image = 1 - image
        image = image[None, :]
        labels = torch.from_numpy(np.array([[self.vocab.token2id['_START_'], self.vocab.token2id['_END_']]])).half().to(torch.float).to(self.config['device'])
        outputs = self.model(image.to(self.config['device']), labels, 0)
        return self.vocab.seq2text(outputs[0][0].cpu().numpy())

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

if __name__ == '__main__':
    model_files = {
        'config': "trainedModels/run_2023-3-31_12-52-19/config.yml",
        'ckpt': "trainedModels/run_2023-3-31_12-52-19/best.pt",
        "vocab": "trainedModels/run_2023-3-31_12-52-19/latex.vocab"
    }
    config = load_config(model_files['config'], False)
    config['dataset']['vocab_file'] = model_files['vocab']
    image = cv2.imread('data/im2latexv2/basic/test/img2-grey/1a0aa88013_basic.png')  # image for inference
    model = Inference(config, model_files['ckpt'])
    altText = model.forward(image)
    print(altText)