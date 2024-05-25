import torch
from torch import nn
import math
from model_parts.dcn import DeformableConv2d


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class CNNEncoder(nn.Module):
    def __init__(self, config, white_pixel_mask=False):
        super(CNNEncoder, self).__init__()
        self.cnn = CNNModel(config)
        self.white_pixels = white_pixel_mask

    def forward(self, imgs):
        features = self.cnn(imgs)
        if self.white_pixels:
            with torch.no_grad():
                white_pixel_value = torch.mode(torch.flatten(torch.sum(features, axis=-1))).values
                white_pixel_mask = torch.sum(features, axis=-1) == white_pixel_value
                white_pixel_mask = white_pixel_mask.view(white_pixel_mask.size(0), -1)
        positional_encoding = positionalencoding2d(features.size(3), features.size(1), features.size(2))
        positional_encoding = positional_encoding.unsqueeze(0)
        positional_encoding = positional_encoding.permute(0, 2, 3, 1).to(features.device)
        positional_encoding = features + positional_encoding
        positional_encoding = positional_encoding.view(positional_encoding.size(0), -1, positional_encoding.size(3))
        positional_encoding = positional_encoding.permute(1, 0, 2)
        if self.white_pixels:
            return positional_encoding, white_pixel_mask
        else:
            return positional_encoding

class CNNModel(nn.Module):
    def __init__(self, config):
        super(CNNModel, self).__init__()
        in_channels = config['in_channels']
        out_channels = 64
        cnn_structure = []
        for i in range(config['num_layers']):
            if i in config['deformable_cnn']:
                conv = DeformableConv2d(in_channels, out_channels, kernel_size=config['kernel_size'], stride=config['stride'], padding=config['padding'])
            else:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size=config['kernel_size'], stride=config['stride'], padding=config['padding'])
            cnn_structure += [conv, nn.ReLU(), nn.MaxPool2d(2, 2)]
            in_channels = out_channels
            out_channels *= 2
        self.base_model_ = nn.Sequential(*cnn_structure,

            nn.Conv2d(in_channels, in_channels, config['kernel_size'], config['stride'], config['padding']),
            nn.ReLU(),
            nn.MaxPool2d([1, 2], [1, 2]),

            nn.Conv2d(in_channels, config['cnn_features'], config['kernel_size'], config['stride'], config['padding']),
            nn.BatchNorm2d(config['cnn_features']),
            nn.ReLU(),
            nn.MaxPool2d([2, 1], [2, 1]),

            nn.Conv2d(config['cnn_features'], config['cnn_features'], config['kernel_size'], config['stride'], config['padding']),
            nn.BatchNorm2d(config['cnn_features']),
            nn.ReLU(),

        )

    def forward(self, x):
        x = self.base_model_(x)
        x = x.permute(0, 2, 3, 1)
        return x