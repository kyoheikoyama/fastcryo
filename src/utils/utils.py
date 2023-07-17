import math
import torch
import numpy as np
from PIL import Image
from torch import nn


# def split_image(image, n=10):
#     height, width = image.shape
#     tile_height, tile_width = height // n, width // n

#     tiles = []
#     for i in range(n):
#         for j in range(n):
#             tile = image[
#                 i * tile_height : (i + 1) * tile_height,
#                 j * tile_width : (j + 1) * tile_width,
#             ]
#             tiles.append(tile)

#     return tiles


def split_image(image, n=10, margin=5):
    height, width = image.shape
    tile_height, tile_width = height // n, width // n

    tiles = []
    for i in range(n):
        temp = []
        for j in range(n):
            tile = image[
                i * tile_height - margin : (i + 1) * tile_height + margin,
                j * tile_width - margin : (j + 1) * tile_width + margin,
            ]
            temp.append(tile)

        tiles.append(temp)

    return tiles


def equal_var_init(model):
    for name, param in model.named_parameters():
        # print(name, param.shape)
        if len(param.shape) == 1:
            continue
        if name.endswith(".bias"):
            param.data.fill_(0)
        else:
            param.data.normal_(std=1.0 / math.sqrt(param.shape[1]))


def weights_init(m):
    """
    # Apply the weight initialization
    model.apply(weights_init)
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)


def clip_image(image, max_percentile=99, min_percentile=0):
    upper_limit = np.percentile(image, max_percentile)
    lower_limit = np.percentile(image, min_percentile)
    return np.clip(image, lower_limit, upper_limit)
