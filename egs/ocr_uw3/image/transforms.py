import math
import numpy as np
import random
import torch
import torchvision

from PIL import Image

def get_transforms(config, preprocess_only=False):
    transform_list = []

    if hasattr(config, 'scale_height'):
        transform_list.append(torchvision.transforms.ToPILImage())
        transform_list.append(torchvision.transforms.Lambda(lambda img: _scale_height(img, config.scale_height)))
        transform_list.append(torchvision.transforms.ToTensor())
    if hasattr(config, 'augment') and preprocess_only == False:
        if 'rotation' in config.augment:
            transform_list.append(
                torchvision.transforms.RandomAffine(
                    degrees=(-1*config.augment['rotation'], config.augment['rotation']),
                    shear=(0,0),
                    fill=1,
                )
            )
        if 'shear' in config.augment:
            transform_list.append(
                torchvision.transforms.RandomAffine(
                    degrees=(0,0),
                    shear=(-1*config.augment['shear'], config.augment['shear']),
                    fill=1,
                )
            )
        if 'resample' in config.augment:
            transform_list.append(torchvision.transforms.ToPILImage())
            transform_list.append(torchvision.transforms.Lambda(lambda img: _resample(img)))
            transform_list.append(torchvision.transforms.ToTensor())
        if 'resized_crop' in config.augment:
            transform_list.append(torchvision.transforms.Lambda(lambda img: _resized_crop(img)))
    if hasattr(config, 'padding'):
        transform_list.append(
            torchvision.transforms.Pad(
                padding=(config.padding, 0, config.padding, 0),
                fill=1,
            )
        )
    if hasattr(config, 'gaussian_noise'):
        transform_list.append(
            torchvision.transforms.Lambda(lambda x: x + (config.gaussian_noise * torch.randn_like(x)))
        )
    if hasattr(config, 'grayscale'):
        transform_list.append(torchvision.transforms.Grayscale(config.grayscale))

    if config.grayscale == 1:
        transform_list.append(torchvision.transforms.Normalize((0.5,), (0.5,)))
    else:
        transform_list.append(torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return torchvision.transforms.Compose(transform_list)

def _resample(img):
    ow, oh = torchvision.transforms.functional.get_image_size(img)
    target_height = random.randint(int(oh*0.25), int(oh*0.75))
    img = _scale_height(img, target_height)
    img = _scale_height(img, oh)
    return img

def _resized_crop(img):
    ow, oh = torchvision.transforms.functional.get_image_size(img)
    resized_crop = torchvision.transforms.RandomResizedCrop((oh, ow), scale=(0.95,1.0), ratio=(ow/oh,ow/oh))
    return resized_crop(img)

def _scale_height(img, target_height):
    ow, oh = torchvision.transforms.functional.get_image_size(img)
    if (oh == target_height):
        return img
    h = target_height
    w = int(target_height * ow / oh)
    return img.resize((w, h))
