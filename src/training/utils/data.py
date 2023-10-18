import torch
import torch.nn.functional as F

import numpy as np

import torchvision
import torchvision.transforms as transforms

from src.constants import INPUT_DIR

def dataset_factory(params):
    if params["dataset_name"] == "mnist":
        return torchvision.datasets.MNIST(
            root=INPUT_DIR,
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        ), torchvision.datasets.MNIST(
            root=INPUT_DIR,
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )

    else:
        raise ValueError("Invalid dataset name")