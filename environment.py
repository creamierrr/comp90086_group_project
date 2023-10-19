import os
import sys
import math
import inspect
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from scipy.stats import pearsonr
from einops import rearrange, repeat
from sklearn.model_selection import train_test_split

import json
from datetime import datetime
import pickle
import copy
import gc
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.hub
from torchvision import transforms
import torchvision
from torchvision.io import read_image
import torchvision.transforms.functional as transF

import sklearn
from sklearn.metrics import accuracy_score

from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian

import tensorflow as tf

import torch.nn.functional as F

def cosine_similarity(a, b):
    return 1 - F.cosine_similarity(a, b)



