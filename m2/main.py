from matplotlib import pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

from m2.dataset import CifarDS

from sklearn.mixture import GaussianMixture
import numpy as np

# Sample data (e.g., 300 points, 2 features)

