import argparse

from matplotlib import pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

from m2.dataset import CifarDS

from sklearn.mixture import GaussianMixture
import numpy as np

from m2.flow_matching import train_fm
from m2.flow_matching_eval import eval_fm


def model_type_to_runnable(model_type, train=True):
    run_fm = train_fm if train else eval_fm
    match model_type:
        case "base": run_fm('./m2/configs/base_fm.yaml')
        case "ot": run_fm('./m2/configs/base_fm.yaml', "./m2/configs/ot.yaml")
        case "tpc": run_fm('./m2/configs/base_fm.yaml', "./m2/configs/tpc.yaml")
        case "gmm": run_fm('./m2/configs/base_fm.yaml', "./m2/configs/gmm.yaml")
        case "rectified": run_fm('./m2/configs/base_fm.yaml', "./m2/configs/rectified.yaml")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train and evaluate flow matching for differing sample and loss structures"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=['base', 'ot', 'tpc', 'gmm', 'rectified'],
        default='base',
        help="Choose a model to run.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="True to train, false to evaluate",
    )

    args = parser.parse_args()
    model_type_to_runnable(args.model_type, args.train)
    

