import os, sys
sys.path.insert(0, './')
import torch
import torchvision
seed=23333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import random
random.seed(seed)

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import inversefed
import torchvision.transforms as transforms
import argparse
from autoaugment import SubPolicy
from inversefed.data.data_processing import _build_cifar100, _get_meanstd
from inversefed.data.loss import LabelSmoothing
from inversefed.utils import Cutout
import torch.nn.functional as F
import torch.nn as nn
import policy

from benchmark.comm import create_model, build_transform, preprocess, create_config, vit_preprocess



replace = False
if replace:
    policies = policy.policies_replace
    print('Warning: using replace policies, make sure use it correctly')
    exit(0)
else:
    policies = policy.policies

parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
parser.add_argument('--epochs', default=None, required=True, type=int, help='Vision epoch.')
parser.add_argument('--aug_list', default=None, required=True, type=str, help='Augmentation method.')
parser.add_argument('--mode', default=None, required=True, type=str, help='Mode.')
parser.add_argument('--rlabel', default=False, type=bool, help='remove label.')
parser.add_argument('--evaluate', default=False, type=bool, help='Evaluate')

parser.add_argument('--defense', default=None, type=str, help='Existing Defenses')
parser.add_argument('--tiny_data', default=False, action='store_true', help='Use 0.1 training dataset')

opt = parser.parse_args()

# init env
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative'); defs.epochs = opt.epochs

# init training
arch = opt.arch
trained_model = True
mode = opt.mode
assert mode in ['normal', 'aug', 'crop', 'GAN']


def create_save_dir():
    if opt.tiny_data:
        return 'checkpoints/tiny_data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.mode, opt.aug_list, opt.rlabel)
    return 'checkpoints/data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.mode, opt.aug_list, opt.rlabel)


def main():
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative'); defs.epochs = opt.epochs
    if opt.arch not in ['vit']: 
        loss_fn, trainloader, validloader = preprocess(opt, defs, valid=False)
        model = create_model(opt)
    else: 
        # defs = inversefed.training_strategy('vitadam'); defs.epochs = opt.epochs # for tiny data
        loss_fn, trainloader, validloader, model, _, _ = vit_preprocess(opt, defs, valid=False) # batch size rescale to 16

    # init model
    model.to(**setup)
    save_dir = create_save_dir()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file = f'{save_dir}/{arch}_{defs.epochs}.pth'
    # inversefed.train(model, loss_fn, trainloader, validloader, defs, setup=setup, save_dir=save_dir)
    inversefed.train_pl(model, loss_fn, trainloader, validloader, defs, setup=setup, save_dir=save_dir, opt=opt)
    torch.save(model.state_dict(), f'{file}')
    model.eval()


def evaluate():
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative'); defs.epochs=opt.epochs
    if opt.arch not in ['vit']: 
        loss_fn, trainloader, validloader = preprocess(opt, defs, valid=False)
        model = create_model(opt)
    else: 
        loss_fn, trainloader, validloader, model = vit_preprocess(opt, defs, valid=False)
    model.to(**setup)
    root = create_save_dir()

    filename = os.path.join(root, '{}_{}.pth'.format(opt.arch, opt.epochs))
    print(filename)
    if not os.path.exists(filename):
        assert False

    print(filename)
    model.load_state_dict(torch.load(filename))
    model.eval()
    stats = {'valid_losses':list(), 'valid_Accuracy':list()}
    inversefed.training.training_routine.validate(model, loss_fn, validloader, defs, setup=setup, stats=stats)
    print(stats)

if __name__ == '__main__':
    if opt.evaluate:
        evaluate()
        exit(0)
    main()
