from ast import AugLoad
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
import torch.nn.functional as F
from benchmark.comm import create_model, build_transform, preprocess, create_config, vit_preprocess, build_vit_transform, split
import policy
import copy
from transformers import ViTFeatureExtractor, ViTForImageClassification
from functools import partial
from tqdm import tqdm


policies = policy.policies

parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--mode', default=None, required=True, type=str, help='Mode.')
parser.add_argument('--aug_list', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--rlabel', default=False, type=bool, help='rlabel')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
parser.add_argument('--epochs', default=None, required=True, type=int, help='Vision epoch.')
parser.add_argument('--num_samples', default=5, type=int, help='Images per class')

parser.add_argument('--tiny_data', default=False, action='store_true', help='Use 0.1 training dataset')
parser.add_argument('--aug_file', default=None, type=str, help='Candidate policy file') # e.g. line1 3-1-7, line2 15-4-27
parser.add_argument('--file_exist_ok', default=False, action='store_true', help='if file alread exist and file_exist_ok equal false, skip current evaluation')
opt = parser.parse_args()


# init env
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative'); defs.epochs = opt.epochs

# init training
arch = opt.arch
trained_model = True
mode = opt.mode
assert mode in ['normal', 'aug', 'crop']
num_images = 1

def similarity_measures(img_batch, ref_batch, batched=True, method='fsim'):
        
    from image_similarity_measures.quality_metrics import fsim, issm, rmse, sam, sre, ssim, uiq
    methods = {'fsim':fsim, 'issm':issm, 'rmse':rmse, 'sam':sam, 'sre':sre, 'ssim':ssim, 'uiq':uiq }

    def get_similarity(img_in, img_ref):
        return methods[method](img_in.permute(1,2,0).numpy(), img_ref.permute(1,2,0).numpy())
        
    if not batched:
        sim = get_similarity(img_batch.detach(), ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        sim_list = []
        for sample in range(B):
            sim_list.append(get_similarity(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))

        sim_list = np.array(sim_list)
        sim_list = sim_list[~np.isnan(sim_list)]
        sim = np.mean(sim_list)
    return sim

def collate_fn(examples, label_key='fine_label'):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example[label_key] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def eval_score(jacob, labels=None):
    corrs = np.corrcoef(jacob)
    v, _  = np.linalg.eig(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1./(v + k))


def get_batch_jacobian(net, x, target):
    net.eval()
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x)
    if isinstance(y, tuple):
        y = y[0] # vit model return  (logit)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach()

def calculate_dw(model, inputs, labels, loss_fn):
    model.zero_grad()
    target_loss = loss_fn(model(inputs), labels)
    dw = torch.autograd.grad(target_loss, model.parameters())
    return dw


def cal_dis(a, b, metric='L2'):
    a, b = a.flatten(), b.flatten()
    if metric == 'L2':
        return torch.mean((a - b) * (a - b)).item()
    elif metric == 'L1':
        return torch.mean(torch.abs(a-b)).item()
    elif metric == 'cos':
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    else:
        raise NotImplementedError



def accuracy_metric(idx_list, model, loss_fn, trainloader, validloader, label_key='fine_label'):
    if opt.data == 'cifar100':
        dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    elif opt.data == 'FashionMinist':
        dm = torch.Tensor([0.1307]).view(1, 1, 1).cuda()
        ds = torch.Tensor([0.3081]).view(1, 1, 1).cuda()
    elif opt.data == 'ImageNet':
        dm = torch.as_tensor(inversefed.consts.imagenet_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.imagenet_std, **setup)[:, None, None]
    elif opt.data.startswith('CelebA'):
        dm = torch.as_tensor(inversefed.consts.celeba_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.celeba_std, **setup)[:, None, None]
        
    else:
        raise NotImplementedError

    # prepare data
    ground_truth, labels = [], []

    if isinstance(model, ViTForImageClassification):
        #return tuple(logits,) instead of ModelOutput object
        model.forward = partial(model.forward, return_dict=False)
        for idx in idx_list:
            example = validloader.dataset[idx]
            label = example[label_key]

            idx += 1
            if label not in labels:
                labels.append(torch.as_tensor((label,), device=setup['device']))
                ground_truth.append(example)
        
        ground_truth = collate_fn(ground_truth, label_key=label_key)['pixel_values'].to(**setup)

    else: 
        for idx in idx_list:
            img, label = validloader.dataset[idx]
            idx += 1
            if label not in labels:
                labels.append(torch.as_tensor((label,), device=setup['device']))
                ground_truth.append(img.to(**setup))

        ground_truth = torch.stack(ground_truth)
        
    labels = torch.cat(labels)
    model.zero_grad()
    jacobs, labels= get_batch_jacobian(model, ground_truth, labels)
    jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
    return eval_score(jacobs, labels)



def reconstruct(idx, model, loss_fn, trainloader, validloader, label_key='fine_label'):
    if opt.data == 'cifar100':
        dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    elif opt.data == 'FashionMinist':
        dm = torch.Tensor([0.1307]).view(1, 1, 1).cuda()
        ds = torch.Tensor([0.3081]).view(1, 1, 1).cuda()
    elif opt.data == 'ImageNet':
        dm = torch.as_tensor(inversefed.consts.imagenet_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.imagenet_std, **setup)[:, None, None]
    elif opt.data.startswith('CelebA'):
        dm = torch.as_tensor(inversefed.consts.celeba_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.celeba_std, **setup)[:, None, None]
    else:
        raise NotImplementedError
    
    # prepare data
    ground_truth, labels = [], []
    if isinstance(model, ViTForImageClassification):
        #return tuple(logits,) instead of ModelOutput object
        model.forward = partial(model.forward, return_dict=False)
        while len(labels) < num_images:
            example = validloader.dataset[idx]
            label = example[label_key]

            idx += 1
            if label not in labels:
                labels.append(torch.as_tensor((label,), device=setup['device']))
                ground_truth.append(example)
        
        ground_truth = collate_fn(ground_truth, label_key=label_key)['pixel_values'].to(**setup)

    else: 
        while len(labels) < num_images:
            img, label = validloader.dataset[idx]
            idx += 1
            if label not in labels:
                if isinstance(label, torch.Tensor):
                    labels.append(label.to(device=setup['device']).unsqueeze(0))
                else:
                    labels.append(torch.as_tensor((label,), device=setup['device']))

                ground_truth.append(img.to(**setup))

        ground_truth = torch.stack(ground_truth)

    # while len(labels) < num_images:
    #     img, label = validloader.dataset[idx]
    #     idx += 1
    #     if label not in labels:
    #         labels.append(torch.as_tensor((label,), device=setup['device']))
    #         ground_truth.append(img.to(**setup))

    labels = torch.cat(labels)
    model.zero_grad()
    # calcuate ori dW
    target_loss = loss_fn(model(ground_truth), labels)
    input_gradient = torch.autograd.grad(target_loss, model.parameters())

    metric = 'cos'

    # attack model
    model.eval()
    dw_list = list()
    dx_list = list()
    bin_num = 20
    noise_input = (torch.rand((ground_truth.shape)).cuda() - dm) / ds
    for dis_iter in range(bin_num+1):
        model.zero_grad()
        fake_ground_truth = (1.0 / bin_num * dis_iter * ground_truth + 1. / bin_num * (bin_num - dis_iter) * noise_input).detach()
        fake_dw = calculate_dw(model, fake_ground_truth, labels, loss_fn)
        dw_loss = sum([cal_dis(dw_a, dw_b, metric=metric) for dw_a, dw_b in zip(fake_dw, input_gradient)]) / len(input_gradient)

        dw_list.append(dw_loss)

    interval_distance = cal_dis(noise_input, ground_truth, metric='L1') / bin_num


    def area_ratio(y_list, inter):
        area = 0
        max_area = inter * bin_num
        for idx in range(1, len(y_list)):
            prev = y_list[idx-1]
            cur = y_list[idx]
            area += (prev + cur) * inter / 2
        return area / max_area

    return area_ratio(dw_list, interval_distance)

def get_num_classes(dataset: str):
    if dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'FashionMinist':
        num_classes = 10
    elif dataset == 'ImageNet':
        num_classes = 25
    elif dataset in ['CelebA_Gender',  'CelebA_Smile', 'CelebAHQ_Gender']:
        num_classes = 2
    elif dataset == 'CelebA_Identity':
        num_classes = 500
    elif dataset == 'CelebAFaceAlign_MLabel':
        num_classes = -1
    return num_classes
    

def main():
    #set meta data
    num_classes = get_num_classes(opt.data)
    label_key = 'fine_label' if opt.data == 'cifar100' else 'label'

    compute_privacy_score = True
    compute_acc_score = False

    # import time
    # start = time.time()

    #load policies list
    if opt.aug_file is not None:
        if not os.path.exists(opt.aug_file):
            raise AttributeError('Aug list file not exits.')
        else:
            with open(opt.aug_file, 'r') as f:
                aug_lists = list(filter(lambda x: x not in ['', '\n'], f.readlines()))
                aug_lists = [aug.strip() for aug in aug_lists]
        # print(len(aug_lists), aug_lists)
        # exit(0)
    else:
        aug_lists = [opt.aug_list]
    
    #load object for model, train_dataloader, test_dataloader 
    if opt.arch not in ['vit']:
        loss_fn, trainloader, validloader = preprocess(opt, defs, valid=True)
        model = create_model(opt)
    else:
        loss_fn, trainloader, validloader, model, mean_std, scale_size = vit_preprocess(opt, defs, valid=True) # batch size rescale to 16
        mean, std = mean_std
    model.to(**setup)

    #save random init model for acc evalution
    old_state_dict = copy.deepcopy(model.state_dict())
 
    #privacy socre
    if compute_privacy_score:
        #load tiny_data_for_privay
        tiny_state_dict = torch.load('checkpoints/tiny_data_{}_arch_{}/{}.pth'.format(opt.data, opt.arch, opt.epochs))
        model.load_state_dict(tiny_state_dict)
        model.eval()

        #create save dir        
        root_dir = f'search/data_{opt.data}_arch_{opt.arch}'
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        
        #select smaples for privay evalution
        if num_classes == -1:
            sample_list = [i for i in range(100)]
        else:
            sample_list = defaultdict(list)
            if opt.arch not in ['vit']:
                for idx, (_, label) in enumerate(validloader.dataset):   
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    sample_list[label].append(idx)
            else:
                for idx, sample in enumerate(validloader.dataset):   
                    sample_list[sample[label_key]].append(idx)

        for p in tqdm(aug_lists):

            opt.aug_list = p

            print(f'Evaluate privay socre of policy {p}')
            pathname = 'search/data_{}_arch_{}/{}.npy'.format(opt.data, opt.arch, opt.aug_list)
            # print(pathname)

            if not opt.file_exist_ok:
                if os.path.exists(pathname):
                    print(f'Find evalution file, continue...')
                    continue
            
            # exit(0)

            policy_list = split(p)

            #update transform
            if opt.arch not in ['vit']:
                train_transforms = build_transform(True, policy_list, opt, defs)
                validloader.dataset.transform =  train_transforms
            else:
                train_transforms, val_transforms = build_vit_transform(True, policy_list, opt, defs, (mean,std), scale_size)
                validloader.dataset.set_transform(train_transforms)
            
            metric_list = list()

            if isinstance(sample_list, list):
                for idx in sample_list:
                    metric_list.append(reconstruct(sample_list[idx], model, loss_fn, trainloader, validloader, label_key))
            
            elif isinstance(sample_list, dict):
                 
                num_samples = opt.num_samples
                for label in range(num_classes):
                    metric = []
                    for idx in range(num_samples):
                        metric.append(reconstruct(sample_list[label][idx], model, loss_fn, trainloader, validloader, label_key))
                        # print('attach {}th in class {}, auglist:{} metric {}'.format(idx, label, opt.aug_list, metric))
                    metric_list.append(np.mean(metric,axis=0))

            if len(metric_list) > 0:
                print(np.mean(metric_list))
                np.save(pathname, metric_list)
   
    #acc score
    if compute_acc_score:
        # maybe need old_state_dict
        model.load_state_dict(old_state_dict)

        for p in tqdm(aug_lists):
            print(f'Evaluate acc socre of policy {p}')
            opt.aug_list = p
            policy_list = split(p)

            #update transform
            train_transforms, val_transforms = build_vit_transform(True, policy_list, opt, defs, (mean,std), scale_size)
            validloader.dataset.set_transform(train_transforms)

            score_list = list()
            for run in range(10):
                large_samle_list = [200 + run  * 100 + i for i in range(100)]
                score = accuracy_metric(large_samle_list, model, loss_fn, trainloader, validloader, label_key)
                score_list.append(score)
        
            pathname = 'accuracy/data_{}_arch_{}/{}'.format(opt.data, opt.arch, p)
            root_dir = os.path.dirname(pathname)
            if not os.path.exists(root_dir):
                os.makedirs(root_dir)
            np.save(pathname, score_list)
            print(score_list)

    # print('time cost ', time.time() - start)


if __name__ == '__main__':
    main()
