from functools import partial
import os, sys
sys.path.insert(0, './')
import inversefed
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
import policy
from benchmark.comm import create_model, build_transform, preprocess, create_config, vit_preprocess
from transformers import ViTFeatureExtractor, ViTForImageClassification


parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--aug_list', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--optim', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--mode', default=None, required=True, type=str, help='Mode.')
parser.add_argument('--rlabel', default=False, type=bool, help='rlabel')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
parser.add_argument('--epochs', default=None, required=True, type=int, help='Vision epoch.')
parser.add_argument('--resume', default=0, type=int, help='rlabel')

parser.add_argument('--defense', default=None, type=str, help='Existing Defenses')
parser.add_argument('--tiny_data', default=False, action='store_true', help='Use 0.1 training dataset')
parser.add_argument('--dryrun', default=False, action='store_true', help='Debug mode')
parser.add_argument('--fix_ckpt', default=False, action='store_true', help='Use fix ckpt for attack')

parser.add_argument('--start', default=0, type=int,  help='The start index of attack smaple')
parser.add_argument('--end', default=100, type=int,  help='The end index of attack smaple')
parser.add_argument('--scale_data', default=False, action='store_true', help='Use scale data for single dataset')
parser.add_argument('--input_shape', default=None, type=int,  help='The input shape for init image')

parser.add_argument('--max_iterations', default=None, type=int,  help='Max iteration for attack')

parser.add_argument('--save_verbose', default=False, action='store_true', help='save middle result')
parser.add_argument('--init_sameattr', default=False, action='store_true', help='Initialize data from same attribute image')
parser.add_argument('--init_random', default=False, action='store_true', help='Initialize data from random image')

parser.add_argument('--sr_scale', type=int, required=True, help='The scale of sr')


opt = parser.parse_args()
num_images = 1

SAME_ATTR_IMAGES = {51:[2392], 54:[2906], 73:[1783], 94:[699], 112:[1887], 124:[1479], 129:[250, 1035], 132:[2120, 2236], 134:[529], 140:[2263] }

RANDOM_ATTR_IMAGES = {51: [1595, 2320, 125], 54: [803, 1288, 2082], 73: [320, 2581, 2624], 94: [968, 746, 2468], 112: [2902, 321, 2898], 124: [2556, 2892, 2703], 129: [1507, 1004, 1524], 132: [2379, 1832, 916], 134: [1621, 1495, 2945], 140: [2241, 285, 1908]}
# init env
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative'); defs.epochs = opt.epochs


# init training
arch = opt.arch
trained_model = True
mode = opt.mode
assert mode in ['normal', 'aug', 'crop']

config = create_config(opt)
if opt.max_iterations is not None:
    config["max_iterations"] = opt.max_iterations


def create_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)



def collate_fn(examples, label_key='fine_label'):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example[label_key] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def create_save_dir():
    if opt.fix_ckpt:
        return 'benchmark/images/data_{}_arch_{}_epoch_{}_optim_{}_mode_{}_auglist_{}_rlabel_{}_fix'.format(opt.data, opt.arch, opt.epochs, opt.optim, opt.mode, \
            opt.aug_list, opt.rlabel)
    elif opt.input_shape is not None:
        size = opt.input_shape
        return f'benchmark/images/data_{opt.data}_arch_{opt.arch}_epoch_{opt.epochs}_optim_{opt.optim}_mode_{opt.mode}_auglist_{opt.aug_list}_rlabel_{opt.rlabel}_input_{size}'
    elif opt.scale_data:
        name_and_size = opt.data.split('_')
        name = '_'.join(name_and_size[1:-1])
        size = name_and_size[-1]
        save_dir = f'benchmark/images/scale_{name}/data_{name}_arch_{opt.arch}_epoch_{opt.epochs}_optim_{opt.optim}_mode_{opt.mode}_auglist_{opt.aug_list}_rlabel_{opt.rlabel}_{size}'
        
        if opt.save_verbose:
            save_dir += '_verbose'
            
        if opt.init_sameattr:
            save_dir += "_initSameAttr"
        elif opt.init_random:
            save_dir += "_initRandom"
        
        if opt.sr_scale != 1:
            save_dir += f"_sr_{opt.sr_scale}"
        # print(save_dir)
        # exit(0)
        return save_dir
    elif opt.max_iterations is not None:
        return 'benchmark/images/Iteration/data_{}_arch_{}_epoch_{}_optim_{}_mode_{}_auglist_{}_rlabel_{}_{}'.format(opt.data, opt.arch, opt.epochs, opt.optim, opt.mode, opt.aug_list, opt.rlabel, opt.max_iterations)

    return 'benchmark/images/data_{}_arch_{}_epoch_{}_optim_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.epochs, opt.optim, opt.mode, \
        opt.aug_list, opt.rlabel)


def reconstruct(idx, model, loss_fn, trainloader, validloader, mean_std, shape, label_key, init_index=None):

    dm, ds = mean_std
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
                # print(label, type(label))
                if isinstance(label, torch.Tensor):
                    labels.append(label.to(device=setup['device']).unsqueeze(0))
                else:
                    labels.append(torch.as_tensor((label,), device=setup['device']))
                ground_truth.append(img.to(**setup))

        ground_truth = torch.stack(ground_truth)

    labels = torch.cat(labels)
    model.zero_grad()
    target_loss = loss_fn(model(ground_truth), labels)
    param_list = [param for param in model.parameters() if param.requires_grad]
    input_gradient = torch.autograd.grad(target_loss, param_list)


    # attack
    print('ground truth label is ', labels)
    
    
    save_dir = create_save_dir()
    # print(save_dir)
    # exit(0)
    create_if_not_exist(save_dir)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    save_verbose = None 
    if opt.save_verbose is not None:
        save_verbose = idx 
    
    
    #pass loss_fn that accepts tuple input
    # rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=num_images, loss_fn=loss_fn)
    rec_machine = inversefed.GradientSReconstructor(model, (dm, ds), config, num_images=num_images, loss_fn=loss_fn, sr_scale=opt.sr_scale)

    if opt.rlabel:
        output, stats = rec_machine.reconstruct(input_gradient, None, img_shape=shape) # reconstruction label
    else:
        init_data = None
        if init_index is not None:
            init_data, label = validloader.dataset[init_index]
            # print(init_data.shape) 
            # exit(0)
            _, h, w = init_data.shape 
            init_data = torchvision.transforms.Resize((h//opt.sr_scale, w//opt.sr_scale))(init_data)
            init_data_denormalized = init_data.clone().to(**setup) * ds + dm
            save_dir = os.path.join(save_dir, f"{idx}/{init_index}")
            create_if_not_exist(save_dir)
            torchvision.utils.save_image(init_data_denormalized.cpu().clone(), f"{save_dir}/init.png")

            # exit(0)
        
        
        output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=shape, dryrun=opt.dryrun, init_data=init_data, save_verbose=save_verbose, save_dir=save_dir) # specify label
        # output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=shape, dryrun=True) # specify label

    output_denormalized = output * ds + dm
    input_denormalized = ground_truth * ds + dm


    torchvision.utils.save_image(output_denormalized.cpu().clone(), '{}/rec_{}.png'.format(save_dir, idx))
    torchvision.utils.save_image(input_denormalized.cpu().clone(), '{}/ori_{}.png'.format(save_dir, idx))


    print("optimization end")
    return {"metric":-1}
    # if opt.input_shape is not None:
    #     return 0
    # else:
    #     mean_loss = torch.mean((input_denormalized - output_denormalized) * (input_denormalized - output_denormalized))
    #     print("after optimization, the true mse loss {}".format(mean_loss))

    #     test_mse = (output_denormalized.detach() - input_denormalized).pow(2).mean().cpu().detach().numpy()
    #     if isinstance(model(output.detach()), tuple): 
    #         feat_mse = (model(output.detach())[0]- model(ground_truth)[0]).pow(2).mean()
    #     else:
    #         feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()
            
    #     test_psnr = inversefed.metrics.psnr(output_denormalized, input_denormalized)

    #     return {'test_mse': test_mse,
    #         'feat_mse': feat_mse.detach(), # if not, the computation graph would store in list for each iteration, case OOM error. https://discuss.pytorch.org/t/memory-leak-when-appending-tensors-to-a-list/25937 If you store something from your model (for debugging purpose) and don’t need to calculate gradients with it anymore, I would recommend to call detach on it as it won’t have any effects if the tensor is already detached.
    #         'test_psnr': test_psnr
    #     }




def create_checkpoint_dir():
    if opt.fix_ckpt:
        return 'checkpoints/data_{}_arch_{}_mode_crop_auglist__rlabel_{}'.format(opt.data, opt.arch, opt.rlabel)
    elif opt.scale_data:
        name_and_size = opt.data.split('_')
        name = '_'.join(name_and_size[1:-1])
        size = name_and_size[-1]
        save_dir = f'checkpoints/scale_{name}/data_{name}_arch_{opt.arch}_mode_{opt.mode}_auglist_{opt.aug_list}_rlabel_{opt.rlabel}_{size}'
        return save_dir
    return 'checkpoints/data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.mode, opt.aug_list, opt.rlabel)


def main():
    global trained_model
    print(opt)

    if opt.arch not in ['vit']: 
        loss_fn, trainloader, validloader = preprocess(opt, defs, valid=True)
        model = create_model(opt)
        if opt.data == 'cifar100':
            dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
            ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
            shape = (3, 32, 32)
        elif opt.data == 'FashionMinist':
            dm = torch.Tensor([0.1307]).view(1, 1, 1).cuda()
            ds = torch.Tensor([0.3081]).view(1, 1, 1).cuda()
            shape = (1, 32, 32)
        elif opt.data == 'ImageNet':
            dm = torch.as_tensor(inversefed.consts.imagenet_mean, **setup)[:, None, None]
            ds = torch.as_tensor(inversefed.consts.imagenet_std, **setup)[:, None, None]
            shape = (3, 224, 224)
        
        elif opt.data == 'CelebAHQ_Gender':
            dm = torch.as_tensor(inversefed.consts.celeba_mean, **setup)[:, None, None]
            ds = torch.as_tensor(inversefed.consts.celeba_std, **setup)[:, None, None]
            shape = (3, 256, 256)
            # shape = (3, 112, 112)
        elif opt.data.startswith('Scale_CelebAHQ_Gender'):
            dm = torch.as_tensor(inversefed.consts.celeba_mean, **setup)[:, None, None]
            ds = torch.as_tensor(inversefed.consts.celeba_std, **setup)[:, None, None]

            size = opt.data.split('_')[-1]
            if size.isdigit():
                shape = (3, int(size), int(size))
            else:
                raise AttributeError(f'Error scale size, exptectd a number but got {size} ')
        elif opt.data.startswith('CelebA'):
            dm = torch.as_tensor(inversefed.consts.celeba_mean, **setup)[:, None, None]
            ds = torch.as_tensor(inversefed.consts.celeba_std, **setup)[:, None, None]
            # shape = (3, 128, 128)
            shape = (3, 112, 112)
        
        else:
            raise NotImplementedError
        
        if opt.input_shape is not None:
            shape = (3, opt.input_shape, opt.input_shape)
    else: 
        loss_fn, trainloader, validloader, model, mean_std, scale_size = vit_preprocess(opt, defs, valid=True) # batch size rescale to 16
        dm, ds = mean_std
        if opt.data == 'cifar100':
            dm = torch.as_tensor(dm, **setup)[:, None, None]
            ds = torch.as_tensor(ds, **setup)[:, None, None]
            shape = (3, scale_size, scale_size)
        elif opt.data == 'FashionMinist': 
            dm = torch.Tensor(dm).view(1, 1, 1).cuda()
            ds = torch.Tensor(ds).view(1, 1, 1).cuda()
            shape = (1, scale_size, scale_size)

    label_key = 'fine_label' if opt.data == 'cifar100' else 'label'

    # loss_fn, trainloader, validloader = preprocess(opt, defs, valid=True)
    # model = create_model(opt)
    model.to(**setup)
    if opt.epochs == 0:
        trained_model = False
        
    if trained_model:
        checkpoint_dir = create_checkpoint_dir()
        if 'normal' in checkpoint_dir:
            checkpoint_dir = checkpoint_dir.replace('normal', 'crop')
        filename = os.path.join(checkpoint_dir, f'{opt.arch}_{defs.epochs}.pth')
        # filename = os.path.join(checkpoint_dir, str(defs.epochs) + '.pth')

        if not os.path.exists(filename):
            filename = os.path.join(checkpoint_dir, str(defs.epochs - 1) + '.pth')

        print(filename)
        assert os.path.exists(filename)
        model.load_state_dict(torch.load(filename))

    if opt.rlabel:
        for name, param in model.named_parameters():
            if 'fc' in name:
                param.requires_grad = False

    model.eval()

    save_dir = create_save_dir()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    metric_list = list()
    #resume
    metric_path = save_dir + '/metric.npy'
    if os.path.exists(metric_path):
        metric_list = np.load(metric_path, allow_pickle=True).tolist()

    sample_list = [i for i in range(opt.start, opt.end)]

    if opt.arch ==  'ResNet18_tv' and opt.data == 'ImageNet':
        valid_size = len(validloader.dataset)
        sample_array = np.linspace(0, valid_size, opt.end - opt.start, endpoint=False,dtype=np.int32)
        sample_list = [int(i)+5 for i in sample_array]
        # print(sample_list)
        # print(valid_size)
        # exit(0)

        # sample_list = [25] #debug
        
        
        
    idx_dict = None
    if opt.init_sameattr:
        sample_list = list(SAME_ATTR_IMAGES.keys())
        idx_dict = SAME_ATTR_IMAGES
    elif opt.init_random:
        sample_list = list(RANDOM_ATTR_IMAGES.keys())
        idx_dict = RANDOM_ATTR_IMAGES
    
    mse_loss = 0
    for attack_id, idx in enumerate(sample_list):
        if idx < opt.resume:
            continue
        print('attach {}th in {}'.format(idx, opt.aug_list))
        
        if opt.init_sameattr or opt.init_random:
            for init_index in idx_dict[idx]:
                metric = reconstruct(idx, model, loss_fn, trainloader, validloader, (dm, ds), shape, label_key, init_index=init_index)
                metric_list.append(metric)
        else:
            metric = reconstruct(idx, model, loss_fn, trainloader, validloader, (dm, ds), shape, label_key, init_index=None)
            metric_list.append(metric)
        #save metric after each reconstruction
        np.save('{}/metric.npy'.format(save_dir), metric_list)



if __name__ == '__main__':
    main()
