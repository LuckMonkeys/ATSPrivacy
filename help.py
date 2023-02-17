import torchvision
from torchvision import transforms
import numpy as np
from typing import Any, Callable, List, Optional, Tuple, Union
import torch
import os
from PIL import Image

#find the same attribute image in celeba
"""
class CelebAForGender(torchvision.datasets.CelebA):
    def __init__(self, root: str, split: str = "train", target_type: Union[List[str], str] = "attr", transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, split, target_type, transform, target_transform, download)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X, meta =  super().__getitem__(index) 
        gender_label = meta[20]

        return X, gender_label

data_path = '/home/zx/nfs/server3/data/'

test_dataset = CelebAForGender(data_path, split='valid')
attr = test_dataset.attr

unique_attr, unique_attr_idx, counts = torch.unique(attr, return_inverse=True, dim=0, return_counts=True, sorted=True)

from collections import defaultdict
same_labels =  defaultdict(list)

for i, idx in enumerate(unique_attr_idx):
    same_labels[idx.item()].append(i) 

same_labels_g2 = [] # images have same attributes
for v in same_labels.values():
    if len(v) > 1:
        same_labels_g2.append(v)
print(same_labels_g2[:100])
"""
    
    
#calcuate the gradient distance for each layer on [groud_truth, rand, recoverd]
import inversefed

data = "CelebA_Gender" # FashionMinist, CelebA_Gender
arch = "ResNet18_tv" #Resnet18_tv

model, _ = inversefed.construct_model(arch, num_classes=2, num_channels=3)

"""

ckpt = "/home/zx/nfs/server3/ATSPrivacy/checkpoints/data_CelebA_Gender_arch_ResNet18_tv_mode_crop_auglist__rlabel_False/ResNet18_tv_2.pth"

model.load_state_dict(torch.load(ckpt))

imgs_dir = "/home/zx/nfs/server3/ATSPrivacy/benchmark/images/data_CelebA_Gender_arch_ResNet18_tv_epoch_2_optim_inversed_large_mode_normal_auglist__rlabel_False_verbose"

gt_path = os.path.join(imgs_dir, 'ori_1.png')
rec_path = os.path.join(imgs_dir, '1/4799.png')

gt = Image.open(gt_path)
rec = Image.open(rec_path)

data_mean, data_std = inversefed.consts.celeba_mean, inversefed.consts.celeba_std

transform_list = [
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(data_mean, data_std)
]
transform = transforms.Compose(transform_list)

gt_t, rec_t  = transform(gt), transform(rec)
gt_t, rec_t = gt_t.unsqueeze(0), rec_t.unsqueeze(0)
rand = torch.randn_like(gt_t)

from inversefed.data.data_processing import _build_celeba_gender

trainset, validset = _build_celeba_gender('~/data/')
validloader = torch.utils.data.DataLoader(validset, batch_size=128,
        shuffle=False, drop_last=True, num_workers=16, pin_memory=True)


_, label = validloader.dataset[0]

loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                         reduce=None, reduction='mean')

parmameters = [p for p in model.parameters() if p.requires_grad]

model.eval()

grad_list = []
# for input in [gt_t, rec_t, rand]:
for input in [rec_t]:

    model.zero_grad()
    loss = loss_fn(model(input), torch.tensor([label]))
    grad_list.append(torch.autograd.grad(loss, parmameters)) 
    print(grad_list[-1])
    exit(0)


loss_list = []
grad_loss = torch.nn.MSELoss(reduction="mean")
for g_gt, g_rec in zip(grad_list[0], grad_list[1]):
    loss_list.append(grad_loss(g_gt, g_rec))

# for name, loss in zip(names, loss_list):
#     print(name, loss)
print(loss_list)

"""


# names = [p[0] for p in model.named_parameters()]

# total_loss1 = torch.load("/home/zx/nfs/server3/ATSPrivacy/layer_grad_compare/ResNet18_tv_CelebA_Gender_MSE.pt")
# total_loss2 = torch.load("/home/zx/nfs/server3/ATSPrivacy/layer_grad_compare/ResNet18_tv_CelebA_Gender_sim.pt")
# total_loss3 = torch.load("/home/zx/nfs/server3/ATSPrivacy/layer_grad_compare/ResNet18_tv_CelebA_Gender_sim_local.pt")

# total_loss_mean1, total_loss_mean2, total_loss_mean3  = torch.mean(total_loss1, dim=0), torch.mean(total_loss2, dim=0), torch.mean(total_loss3, dim=0)


# for name, loss1, loss2, loss3 in zip(names, total_loss_mean1, total_loss_mean2, total_loss_mean3):
#     print(f"{name} & {loss1.item():.3e} & {loss2.item():.3e} & {loss3.item():.3e}")



import torch
g_112 = torch.load("/home/zx/nfs/server3/ATSPrivacy/gradients_in_diff_imgsize/112", map_location=torch.device("cpu"))
# g_224 = torch.load("/home/zx/nfs/server3/ATSPrivacy/gradients_in_diff_imgsize/224")
print(g_112)