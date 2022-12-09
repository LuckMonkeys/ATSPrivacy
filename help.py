

# import torch
# import torch.utils.data as data

# from os import listdir
# from os.path import join
# from PIL import Image
# from torchvision.datasets import CelebA
# import os
# import PIL
# from torchvision.datasets.utils import download_and_extract_archive, check_integrity, download_file_from_google_drive, extract_archive, verify_str_arg
# from torchvision.datasets.vision import VisionDataset
# import csv
# from collections import namedtuple
# CSV = namedtuple("CSV", ["header", "index", "data"])

# from typing import Any, Callable, List, Optional, Tuple, Union
# class CelebaHQ_Gender(VisionDataset):
#     """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

#     Args:
#         root (string): Root directory where images are downloaded to.
#         split (string): One of {'train', 'valid', 'test', 'all'}.
#             Accordingly dataset is selected.
#         target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
#             or ``landmarks``. Can also be a list to output a tuple with all specified target types.
#             The targets represent:

#                 - ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
#                 - ``identity`` (int): label for each person (data points with the same identity are the same person)
#                 - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
#                 - ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
#                   righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

#             Defaults to ``attr``. If empty, ``None`` will be returned as target.

#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.PILToTensor``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         download (bool, optional): If true, downloads the dataset from the internet and
#             puts it in root directory. If dataset is already downloaded, it is not
#             downloaded again.
#     """

#     base_folder = "celeba_hq"
#     # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
#     # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
#     # right now.
#     file_list = [
#         # File ID                                      MD5 Hash                            Filename
#         # ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
#         # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
#         # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
#         ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
#         # ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
#         # ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
#         # ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
#         # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
#         ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
#     ]

#     def __init__(
#         self,
#         root: str,
#         split: str = "train",
#         target_type: Union[List[str], str] = "attr",
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         download: bool = False,
#     ) -> None:
#         super().__init__(root, transform=transform, target_transform=target_transform)
#         self.split = split
#         if isinstance(target_type, list):
#             self.target_type = target_type
#         else:
#             self.target_type = [target_type]

#         if not self.target_type and self.target_transform is not None:
#             raise RuntimeError("target_transform is specified but target_type is empty")

#         # if download:
#         #     self.download()

#         if not self._check_integrity():
#             raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

#         split_map = {
#             "train": 0,
#             "valid": 1,
#             "test": 2,
#             "all": None,
#         }
#         split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]
#         splits = self._load_csv("list_eval_partition.txt")
#         # identity = self._load_csv("identity_CelebA.txt")
#         # bbox = self._load_csv("list_bbox_celeba.txt", header=1)
#         # landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
#         attr = self._load_csv("list_attr_celeba.txt", header=1)

#         mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

#         if mask == slice(None):  # if split == "all"
#             self.filename = splits.index
#         else:
#             self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
#         # self.identity = identity.data[mask]
#         # self.bbox = bbox.data[mask]
#         # self.landmarks_align = landmarks_align.data[mask]
#         self.attr = attr.data[mask]
#         # map from {-1, 1} to {0, 1}
#         self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
#         self.attr_names = attr.header

#     def _load_csv(
#         self,
#         filename: str,
#         header: Optional[int] = None,
#     ) -> CSV:
#         with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
#             data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

#         if header is not None:
#             headers = data[header]
#             data = data[header + 1 :]
#         else:
#             headers = []

#         indices = [row[0] for row in data]
#         data = [row[1:] for row in data]
#         data_int = [list(map(int, i)) for i in data]

#         return CSV(headers, indices, torch.tensor(data_int))

#     def _check_integrity(self) -> bool:
#         return os.path.isdir(os.path.join(self.root, self.base_folder, "data256"))


#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         X = PIL.Image.open(os.path.join(self.root, self.base_folder, "data256", self.filename[index]))

#         target: Any = []
#         for t in self.target_type:
#             if t == "attr":
#                 target.append(self.attr[index, :])
#             elif t == "identity":
#                 target.append(self.identity[index, 0])
#             elif t == "bbox":
#                 target.append(self.bbox[index, :])
#             elif t == "landmarks":
#                 target.append(self.landmarks_align[index, :])
#             else:
#                 # TODO: refactor with utils.verify_str_arg
#                 raise ValueError(f'Target type "{t}" is not recognized.')

#         if self.transform is not None:
#             X = self.transform(X)

#         if target:
#             target = tuple(target) if len(target) > 1 else target[0]

#             if self.target_transform is not None:
#                 target = self.target_transform(target)
#         else:
#             target = None

#         return X, target[20].item()


#     def __len__(self) -> int:
#         return len(self.attr)

#     def extra_repr(self) -> str:
#         lines = ["Target type: {target_type}", "Split: {split}"]
#         return "\n".join(lines).format(**self.__dict__)

# data_path = "~/data"



# test_dataset = CelebaHQ_Gender(data_path, split='valid')
# attr = test_dataset.attr
# # print(test_dataset.attr.shape)
# # print(torch.unique(attr, return_inverse=True, dim=1, return_counts=True))

# unique_attr, unique_attr_idx, counts = torch.unique(attr, return_inverse=True, dim=0, return_counts=True, sorted=True)

# # print(attr.shape, counts.shape)
# from collections import defaultdict
# same_labels =  defaultdict(list)

# for i, idx in enumerate(unique_attr_idx):
#     same_labels[idx.item()].append(i) 
# #    print(type(idx.item()), idx.item()) 
# #    break

# # values = list(same_labels.values())

# same_labels_g2 = []
# for v in same_labels.values():
#     if len(v) > 1:
#         same_labels_g2.append(v)
# print(same_labels_g2[:100])


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


names = [p[0] for p in model.named_parameters()]
# /home/zx/data/GitRepo/ATSPrivacy/layer_grad_compare
total_loss1 = torch.load("/home/zx/data/GitRepo/ATSPrivacy/layer_grad_compare/ResNet18_tv_ImageNet_MSE.pt")
total_loss2 = torch.load("/home/zx/data/GitRepo/ATSPrivacy/layer_grad_compare/ResNet18_tv_ImageNet_sim.pt")
total_loss3 = torch.load("/home/zx/data/GitRepo/ATSPrivacy/layer_grad_compare/ResNet18_tv_ImageNet_sim_local.pt")

total_loss_mean1, total_loss_mean2, total_loss_mean3  = torch.mean(total_loss1, dim=0), torch.mean(total_loss2, dim=0), torch.mean(total_loss3, dim=0)


for name, loss1, loss2, loss3 in zip(names, total_loss_mean1, total_loss_mean2, total_loss_mean3):
    print(f"{name} & {loss1.item():.3e} & {loss2.item():.3e} & {loss3.item():.3e}")