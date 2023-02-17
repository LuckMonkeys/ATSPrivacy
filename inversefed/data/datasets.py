"""This is dataset.py from pytorch-examples.

Refer to

https://github.com/pytorch/examples/blob/master/super_resolution/dataset.py.
"""
import torch
import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image
from torchvision.datasets import CelebA
import os
import PIL
from glob import glob

def _is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def _load_img(filepath, RGB=True):
    img = Image.open(filepath)
    if RGB:
        pass
    else:
        img = img.convert('YCbCr')
        img, _, _ = img.split()
    return img


class DatasetFromFolder(data.Dataset):
    """Generate an image-to-image dataset from images from the given folder."""

    def __init__(self, image_dir, replicate=1, input_transform=None, target_transform=None, RGB=True, noise_level=0.0):
        """Init with directory, transforms and RGB switch."""
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if _is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

        self.replicate = replicate
        self.classes = [None]
        self.RGB = RGB
        self.noise_level = noise_level

    def __getitem__(self, index):
        """Index into dataset."""
        input = _load_img(self.image_filenames[index % len(self.image_filenames)], RGB=self.RGB)
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        if self.noise_level > 0:
            # Add noise
            input += self.noise_level * torch.randn_like(input)

        return input, target

    def __len__(self):
        """Length is amount of files found."""
        return len(self.image_filenames) * self.replicate

        

from typing import Any, Callable, List, Optional, Tuple, Union

class CelebAForGender(CelebA):
    def __init__(self, root: str, split: str = "train", target_type: Union[List[str], str] = "attr", transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, split, target_type, transform, target_transform, download)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X, meta =  super().__getitem__(index) 
        gender_label = meta[20]

        return X, gender_label.item()

class CelebAForMLabel(CelebA):
    def __init__(self, root: str, split: str = "train", target_type: Union[List[str], str] = "attr", transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, split, target_type, transform, target_transform, download)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X, meta =  super().__getitem__(index) 

        return X, meta.to(torch.float32)


class CelebAForSmile(CelebA):
    def __init__(self, root: str, split: str = "train", target_type: Union[List[str], str] = "attr", transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, split, target_type, transform, target_transform, download)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X, meta =  super().__getitem__(index) 
        smile_label = meta[31]

        return X, smile_label.item()

        

class CelebAFaceAlignForMLabel(CelebA):
    def __init__(self, root: str, split: str = "train", target_type: Union[List[str], str] = "attr", transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root, split, target_type, transform, target_transform, download)

    def _check_integrity(self) -> bool:
        # for (_, md5, filename) in self.file_list:
        #     fpath = os.path.join(self.root, self.base_folder, filename)
        #     _, ext = os.path.splitext(filename)
        #     # Allow original archive to be deleted (zip and 7z)
        #     # Only need the extracted images
        #     if ext not in [".zip", ".7z"] and not check_integrity(fpath, md5):
        #         return False

    # Should check a hash of the images
        return os.path.isdir(os.path.join(self.root, self.base_folder, "celeba_face_align_landmarks"))
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "celeba_face_align_landmarks", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target.to(torch.float32)


class TensorDatasetWithTransform(data.Dataset):
    
    def __init__(self, *tensors) -> None:
            assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
            # self.tensors = tensors
            self.inputs = tensors[0]
            self.targets = tensors[1]
            # print(self.inputs[0])
            # exit(0)

    def __getitem__(self, index):
        X = self.inputs[index] 
        target = self.targets[index]
        
        if self.transform is not None:
            X = self.transform(X)
        
        return X, target
        # return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.inputs.size(0)




def get_condensation_Dataset(data_path, exp_idx=0):
    '''
    args:
        data_path: the path of condensation dataset tensor .pt file
        {data:[[[data_tensor], [lables]],[]], accs_all_exps:{}}
    '''
    
    data_save = torch.load(data_path)
    max_exp = len(data_save['data']) - 1
    exp_idx = max(min(exp_idx, max_exp), 0)
    
    inputs, labels  = data_save['data'][exp_idx] #inputs: [num_samples, channel, H, W], labels: [num_samples]
    # if inputs.shape[1] == 1:
    #     inputs = inputs.squeeze(1)
    
    return TensorDatasetWithTransform(inputs, labels)



from torchvision.datasets.utils import download_and_extract_archive, check_integrity, download_file_from_google_drive, extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset
from collections import namedtuple
import csv
CSV = namedtuple("CSV", ["header", "index", "data"])
class CelebAHQForGender(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string): One of {'train', 'valid', 'test', 'all'}.
            Accordingly dataset is selected.
        target_type (string or list, optional): Type of target to use, ``attr``, ``identity``, ``bbox``,
            or ``landmarks``. Can also be a list to output a tuple with all specified target types.
            The targets represent:

                - ``attr`` (np.array shape=(40,) dtype=int): binary (0, 1) labels for attributes
                - ``identity`` (int): label for each person (data points with the same identity are the same person)
                - ``bbox`` (np.array shape=(4,) dtype=int): bounding box (x, y, width, height)
                - ``landmarks`` (np.array shape=(10,) dtype=int): landmark points (lefteye_x, lefteye_y, righteye_x,
                  righteye_y, nose_x, nose_y, leftmouth_x, leftmouth_y, rightmouth_x, rightmouth_y)

            Defaults to ``attr``. If empty, ``None`` will be returned as target.

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "celeba_hq"
    # There currently does not appear to be a easy way to extract 7z in python (without introducing additional
    # dependencies). The "in-the-wild" (not aligned+cropped) images are only in 7z, so they are not available
    # right now.
    file_list = [
        # File ID                                      MD5 Hash                            Filename
        # ("0B7EVK8r0v71pZjFTYXZWM3FlRnM", "00d2c5bc6d35e252742224ab0c1e8fcb", "img_align_celeba.zip"),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc","b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        ("0B7EVK8r0v71pblRyaVFSWGxPY0U", "75e246fa4810816ffd6ee81facbd244c", "list_attr_celeba.txt"),
        # ("1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS", "32bd1bd63d3c78cd57e08160ec5ed1e2", "identity_CelebA.txt"),
        # ("0B7EVK8r0v71pbThiMVRxWXZ4dU0", "00566efa6fedff7a56946cd1c10f1c16", "list_bbox_celeba.txt"),
        # ("0B7EVK8r0v71pd0FJY3Blby1HUTQ", "cc24ecafdb5b50baae59b03474781f8c", "list_landmarks_align_celeba.txt"),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        ("0B7EVK8r0v71pY0NSMzRuSXJEVkk", "d32c9cbf5e040fd4025c592c306e6668", "list_eval_partition.txt"),
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        target_type: Union[List[str], str] = "attr",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.classes = [0, 1]
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError("target_transform is specified but target_type is empty")

        # if download:
        #     self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]
        splits = self._load_csv("list_eval_partition.txt")
        # identity = self._load_csv("identity_CelebA.txt")
        # bbox = self._load_csv("list_bbox_celeba.txt", header=1)
        # landmarks_align = self._load_csv("list_landmarks_align_celeba.txt", header=1)
        attr = self._load_csv("list_attr_celeba.txt", header=1)

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        if mask == slice(None):  # if split == "all"
            self.filename = splits.index
        else:
            self.filename = [splits.index[i] for i in torch.squeeze(torch.nonzero(mask))]
        # self.identity = identity.data[mask]
        # self.bbox = bbox.data[mask]
        # self.landmarks_align = landmarks_align.data[mask]
        self.attr = attr.data[mask]
        # map from {-1, 1} to {0, 1}
        self.attr = torch.div(self.attr + 1, 2, rounding_mode="floor")
        self.attr_names = attr.header

    def _load_csv(
        self,
        filename: str,
        header: Optional[int] = None,
    ) -> CSV:
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1 :]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    def _check_integrity(self) -> bool:
        return os.path.isdir(os.path.join(self.root, self.base_folder, "data256"))


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "data256", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError(f'Target type "{t}" is not recognized.')

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target[20].item()


    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return "\n".join(lines).format(**self.__dict__)



class bFFHQForGender(torch.utils.data.Dataset):
    base_folder = 'bffhq'
    # target_attr_index = 0
    # bias_attr_index = 1

    def __init__(self, root, split, transform=None):
        super(bFFHQForGender, self).__init__()
        self.transform = transform
        root = os.path.join(root, self.base_folder)

        self.root = root

        if split == 'train':
            self.align = glob(os.path.join(root, split, 'align', "*", "*"))
            self.conflict = glob(os.path.join(root, split, 'conflict', "*", "*"))
            self.data = self.align + self.conflict

        elif split == 'valid':
            self.data = glob(os.path.join(root, split, "*"))

        elif split == 'test':
            self.data = glob(os.path.join(root, split, "*"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fpath = self.data[index]
        age_attr = int(fpath.split('_')[-2])
        gender_attr = int(fpath.split('_')[-1].split('.')[0])
        # attr = torch.LongTensor([first_attr, second_attr])
        image = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, gender_attr

