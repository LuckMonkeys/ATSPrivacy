# Moving classes image from idetnity_dir/train_src_classes to idetntity_dir / train_num_classes
from pathlib import Path
import os
from shutil import copytree
from natsort import os_sorted

identity_dir = '/home/zx/nfs/server3/data/celeba/celeba_identity'
sub_dirs = ['train', 'test']

src_classes = 1000
num_classes = 100

if not os.path.exists(identity_dir + f'/train_{num_classes}'):
    os.mkdir(identity_dir + f'/train_{num_classes}')

if not os.path.exists(identity_dir + f'/test_{num_classes}'):
    os.mkdir(identity_dir + f'/test_{num_classes}')


class_names = os_sorted(list(os.listdir(identity_dir + f'/train_{src_classes}'))) 

for cls in class_names[:num_classes]:
    for sub_dir in sub_dirs:
        src_dir = identity_dir + f'/{sub_dir}_{src_classes}/{cls}'
        dst_dir = identity_dir + f'/{sub_dir}_{num_classes}/{cls}'
        copytree(src_dir, dst_dir, dirs_exist_ok=True)
        

    
