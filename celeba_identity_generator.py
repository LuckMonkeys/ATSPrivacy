from collections import defaultdict

identities = defaultdict(list)


with open('/home/zx/data/celeba/identity_CelebA.txt') as f:
    lines = f.readlines()
    for line in lines:
        file_name, identity = line.strip().split()
        # identities[file_name] = identity
        identities[identity].append(file_name)

print(f'There are {len(set(identities.keys()))} identities.')
# print(f'There are {len(identities.keys())} images.')

import os
from shutil import copyfile
from natsort import os_sorted

source_root = '/home/zx/data/celeba/img_align_celeba'
target_root = '/home/zx/data/celeba/celeba_identity'

target_number = 30
train_num = 25
num_classes = 500
qualified_keys = []
for key, value in identities.items():
    if len(value) == target_number:
        qualified_keys.append(key)

qualified_keys = os_sorted(qualified_keys)
assert len(qualified_keys) >= num_classes, 'Num classes to large'
# print(qualified_keys, len(qualified_keys))
# exit(0)

for key in qualified_keys[:num_classes]:
    for name in identities[key][:train_num]:
        src_path =  source_root + f'/{name}' 
        dst_path =  target_root + f'/train_{num_classes}/{key}/{name}'

        if not os.path.exists(target_root + f'/train_{num_classes}/{key}') :
            os.makedirs(target_root + f'/train_{num_classes}/{key}')

        copyfile(src_path, dst_path)
    
    for name in identities[key][train_num:]:
        src_path =  source_root + f'/{name}' 
        dst_path =  target_root + f'/test_{num_classes}/{key}/{name}'

        if not os.path.exists(target_root + f'/test_{num_classes}/{key}') :
            os.makedirs(target_root + f'/test_{num_classes}/{key}')

        copyfile(src_path, dst_path)
    # break