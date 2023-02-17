# identities = {}

# with open('/home/zx/nfs/server3/data/celeba/identity_CelebA.txt') as f:
#     lines = f.readlines()
#     for line in lines:
#         file_name, identity = line.strip().split()
#         identities[file_name] = identity

# print(f'There are {len(set(identities.values()))} identities.')
# print(f'There are {len(identities.keys())} images.')


# import os
# from shutil import copyfile


# source_root = '/home/zx/nfs/server3/data/celeba/img_align_celeba/'
# target_root = '/home/zx/nfs/server3/data/celeba/identity_dataset/'
# file_list = os.listdir(source_root)

# for file in file_list:
#     identity = identities[file]
#     source = os.path.join(source_root, file)
#     target = os.path.join(target_root, str(identity), file)
#     if not os.path.exists(os.path.join(target_root, str(identity))):
#         os.makedirs(os.path.join(target_root, str(identity)))
#     copyfile(source, target)

# folder_root = '/home/zx/nfs/server3/data/celeba/identity_dataset/'
# folder_list = os.listdir(folder_root)

# threshold = 30
# identity_cnt = 0

# train_images = 0
# test_images = 0
# # train_ratio = 0.8
# num_train = 25

# for folder in folder_list:
#     file_list = os.path.join(folder_root, folder)
#     file_list = os.listdir(file_list)
#     if len(file_list) >= threshold:
#         identity_cnt += 1
#         # num_train = int(train_ratio * len(file_list))
#         for file in file_list[:num_train]:
#             train_images += 1
#             source = os.path.join(folder_root, folder, file)
#             target = os.path.join(folder_root, 'train', folder, file)
#             if not os.path.exists(os.path.join(folder_root, 'train', folder)):
#                 os.makedirs(os.path.join(folder_root, 'train', folder))
#             # os.rename(source, target)
#             copyfile(source, target)
#         for file in file_list[num_train:]:
#             test_images += 1
#             source = os.path.join(folder_root, folder, file)
#             target = os.path.join(folder_root, 'test', folder, file)
#             if not os.path.exists(os.path.join(folder_root, 'test', folder)):
#                 os.makedirs(os.path.join(folder_root, 'test', folder))
#             copyfile(source, target)

# print(f'There are {identity_cnt} identities that have more than {threshold} images.')
# print(f'There are {train_images} train images.')
# print(f'There are {test_images} test images.')



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

source_root = '/home/zx/data/celeba/img_align_celeba/'
target_root = '/home/zx/data/celeba/celeba_identity/'

target_number = 30
train_num = 25
num_classes = 100
qualified_keys = []
for key, value in identities.items():
    if len(value) == target_number:
        qualified_keys.append(key)

qualified_keys = os_sorted(qualified_keys)
assert len(qualified_keys) >= num_classes, 'Num classes to large'

for key in qualified_keys[:num_classes]:
    for name in identities[key][:train_num]:
        src_path =  source_root + f'/{name}' 
        dst_path =  target_root + f'/train_{num_classes}/{key}/{name}'

        if not os.path.exists(target_root + f'/train_{num_classes}/{key}') :
            os.mkdir(target_root + f'/train_{num_classes}/{key}')

        copyfile(src_path, dst_path)
    
    for name in identities[key][train_num:]:
        src_path =  source_root + f'/{name}' 
        dst_path =  target_root + f'/test_{num_classes}/{key}/{name}'

        if not os.path.exists(target_root + f'/test_{num_classes}/{key}') :
            os.mkdir(target_root + f'/test_{num_classes}/{key}')

        copyfile(src_path, dst_path)
    






        