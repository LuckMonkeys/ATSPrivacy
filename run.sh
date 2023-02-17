# ----------------------------------------------------------------vit on Cifar100--------------------------------------------------------------------------------
##Train original and tiny_data model 
# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=cifar100 --arch=vit --epochs=10 --aug_list='' --mode=crop 

##search
# CUDA_VISIBLE_DEVICES=0 python benchmark/search_transform_attack.py --aug_list=6-18-30 --mode=aug --arch=vit --data=cifar100 --epochs=10 --num_samples 50 &

##Test candidate policy acc
# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=cifar100 --arch=vit --epochs=10 --aug_list='15-37-34' --mode=aug & #93.63

##Attack the policy
# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=cifar100 --arch=vit --epochs=10 --aug_list='' --mode=normal --optim=inversed_large
# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=cifar100 --arch=vit --epochs=10 --aug_list='' --mode=normal --optim=inversed_large
# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=cifar100 --arch=vit --epochs=10 --aug_list='3-2-38' --mode=aug --optim=inversed_large

#find policy
##train vit on tiny_cifar100
# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=cifar100 --arch=vit --epochs=10 --aug_list='' --mode=crop
# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=cifar100 --arch=vit --epochs=10 --aug_list='' --mode=crop --tiny_data

##search
# CUDA_VISIBLE_DEVICES=0 python benchmark/search_transform_attack.py --aug_list=6-18-30 --mode=aug --arch=vit --data=cifar100 --epochs=5

# ----------------------------------------------------------------ResNet on ImageNet------------------------------------------------------------------
# Epoch 30
##train resnet on ori_imagenet
# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='' --mode=crop
##train resnet on tiny_imagenet
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=ImageNet --arch=ResNet18_tv --epochs=10 --aug_list='' --mode=crop --tiny_data

##Attack ori imagenet
# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='' --mode=normal --optim=inversed_large

##search policy
# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='24-3-33' --mode=aug &
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='43-21-21' --mode=aug &

# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='26-3-15+43-21-21' --mode=aug &

##attack search policy

#The default validation set image is in same class
# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='' --mode=normal --optim=inversed_large
# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='26-3-15' --mode=aug --optim=inversed_large
# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='43-21-21' --mode=aug --optim=inversed_large

# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='21-13-3' --mode=aug --optim=inversed_large --fix_ckpt

# ----------------------------------------------------------------ResNet on CelebA Gender Classification--------------------------------------------------------------------------------
##Train original and tiny_data model 
# CUDA_VISIBLE_DEVICES=2 python -u benchmark/cifar100_train.py --data=CelebA --arch=ResNet18_tv --epochs=2 --aug_list='' --mode=crop #97.31
# CUDA_VISIBLE_DEVICES=2 python -u benchmark/cifar100_train.py --data=CelebA --arch=ResNet18_tv --epochs=2 --aug_list='' --mode=crop --tiny_data

###Rerun for size 112*112
# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=CelebA --arch=ResNet18_tv --epochs=2 --aug_list='' --mode=crop
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebA --arch=ResNet18_tv --epochs=2 --aug_list='' --mode=crop --tiny_data

##search
# CUDA_VISIBLE_DEVICES=0 python benchmark/search_transform_attack.py --aug_list=6-18-30 --mode=aug --arch=ResNet18_tv --data=CelebA --epochs=2 --num_samples 50 &
# CUDA_VISIBLE_DEVICES=0 python benchmark/search_transform_attack.py --aug_list=41-28-11 --mode=aug --arch=ResNet18_tv --data=CelebA --epochs=2 --num_samples 50 &
# CUDA_VISIBLE_DEVICES=0 python benchmark/search_transform_attack.py --aug_list=6-18-30 --mode=aug --arch=ResNet18_tv --data=CelebA --epochs=2 --num_samples 50 &
# CUDA_VISIBLE_DEVICES=0 python benchmark/search_transform_attack.py --aug_list=41-28-11 --mode=aug --arch=ResNet18_tv --data=CelebA --epochs=2 --num_samples 50 &

##Test candidate policy
# origin: 97.31
# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=CelebA --arch=ResNet18_tv --epochs=2 --aug_list='15-37-34' --mode=aug & #93.63
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebA --arch=ResNet18_tv --epochs=2 --aug_list='3-2-38' --mode=aug & # 62.94
# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=CelebA --arch=ResNet18_tv --epochs=2 --aug_list='20-3-38' --mode=aug & #60.03
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebA --arch=ResNet18_tv --epochs=2 --aug_list='3-14-39' --mode=aug & # 57.05
# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=CelebA --arch=ResNet18_tv --epochs=2 --aug_list='3-20-20' --mode=aug & # 58.66
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebA --arch=ResNet18_tv --epochs=2 --aug_list='13-30-19' --mode=aug & # 96.14


# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebA --arch=ResNet18_tv --epochs=2 --aug_list='3-2-38+3-20-20+20-3-38+3-14-39' --mode=aug
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebA --arch=ResNet18_tv --epochs=2 --aug_list='3-2-38+3-20-20+2-43-21' --mode=aug

##Attack the policy
# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=CelebA --arch=ResNet18_tv --epochs=2 --aug_list='' --mode=normal --optim=inversed_large
# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=CelebA --arch=ResNet18_tv --epochs=2 --aug_list='15-37-34' --mode=aug --optim=inversed_large & #less effect
# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=CelebA --arch=ResNet18_tv --epochs=2 --aug_list='13-30-19' --mode=aug --optim=inversed_large & #less effect


# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=CelebA --arch=ResNet18_tv --epochs=2 --aug_list='3-2-38' --mode=aug --optim=inversed_large --fix_ckpt


# ----------------------------------------------------------------ResNet on CelebA Face Recongnition--------------------------------------------------------------------------------

# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=CelebA_Identity --arch=ResNet18_tv --epochs=100 --aug_list='' --mode=crop

# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=CelebA_Identity --arch=ResNet18_tv --epochs=100 --aug_list='' --mode=normal --optim=inversed_large
# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=CelebA_Identity --arch=ResNet18_tv --epochs=100 --aug_list='3-20-20' --mode=aug --optim=inversed_large --fix_ckpt



# ----------------------------------------------------------------ResNet on CelebA Face Align Multilabel--------------------------------------------------------------------------------
#train
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=crop
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=crop --tiny_data
# CUDA_VISIBLE_DEVICES=2 python -u benchmark/cifar100_train.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='3-2-38' --mode=aug

# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='15-8' --mode=aug #完全没有防御效果
# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='26-19-31' --mode=aug
# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='38-8-8' --mode=aug
# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='19-34' --mode=aug




# Search policy

# CUDA_VISIBLE_DEVICES=1 python benchmark/search_transform_attack_seq.py --aug_list='' --mode=aug --arch=ResNet18_tv --data=CelebAFaceAlign_MLabel --epochs=3  --aug_file candidate_policies_0 copy.txt

# CUDA_VISIBLE_DEVICES=1 python benchmark/search_transform_attack_seq.py --aug_list='' --mode=aug --arch=ResNet18_tv --data=CelebAFaceAlign_MLabel --epochs=3  --aug_file candidate_policies_0.txt
# CUDA_VISIBLE_DEVICES=2 python benchmark/search_transform_attack_seq.py --aug_list='' --mode=aug --arch=ResNet18_tv --data=CelebAFaceAlign_MLabel --epochs=3  --aug_file candidate_policies_1.txt 
# CUDA_VISIBLE_DEVICES=3 python benchmark/search_transform_attack_seq.py --aug_list='' --mode=aug --arch=ResNet18_tv --data=CelebAFaceAlign_MLabel --epochs=3  --aug_file candidate_policies_2.txt 
# CUDA_VISIBLE_DEVICES=4 python benchmark/search_transform_attack_seq.py --aug_list='' --mode=aug --arch=ResNet18_tv --data=CelebAFaceAlign_MLabel --epochs=3  --aug_file candidate_policies_3.txt 
# CUDA_VISIBLE_DEVICES=5 python benchmark/search_transform_attack_seq.py --aug_list='' --mode=aug --arch=ResNet18_tv --data=CelebAFaceAlign_MLabel --epochs=3  --aug_file candidate_policies_4.txt 
# CUDA_VISIBLE_DEVICES=6 python benchmark/search_transform_attack_seq.py --aug_list='' --mode=aug --arch=ResNet18_tv --data=CelebAFaceAlign_MLabel --epochs=3  --aug_file candidate_policies_5.txt 
# CUDA_VISIBLE_DEVICES=7 python benchmark/search_transform_attack_seq.py --aug_list='' --mode=aug --arch=ResNet18_tv --data=CelebAFaceAlign_MLabel --epochs=3  --aug_file candidate_policies_6.txt 


#Test candidate policy
# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large
# CUDA_VISIBLE_DEVICES=2 python benchmark/cifar100_attack.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='3-2-38' --mode=aug --optim=inversed_large

# CUDA_VISIBLE_DEVICES=2 python benchmark/cifar100_attack.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='15-37-34' --mode=aug --optim=inversed_large
# 


# ----------------------------------------------------------------ConvNet8 FMNIST For DatasetCondensation------------------------
# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=DM_FashionMinist --arch=ConvNet8_embed --epoch=2000 --aug_list='' --mode=DM --optim=inversed
# #dataname data.pt, state_dict name: ConvNet8_2000.pth
# #

# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=DC_FashionMinist --arch=ConvNet8_embed --epoch=1000 --aug_list='' --mode=DC --optim=inversed

# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=DSA_FashionMinist --arch=ConvNet8_embed --epoch=1000 --aug_list='' --mode=DSA --optim=inversed



# ----------------------------------------------------------------ConvNet8 FMNIST For DatasetCondensation-----------------------

## Rerun cifar100 ResNet20-4

# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='42-1-12' --mode=aug
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='7-23-48' --mode=aug


# ----------------------------------------------------------------Combine top n augmentations-----------------------

# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='3' --mode=aug
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='3-24' --mode=aug
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='3-24-15' --mode=aug



# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=cifar100 --arch=ResNet20-4 --epoch=200 --aug_list='3' --mode=aug --optim=inversed
# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=cifar100 --arch=ResNet20-4 --epoch=200 --aug_list='3-24' --mode=aug --optim=inversed
# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=cifar100 --arch=ResNet20-4 --epoch=200 --aug_list='3-24-15' --mode=aug --optim=inversed
# 



# ----------------------------------------------------------------ConvNet8 FMNIST DP MERF----------------------------------------

# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='42-1-12' --mode=aug
#
# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_train.py --data=DP_MERF_FashionMinist --arch=ConvNet8 --epoch=10 --aug_list='' --mode=GAN

# CUDA_VISIBLE_DEVICES=3 python benchmark/cifar100_attack.py --data=DP_MERF_FashionMinist  --arch=ConvNet8 --epoch=10 --aug_list='' --mode=GAN --optim=inversed


# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_train.py --data=GS_WGAN_FashionMinist --arch=ConvNet8 --epoch=10 --aug_list='' --mode=GAN

CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=GS_WGAN_FashionMinist  --arch=ConvNet8 --epoch=10 --aug_list='' --mode=GAN --optim=inversed



# ----------------------------------------------------------------cifar100 renset20 baseline----------------------------------------

CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='' --mode=crop
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='' --mode=normal

# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='3-24' --mode=aug
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='3-24-15' --mode=aug
# 


# ----------------------------------------------------------------rerun the iteration for same times ----------------------------------------
# 



# ----------------------------------------------------------------attack celeba save intermediate result ----------------------------------------

CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack_celeba.py --data=CelebA_Gender --arch=ResNet18_tv --epochs=2 --aug_list='' --mode=normal --optim=inversed_large --save_verbose


CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack_celeba.py --data=CelebA_Gender --arch=ResNet18_tv --epochs=2 --aug_list='' --mode=normal --optim=inversed_large --save_verbose --init_sameattr



CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack_compare_grad.py --data=CelebA_Gender --arch=ResNet18_tv --epochs=2 --aug_list='' --mode=normal --optim=inversed_large --save_verbose




# ----------------------------------------------------------------Combine top n augmentations-----------------------

CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='3' --mode=aug
CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='3-15' --mode=aug
CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='3-15-18' --mode=aug



CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=cifar100 --arch=ResNet20-4 --epoch=200 --aug_list='3' --mode=aug --optim=inversed
CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=cifar100 --arch=ResNet20-4 --epoch=200 --aug_list='3-15' --mode=aug --optim=inversed
CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=cifar100 --arch=ResNet20-4 --epoch=200 --aug_list='3-15-18' --mode=aug --optim=inversed




CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='3-43' --mode=aug
CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='3-43-15' --mode=aug



CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=cifar100 --arch=ResNet20-4 --epoch=200 --aug_list='3-43' --mode=aug --optim=inversed
CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=cifar100 --arch=ResNet20-4 --epoch=200 --aug_list='3-43-15' --mode=aug --optim=inversed




CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack_celeba.py --data=CelebAHQ_Gender --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --size=112



CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack_celeba.py --data=CelebAHQ_Gender --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --size=224 --num_samples=5

#need to execute
CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack_celeba.py --data=CelebAHQ_Gender --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed-adam-L2_large --size=224 --num_samples=5






# ----------------------------------------------------------------bFFHQ dataset train-----------------------
#train
CUDA_VISIBLE_DEVICES=0 python -u benchmark/bffhq_train.py --data=bFFHQ_Gender --arch=ResNet18_tv --epochs=50 --aug_list='' --mode=crop 
#attack
CUDA_VISIBLE_DEVICES=1 python benchmark/bffhq_attack.py --data=bFFHQ_Gender --arch=ResNet18_tv --epoch=50 --aug_list='' --mode=normal --optim=inversed_large --size=112 --num_samples=5