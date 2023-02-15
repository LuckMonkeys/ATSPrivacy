# ----------------------------------------------------------------vit on Cifar100--------------------------------------------------------------------------------
##Train original and tiny_data model 
# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=cifar100 --arch=vit --epochs=10 --aug_list='' --mode=crop
# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=cifar100 --arch=vit --epochs=10 --aug_list='' --mode=crop --tiny_data

##search
# CUDA_VISIBLE_DEVICES=0 python benchmark/search_transform_attack.py --mode=aug --arch=vit --data=cifar100 --epochs=5 --num_samples 1 --aug_file candidate_policies.txt


# CUDA_VISIBLE_DEVICES=1 python benchmark/search_transform_attack_seq.py --aug_list='' --mode=aug --arch=vit --data=cifar100 --epochs=5 --num_samples 1 --aug_file candidate_policies_0.txt &
# CUDA_VISIBLE_DEVICES=2 python benchmark/search_transform_attack_seq.py --aug_list='' --mode=aug --arch=vit --data=cifar100 --epochs=5 --num_samples 1 --aug_file candidate_policies_1.txt &
# CUDA_VISIBLE_DEVICES=3 python benchmark/search_transform_attack_seq.py --aug_list='' --mode=aug --arch=vit --data=cifar100 --epochs=5 --num_samples 1 --aug_file candidate_policies_2.txt &
# CUDA_VISIBLE_DEVICES=4 python benchmark/search_transform_attack_seq.py --aug_list='' --mode=aug --arch=vit --data=cifar100 --epochs=5 --num_samples 1 --aug_file candidate_policies_3.txt &
# CUDA_VISIBLE_DEVICES=5 python benchmark/search_transform_attack_seq.py --aug_list='' --mode=aug --arch=vit --data=cifar100 --epochs=5 --num_samples 1 --aug_file candidate_policies_4.txt &
# CUDA_VISIBLE_DEVICES=6 python benchmark/search_transform_attack_seq.py --aug_list='' --mode=aug --arch=vit --data=cifar100 --epochs=5 --num_samples 1 --aug_file candidate_policies_5.txt &
# CUDA_VISIBLE_DEVICES=7 python benchmark/search_transform_attack_seq.py --aug_list='' --mode=aug --arch=vit --data=cifar100 --epochs=5 --num_samples 1 --aug_file candidate_policies_6.txt &



##Test candidate policy acc
#normal 87.83
# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=cifar100 --arch=vit --epochs=10 --aug_list='15-37-34' --mode=aug & #93.63

# CUDA_VISIBLE_DEVICES=4 python -u benchmark/cifar100_train.py --data=cifar100 --arch=vit --epochs=10 --aug_list='28-13-37' --mode=aug #82.60
# CUDA_VISIBLE_DEVICES=5 python -u benchmark/cifar100_train.py --data=cifar100 --arch=vit --epochs=10 --aug_list='20-3-38' --mode=aug #80.16
# CUDA_VISIBLE_DEVICES=6 python -u benchmark/cifar100_train.py --data=cifar100 --arch=vit --epochs=10 --aug_list='7-5-21' --mode=aug #85.68
# CUDA_VISIBLE_DEVICES=7 python -u benchmark/cifar100_train.py --data=cifar100 --arch=vit --epochs=10 --aug_list='43-1-18' --mode=aug #86.7


# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=cifar100 --arch=vit --epochs=10 --aug_list='20-3-38+7-5-21' --mode=aug #85.12

##Attack the policy
#normal 12.58

# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=cifar100 --arch=vit --epochs=10 --aug_list='' --mode=normal --optim=inversed_large
# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=cifar100 --arch=vit --epochs=10 --aug_list='3-2-38' --mode=aug --optim=inversed_large

# CUDA_VISIBLE_DEVICES=3 python benchmark/cifar100_attack.py --data=cifar100 --arch=vit --epochs=10 --aug_list='28-13-37' --mode=aug --optim=inversed_large --fix_ckpt
# CUDA_VISIBLE_DEVICES=4 python benchmark/cifar100_attack.py --data=cifar100 --arch=vit --epochs=10 --aug_list='20-3-38' --mode=aug --optim=inversed_large --fix_ckpt
# CUDA_VISIBLE_DEVICES=5 python benchmark/cifar100_attack.py --data=cifar100 --arch=vit --epochs=10 --aug_list='7-5-21' --mode=aug --optim=inversed_large --fix_ckpt
# CUDA_VISIBLE_DEVICES=6 python benchmark/cifar100_attack.py --data=cifar100 --arch=vit --epochs=10 --aug_list='43-1-18' --mode=aug --optim=inversed_large --fix_ckpt


# CUDA_VISIBLE_DEVICES=3 python benchmark/cifar100_attack.py --data=cifar100 --arch=vit --epochs=10 --aug_list='28-13-37' --mode=aug --optim=inversed_large #13.00 ?
# CUDA_VISIBLE_DEVICES=4 python benchmark/cifar100_attack.py --data=cifar100 --arch=vit --epochs=10 --aug_list='20-3-38' --mode=aug --optim=inversed_large #8.46
# CUDA_VISIBLE_DEVICES=5 python benchmark/cifar100_attack.py --data=cifar100 --arch=vit --epochs=10 --aug_list='7-5-21' --mode=aug --optim=inversed_large #8.74
# CUDA_VISIBLE_DEVICES=6 python benchmark/cifar100_attack.py --data=cifar100 --arch=vit --epochs=10 --aug_list='43-1-18' --mode=aug --optim=inversed_large #11.09


# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=cifar100 --arch=vit --epochs=10 --aug_list='20-3-38+7-5-21' --mode=aug --optim=inversed_large # 


# ----------------------------------------------------------------ResNet on ImageNet------------------------------------------------------------------
# Epoch 30
##train resnet on ori_imagenet
# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='' --mode=crop
##train resnet on tiny_imagenet
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=ImageNet --arch=ResNet18_tv --epochs=10 --aug_list='' --mode=crop --tiny_data

##Attack ori imagenet
# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='' --mode=normal --optim=inversed_large

##test policy
#normal 82.24

# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='24-3-33' --mode=aug &
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='43-21-21' --mode=aug &
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='26-3-15+43-21-21' --mode=aug &


# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='42-21-3' --mode=aug & #78.46
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='24-15-48' --mode=aug & #82.16

# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='18-9-37' --mode=aug 

# CUDA_VISIBLE_DEVICES=2 python -u benchmark/cifar100_train.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='42-21-3+24-15-48' --mode=aug & #84.75


##attack search policy
#normal 12.57

#The default validation set image is in same class
# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='' --mode=normal --optim=inversed_large
# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='26-3-15' --mode=aug --optim=inversed_large
# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='43-21-21' --mode=aug --optim=inversed_large

# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='21-13-3' --mode=aug --optim=inversed_large --fix_ckpt



# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='42-21-3' --mode=aug --optim=inversed_large --fix_ckpt &
# CUDA_VISIBLE_DEVICES=2 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='24-15-48' --mode=aug --optim=inversed_large --fix_ckpt &
# CUDA_VISIBLE_DEVICES=3 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='18-9-37' --mode=aug --optim=inversed_large --fix_ckpt &

# CUDA_VISIBLE_DEVICES=4 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='45-5-15' --mode=aug --optim=inversed_large --fix_ckpt &
# CUDA_VISIBLE_DEVICES=5 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='37-18-26' --mode=aug --optim=inversed_large --fix_ckpt &



# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='42-21-3' --mode=aug --optim=inversed_large &
# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='24-15-48' --mode=aug --optim=inversed_large &
# CUDA_VISIBLE_DEVICES=2 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='42-21-3+24-15-48' --mode=aug --optim=inversed_large &


# CUDA_VISIBLE_DEVICES=5 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='42-21-3' --mode=aug --optim=inversed_large 
# CUDA_VISIBLE_DEVICES=6 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='24-15-48' --mode=aug --optim=inversed_large 
# CUDA_VISIBLE_DEVICES=7 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='42-21-3+24-15-48' --mode=aug --optim=inversed_large 


# Test the transferability of cifar100 policy in imagenet25
# CUDA_VISIBLE_DEVICES=7 python -u benchmark/cifar100_train.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='13-43-18' --mode=aug 
# CUDA_VISIBLE_DEVICES=7 python -u benchmark/cifar100_train.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='21-3-16' --mode=aug 
# CUDA_VISIBLE_DEVICES=7 python -u benchmark/cifar100_train.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='13-43-18+21-3-16' --mode=aug 


CUDA_VISIBLE_DEVICES=5 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='13-43-18' --mode=aug --optim=inversed_large 
CUDA_VISIBLE_DEVICES=6 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='21-3-16' --mode=aug --optim=inversed_large 
CUDA_VISIBLE_DEVICES=7 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='13-43-18+21-3-16' --mode=aug --optim=inversed_large 



# ----------------------------------------------------------------ResNet on CelebA Gender--------------------------------------------------------------------------------
##Train original and tiny_data model 
# CUDA_VISIBLE_DEVICES=2 python -u benchmark/cifar100_train.py --data=CelebA_Gender --arch=ResNet18_tv --epochs=2 --aug_list='' --mode=crop
# CUDA_VISIBLE_DEVICES=2 python -u benchmark/cifar100_train.py --data=CelebA_Gender --arch=ResNet18_tv --epochs=2 --aug_list='' --mode=crop --tiny_data

###Rerun for size 112*112
# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=CelebA_Gender --arch=ResNet18_tv --epochs=2 --aug_list='' --mode=crop
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebA_Gender --arch=ResNet18_tv --epochs=2 --aug_list='' --mode=crop --tiny_data


##search
# CUDA_VISIBLE_DEVICES=0 python benchmark/search_transform_attack.py --aug_list=6-18-30 --mode=aug --arch=ResNet18_tv --data=CelebA_Gender --epochs=2 --num_samples 50 &
# CUDA_VISIBLE_DEVICES=0 python benchmark/search_transform_attack.py --aug_list=41-28-11 --mode=aug --arch=ResNet18_tv --data=CelebA_Gender --epochs=2 --num_samples 50 &
# CUDA_VISIBLE_DEVICES=0 python benchmark/search_transform_attack.py --aug_list=6-18-30 --mode=aug --arch=ResNet18_tv --data=CelebA_Gender --epochs=2 --num_samples 50 &
# CUDA_VISIBLE_DEVICES=0 python benchmark/search_transform_attack.py --aug_list=41-28-11 --mode=aug --arch=ResNet18_tv --data=CelebA_Gender --epochs=2 --num_samples 50 &

##Test candidate policy
# '24-3-47', '42-48-29'
# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=CelebA_Gender --arch=ResNet18_tv --epochs=2 --aug_list='24-3-47' --mode=aug &
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebA_Gender --arch=ResNet18_tv --epochs=2 --aug_list='42-48-29' --mode=aug &
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebA_Gender --arch=ResNet18_tv --epochs=2 --aug_list='3-11-6' --mode=aug &


##Attack the policy
# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=CelebA_Gender --arch=ResNet18_tv --epochs=2 --aug_list='' --mode=normal --optim=inversed_large



# ----------------------------------------------------------------ResNet on CelebA Face Recongnition--------------------------------------------------------------------------------

# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebA_Identity --arch=ResNet18_tv --epochs=100 --aug_list='' --mode=crop

# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=CelebA_Identity --arch=ResNet18_tv --epochs=100 --aug_list='' --mode=normal --optim=inversed_large
# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=CelebA_Identity --arch=ResNet18_tv --epochs=100 --aug_list='3-20-20' --mode=aug --optim=inversed_large --fix_ckpt




# ----------------------------------------------------------------ResNet on CelebA Multilabel--------------------------------------------------------------------------------

# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebA_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=crop
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebA_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='3-2-38' --mode=aug

# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebA_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='15-37-34' --mode=aug

# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=CelebA_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large
# CUDA_VISIBLE_DEVICES=2 python benchmark/cifar100_attack.py --data=CelebA_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='3-2-38' --mode=aug --optim=inversed_large

# CUDA_VISIBLE_DEVICES=2 python benchmark/cifar100_attack.py --data=CelebA_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='15-37-34' --mode=aug --optim=inversed_large


#test for ResNet32
CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebA_MLabel --arch=ResNet32 --epochs=10 --aug_list='' --mode=crop
CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_attack.py --data=CelebA_MLabel --arch=ResNet32 --epochs=10 --aug_list='' --mode=normal --optim=inversed_large

# ----------------------------------------------------------------ResNet on CelebA Face Align Multilabel----------------------------
#train
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=crop
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=crop --tiny_data
# CUDA_VISIBLE_DEVICES=2 python -u benchmark/cifar100_train.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='3-2-38' --mode=aug

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

# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='15-8' --mode=aug
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='26-19-31' --mode=aug
# CUDA_VISIBLE_DEVICES=2 python -u benchmark/cifar100_train.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='38-8-8' --mode=aug
# CUDA_VISIBLE_DEVICES=3 python -u benchmark/cifar100_train.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='19-34' --mode=aug

# CUDA_VISIBLE_DEVICES=3 python -u benchmark/cifar100_train.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='21-19' --mode=aug
# CUDA_VISIBLE_DEVICES=3 python -u benchmark/cifar100_train.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='21-19+3-2-38' --mode=aug


# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --start=19
# CUDA_VISIBLE_DEVICES=2 python benchmark/cifar100_attack.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='3-2-38' --mode=aug --optim=inversed_large
# CUDA_VISIBLE_DEVICES=2 python benchmark/cifar100_attack.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='15-37-34' --mode=aug --optim=inversed_large


# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='3-2-38' --mode=aug --optim=inversed_large
# CUDA_VISIBLE_DEVICES=2 python benchmark/cifar100_attack.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='21-19' --mode=aug --optim=inversed_large
# CUDA_VISIBLE_DEVICES=3 python benchmark/cifar100_attack.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='21-19+3-2-38' --mode=aug --optim=inversed_large


# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='15-8' --mode=aug --optim=inversed_large
# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='26-19-31' --mode=aug --optim=inversed_large
# CUDA_VISIBLE_DEVICES=2 python benchmark/cifar100_attack.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='38-8-8' --mode=aug --optim=inversed_large
# CUDA_VISIBLE_DEVICES=3 python benchmark/cifar100_attack.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='19-34' --mode=aug --optim=inversed_large



# ----------------------------------------------------------------ResNet on CelebAHQ Gender --------------------------------------------------------------------------------

# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebAHQ_Gender --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=crop

# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=CelebAHQ_Gender --arch=ResNet18_tv --epochs=3 --aug_list='3-2-38' --mode=aug




# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=CelebAHQ_Gender --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --end=10 --input_shape=112
# CUDA_VISIBLE_DEVICES=2 python benchmark/cifar100_attack.py --data=CelebAHQ_Gender --arch=ResNet18_tv --epochs=3 --aug_list='3-2-38' --mode=aug --optim=inversed_large



# ----------------------------------------------------------------ResNet on Scale CelebAHQ Gender --------------------------------------------------------------------------------

# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=Scale_CelebAHQ_Gender_28 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=crop --scale_data
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=Scale_CelebAHQ_Gender_56 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=crop --scale_data
# CUDA_VISIBLE_DEVICES=2 python -u benchmark/cifar100_train.py --data=Scale_CelebAHQ_Gender_168 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=crop --scale_data
# CUDA_VISIBLE_DEVICES=3 python -u benchmark/cifar100_train.py --data=Scale_CelebAHQ_Gender_224 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=crop --scale_data


# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=Scale_CelebAHQ_Gender_140 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=crop --scale_data

# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=Scale_CelebAHQ_Gender_28 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=10
# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=Scale_CelebAHQ_Gender_56 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=10
# CUDA_VISIBLE_DEVICES=2 python benchmark/cifar100_attack.py --data=Scale_CelebAHQ_Gender_168 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=10
# CUDA_VISIBLE_DEVICES=3 python benchmark/cifar100_attack.py --data=Scale_CelebAHQ_Gender_224 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=10


# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=Scale_CelebAHQ_Gender_140 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=10
# CUDA_VISIBLE_DEVICES=2 python benchmark/cifar100_attack.py --data=CelebAHQ_Gender --arch=ResNet18_tv --epochs=3 --aug_list='3-2-38' --mode=aug --optim=inversed_large


# CUDA_VISIBLE_DEVICES=0 python -u benchmark/cifar100_train.py --data=Scale_CelebAHQ_Gender_140 --arch=ResNet18_tv --epochs=3 --aug_list='3-2-38' --mode=aug --scale_data
# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=Scale_CelebAHQ_Gender_140 --arch=ResNet18_tv --epochs=3 --aug_list='3-2-38' --mode=aug --optim=inversed_large --scale_data --end=10

# ----------------------------------------------------------------ResNet on CelebA Face Align Multilabel Diff Input Shape--------------------------------------------------------------------------------

# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=CelebAFaceAlign_MLabel --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --end=10 --input_shape=56

# ----------------------------------------------------------------ResNet on CelebA Face Align Multilabel Test Gradient from Diff Size Input--------------------------------------------------------
# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack_copy.py --data=Scale_CelebAHQ_Gender_56 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=1
# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack_copy.py --data=Scale_CelebAHQ_Gender_112 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=1
# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack_copy.py --data=Scale_CelebAHQ_Gender_224 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=1
# 


# ----------------------------------------------------------------Rerun the iterations ablation experiment--------------------------------------------------------

# CUDA_VISIBLE_DEVICES=4 python benchmark/cifar100_attack.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='13-43-18+21-3-16' --mode=aug --optim=inversed --max_iterations 7800
# CUDA_VISIBLE_DEVICES=5 python benchmark/cifar100_attack.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='13-43-18+21-3-16' --mode=aug --optim=inversed --max_iterations 10800
# CUDA_VISIBLE_DEVICES=6 python benchmark/cifar100_attack.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='13-43-18+21-3-16' --mode=aug --optim=inversed --max_iterations 13800



# CUDA_VISIBLE_DEVICES=2 python benchmark/cifar100_attack.py --data=cifar100 --arch=ConvNet32 --epochs=100 --aug_list='21-13-3+15-48-15' --mode=aug --optim=inversed --max_iterations 7800 
# CUDA_VISIBLE_DEVICES=3 python benchmark/cifar100_attack.py --data=cifar100 --arch=ConvNet32 --epochs=100 --aug_list='21-13-3+15-48-15' --mode=aug --optim=inversed --max_iterations 10800
# CUDA_VISIBLE_DEVICES=4 python benchmark/cifar100_attack.py --data=cifar100 --arch=ConvNet32 --epochs=100 --aug_list='21-13-3+15-48-15' --mode=aug --optim=inversed --max_iterations 13800



# ----------------------------------------------------------------cifar100 renset20 baseline----------------------------------------

CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='' --mode=crop

CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=cifar100 --arch=ResNet20-4 --epoch=200 --aug_list='' --mode=normal --optim=inversed

# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='3-24' --mode=aug
# CUDA_VISIBLE_DEVICES=1 python -u benchmark/cifar100_train.py --data=cifar100 --arch=ResNet20-4 --epochs=200 --aug_list='3-24-15' --mode=aug
# 


# ----------------------------------------------------------------save middle result----------------------------------------

# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=Scale_CelebAHQ_Gender_28 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=10
# CUDA_VISIBLE_DEVICES=1 python benchmark/cifar100_attack.py --data=Scale_CelebAHQ_Gender_56 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=10

CUDA_VISIBLE_DEVICES=5 python benchmark/cifar100_attack.py --data=Scale_CelebAHQ_Gender_140 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=50 --save_verbose
CUDA_VISIBLE_DEVICES=6 python benchmark/cifar100_attack.py --data=Scale_CelebAHQ_Gender_168 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=50 --save_verbose
CUDA_VISIBLE_DEVICES=7 python benchmark/cifar100_attack.py --data=Scale_CelebAHQ_Gender_224 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=50 --save_verbose


##init from same laebl image

CUDA_VISIBLE_DEVICES=5 python benchmark/cifar100_attack_celebahq.py --data=Scale_CelebAHQ_Gender_140 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=50 --save_verbose --init_sameattr
CUDA_VISIBLE_DEVICES=6 python benchmark/cifar100_attack_celebahq.py --data=Scale_CelebAHQ_Gender_168 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=50 --save_verbose --init_sameattr
CUDA_VISIBLE_DEVICES=7 python benchmark/cifar100_attack_celebahq.py --data=Scale_CelebAHQ_Gender_224 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=50 --save_verbose --init_sameattr




CUDA_VISIBLE_DEVICES=5 python benchmark/cifar100_attack.py --data=Scale_CelebAHQ_Gender_140 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --start=50 --save_verbose
CUDA_VISIBLE_DEVICES=6 python benchmark/cifar100_attack.py --data=Scale_CelebAHQ_Gender_168 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --start=50 --save_verbose
CUDA_VISIBLE_DEVICES=7 python benchmark/cifar100_attack.py --data=Scale_CelebAHQ_Gender_224 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --start=50 --save_verbose


CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack_compare_grad.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='' --mode=normal --optim=inversed_large 



# CUDA_VISIBLE_DEVICES=0 python benchmark/cifar100_attack.py --data=ImageNet --arch=ResNet18_tv --epochs=30 --aug_list='' --mode=normal --optim=inversed_large
# 




##init from random select images

CUDA_VISIBLE_DEVICES=5 python benchmark/cifar100_attack_celebahq.py --data=Scale_CelebAHQ_Gender_140 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=50 --save_verbose --init_random

CUDA_VISIBLE_DEVICES=6 python benchmark/cifar100_attack_celebahq.py --data=Scale_CelebAHQ_Gender_168 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=50 --save_verbose --init_random
CUDA_VISIBLE_DEVICES=7 python benchmark/cifar100_attack_celebahq.py --data=Scale_CelebAHQ_Gender_224 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=50 --save_verbose --init_random


## test sr in celebahq
#


CUDA_VISIBLE_DEVICES=5 python benchmark/cifar100_attack_celebahq_sr.py --data=Scale_CelebAHQ_Gender_224 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=10 --save_verbose  --sr_scale=4

CUDA_VISIBLE_DEVICES=6 python benchmark/cifar100_attack_celebahq_sr.py --data=Scale_CelebAHQ_Gender_224 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=50 --save_verbose --init_sameattr --sr_scale=4

CUDA_VISIBLE_DEVICES=7 python benchmark/cifar100_attack_celebahq_sr.py --data=Scale_CelebAHQ_Gender_224 --arch=ResNet18_tv --epochs=3 --aug_list='' --mode=normal --optim=inversed_large --scale_data --end=50 --save_verbose --init_random --sr_scale=4


## test sr and random select images in celebahq 