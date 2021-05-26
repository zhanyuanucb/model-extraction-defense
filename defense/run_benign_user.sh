#CUDA_VISIBLE_DEVICES=0 python benign_user.py -l CIFAR10 \
#                      --log_suffix=benign_cifar_adv_train \
#                      --return_conf_max \
#                      --device_id=0
                      
#python benign_user.py -l CIFAR10 \
#                      --k=10 \
#                      --thresh=0.13513331776857376 \
#                      --log_suffix=benign_cifar_k10 \
#                      --return_conf_max \
#                      --device_id=0                     
                      
#encoder_name=resnet34
#CUDA_VISIBLE_DEVICES=1 python benign_user.py -l CIFAR10 CINIC10 \
#                      --encoder_arch_name=$encoder_name \
#                      --k=1 \
#                      --thresh=0.054156411439180374 \
#                      --log_suffix=benign_multi_cluster_$encoder_name \
#                      --device_id=1

#encoder_name=resnet34
#CUDA_VISIBLE_DEVICES=1 python benign_user.py -l CIFAR10 CINIC10 \ 
#                      --encoder_arch_name=$encoder_name \
#                      --k=1 \
#                      --thresh=0.054156411439180374 \
#                      --log_suffix=benign_multi_cluster_$encoder_name \
#                      --device_id=1
                      
#encoder_arch_name="simnet"                      
#encoder_suffix="_xaug"
#for thresh in 0.195
#do
#CUDA_VISIBLE_DEVICES=0 python benign_user.py \
#                      --encoder_arch_name=$encoder_arch_name \
#                      --encoder_margin=3.2 \
#                      --encoder_suffix=$encoder_suffix \
#                      --thresh=$thresh \
#                      --k=1 \
#                      --log_suffix=$encoder_arch_name \
#                      -l CIFAR10 CINIC10 \
#                      --device_id=0
#done                      
                      
# --thresh=0.0014249234207673                      
#                      --thresh=0.002088276777882129 \
#                      --thresh=0.0037130360356532033 \

# python benign_user.py -l CIFAR10 \
#                      --encoder_arch_name=wrn28 \
#                      --k=10 \
#                      --thresh=0.9607433207035065 \
#                      --log_suffix=benign_wrn28_cifar_k10 \
#                      --return_conf_max \
#                      --device_id=0                     

#encoder_arch_name="simnet"                      
#encoder_suffix=""
#thresh=0.16197727304697038
#CUDA_VISIBLE_DEVICES=1 python benign_user.py \
#                      --encoder_arch_name=$encoder_arch_name \
#                      --encoder_margin=3.2 \
#                      --encoder_suffix=$encoder_suffix \
#                      --thresh=$thresh \
#                      --k=1 \
#                      --log_suffix=$encoder_arch_name \
#                      -l CIFAR10 CINIC10\
#                      --device_id=1

#encoder_arch_name="vgg16_bn"                      
#encoder_suffix=""
##thresh=1.25
##encoder_arch_name="simnet"                      
##encoder_suffix="_ep30"
#CUDA_VISIBLE_DEVICES=1 python benign_user.py \
#                      --encoder_arch_name=$encoder_arch_name \
#                      --encoder_margin=3.2 \
#                      --encoder_suffix=$encoder_suffix \
#                      --lower=1e-3 \
#                      --upper=3. \
#                      --k=1 \
#                      --log_suffix=$encoder_arch_name \
#                      -l CINIC10 CIFAR10 \
#                      --device_id=1

encoder_ckpt="/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding/"
lk_ckpt_benign="/mydata/model-extraction/model-extraction-defense/defense/likelihood_estimation/vq.ckpt"
pixelcnn_ckpt="/mydata/model-extraction/model-extraction-defense/defense/likelihood_estimation/pixelcnn_ckpt/2021-04-06_04:43:26-pixelcnn-cifar10/pixelcnn.ckpt"
#lk_ckpt="/mydata/model-extraction/model-extraction-defense/defense/likelihood_estimation/vq-vae_ckpt/2021-03-22_00:49:29-cifar10-no-div-var/vq.ckpt"
lk_ckpt_adv_cinic="/mydata/model-extraction/model-extraction-defense/defense/likelihood_estimation/vq-vae_ckpt/2021-03-22_02:47:25-cinic10-no_var/vq.ckpt"
lk_ckpt_adv_cifar_seed="/mydata/model-extraction/model-extraction-defense/defense/likelihood_estimation/vq-vae_ckpt/2021-03-26_05:11:55-cifar10_seed-strvar/vq.ckpt"

lk_ckpt_benign2="/mydata/model-extraction/pytorch-vqvae/models/models/vqvae/best.pt"
pixelcnn_ckpt2="/mydata/model-extraction/pytorch-vqvae/models/models/pixelcnn_prior/prior.pt"
elbo_thresh=-7.331564728763727
vae_thresh=-1.298600417103578
llk_envelope=0.3763033370971679
llk_envelope_2way=0.7369010351350298

encoder_arch_name="simnet"                      
encoder_suffix="_ep30_pgd_epsx3"
simnet_ep30_thresh=0.12986328125
simnet_ep30_pgd_thresh=0.014087677001953125
device_id=1
#for querydata in 'ImageNet1k' 'CIFAR100' 'CINIC10' 
#for querydata in 'CIFAR100' 'CINIC10' 'Indoor67' 'Caltech256' 'TinyImageNet200'  'ImageNet1k' 'SVHN' 'CUBS200' 
#for querydata in 'CIFAR10' 'CIFAR100' 'CINIC10' 'Indoor67' 'Caltech256' 'TinyImageNet200'  'ImageNet1k' 'SVHN' 'CUBS200'
#for querydata in 'TinyImageNet200'
#do
CUDA_VISIBLE_DEVICES=$device_id python benign_user.py \
                      --encoder_arch_name=$encoder_arch_name \
                      --encoder_suffix=$encoder_suffix \
                      --batch_size=128 \
                      --encoder_ckpt=$encoder_ckpt \
                      --encoder_margin=3.2 \
                      --thresh_search \
                      --lower=1e-6 \
                      --upper=3. \
                      --k=1 \
                      --log_suffix=ep30_pgd_epsx3 \
                      -l 'CIFAR10' 'CINIC10' \
                      --device_id=$device_id                     
#done
#    
#  'ImageNet1k' 'SVHN' 'CIFAR100' 'CINIC10' 'Indoor67' 'CUBS200' 'Caltech256' 

#encoder_arch_name="simnet"                      
#encoder_suffix="_adv_seed5000"
#log_suffix="simnet_advseed5000-cluster10"
#CUDA_VISIBLE_DEVICES=1 python benign_user.py \
#                      --encoder_arch_name=$encoder_arch_name \
#                      --encoder_margin=3.2 \
#                      --encoder_suffix=$encoder_suffix \
#                      --thresh_search \
#                      --lower=1e-3 \
#                      --upper=3. \
#                      --k=10 \
#                      --log_suffix=$log_suffix \
#                      -l CIFAR10 CINIC10 \
#                      --device_id=1                                          