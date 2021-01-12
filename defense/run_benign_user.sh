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

encoder_arch_name="vgg16_bn"                      
for thresh in 1 1.5 1.8
do
CUDA_VISIBLE_DEVICES=1 python benign_user.py \
                      --encoder_arch_name=$encoder_arch_name \
                      --encoder_margin=3.2 \
                      --encoder_suffix=$encoder_suffix \
                      --thresh=$thresh \
                      --k=1 \
                      --log_suffix=$encoder_arch_name \
                      -l CINIC10 \
                      --device_id=1
done                                        