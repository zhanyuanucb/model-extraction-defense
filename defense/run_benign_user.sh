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

encoder_name=resnet34
CUDA_VISIBLE_DEVICES=1 python benign_user.py -l CINIC10 \ 
                      --encoder_arch_name=$encoder_name \
                      --k=1 \
                      --thresh=0.054156411439180374 \
                      --log_suffix=benign_multi_cluster_$encoder_name \
                      --device_id=1
#                      
# python benign_user.py -l CIFAR10 \
#                      --encoder_arch_name=wrn28 \
#                      --k=10 \
#                      --thresh=0.9607433207035065 \
#                      --log_suffix=benign_wrn28_cifar_k10 \
#                      --return_conf_max \
#                      --device_id=0                     
                      