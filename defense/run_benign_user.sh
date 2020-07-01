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
                      
CUDA_VISIBLE_DEVICES=0 python benign_user.py \
                      --encoder_arch_name=resnet34 \
                      --encoder_margin=3.2 \
                      --thresh=0.003130266818916425 \
                      --activation="sigmoid" \
                      --k=1 \
                      --log_suffix=benign_resnet34 \
                      -l CIFAR10 CINIC10 \
                      --device_id=0
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
                      