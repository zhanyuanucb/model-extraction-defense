CUDA_VISIBLE_DEVICES=1 python benign_user.py -l CIFAR10 \
                      --log_suffix=benign_cifar_xnorm \
                      --return_conf_max \
                      --device_id=1
#                      
#python benign_user.py -l CIFAR10 \
#                      --k=10 \
#                      --thresh=0.13513331776857376 \
#                      --log_suffix=benign_cifar_k10 \
#                      --return_conf_max \
#                      --device_id=0                     
                      
#python benign_user.py -l CIFAR10 \
#                      --encoder_arch_name=wrn28 \
#                       --k=5 \
#                      --thresh=0.8019031870365143\
#                      --log_suffix=benign_wrn28_cifar_k5 \
#                      --return_conf_max \
#                      --device_id=0
#                      
# python benign_user.py -l CIFAR10 \
#                      --encoder_arch_name=wrn28 \
#                      --k=10 \
#                      --thresh=0.9607433207035065 \
#                      --log_suffix=benign_wrn28_cifar_k10 \
#                      --return_conf_max \
#                      --device_id=0                     
                      