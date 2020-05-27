#python benign_user.py
CUDA_VIDIBLE_DEVICES=0 python benign_user.py -l CIFAR10 \
                      --log_suffix=benign_cifar_conf \
                      --return_conf_max

#CUDA_VIDIBLE_DEVICES=0 python benign_user.py -l CIFAR10 \
#                      --k=5 \
#                      --thresh=0.028577305126935244 \
#                      --log_suffix=benign_cifar_conf \
#                      --return_conf_max                     

#CUDA_VIDIBLE_DEVICES=0 python benign_user.py -l CINIC10 \
#                      --log_suffix=benign_cinic_norm_conf \
#                      --return_conf_max
                      
#CUDA_VIDIBLE_DEVICES=0 python benign_user.py -l CIFAR10 CINIC10 \
#                      --k=5 \
#                      --thresh=0.028577305126935244 \
#                      --log_suffix=benign_cifar_cinic_conf \
#                      --return_conf_max