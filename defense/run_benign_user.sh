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
                      
python benign_user.py -l CIFAR10 \
                      --encoder_arch_name=vgg16_bn \
                      --k=5 \
                      --thresh=0.5826407364010812 \
                      --log_suffix=benign_adv_vggbn_cifar_k5 \
                      --device_id=0
#                      
# python benign_user.py -l CIFAR10 \
#                      --encoder_arch_name=wrn28 \
#                      --k=10 \
#                      --thresh=0.9607433207035065 \
#                      --log_suffix=benign_wrn28_cifar_k10 \
#                      --return_conf_max \
#                      --device_id=0                     
                      