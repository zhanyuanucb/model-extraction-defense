## 43.
#python main.py --momentum=0.5 \
#               --blinders_dir=/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/autoencoder_blind/phase2_

# 44.
#python main.py --momentum=0.5 \
#               --eps=0.1 \
# 45.
#python main.py --momentum=0.5 \
#               --eps=0.1 \
#               --blinders_dir=/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/autoencoder_blind/phase2_

# New logging since May 16
# ----------- Searching for optimal attack parameters
## 0. Baseline: 54.14%
#CUDA_VISIBLE_DEVICES=0 python main.py --phi=1 \
#               --log_suffix=baseline
#               --epochs=20
#               --random_adv

#CUDA_VISIBLE_DEVICES=0 python main.py --model_name=wrn28 \
#                                      --log_suffix=max_query \
#                                      --phi=19 \

#CUDA_VISIBLE_DEVICES=1 python main.py --model_name=wrn28 \
#                                      --phi=3 \
#                                      --delta_step=10 \
#                                      --log_suffix=cifar_delta3 \

#CUDA_VISIBLE_DEVICES=1 python main.py --model_name=wrn28 \
#                                      --phi=3 \
#                                      --delta_step=20 \
#                                      --log_suffix=cifar_delta3 \
# OOD attack
#CUDA_VISIBLE_DEVICES=0 python main.py --model_name=wrn28 \
#                                      --phi=1 \
#                                      --epochs=50 \
#                                      --log_suffix=cinic_35000\
#                                      --random_adv

#CUDA_VISIBLE_DEVICES=0 python main.py --model_name=wrn28 \
#                                      --log_suffix=cinic_blinder \
#                                      --blinders_dir=/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/autoencoder_blind/phase2_cinic10_0_
                                     
 
encoder_name="simnet"
#encoder_suffix="_xaug"
#thresh=0.195
encoder_suffix="_ep30"
thresh=0.225
#encoder_suffix="_random_start"
#thresh=0.0425

#CUDA_VISIBLE_DEVICES=0 python main.py --model_name="wrn28_2" \
#                                     --encoder_arch_name=$encoder_name \
#                                     --encoder_suffix=$encoder_suffix \
#                                     --encoder_margin=3.2 \
#                                     --thresh=$thresh \
#                                     --log_suffix=adv_cinic_autoencoder\
#                                     --blinders_dir=/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/autoencoder_blind/phase2_cinic10_adv_rand_0_ \
#                                     --device_id=0
#                                    
for seedset_dir in /mydata/model-extraction/data/cifar10_balanced_subset2500 /mydata/model-extraction/data/cifar10_balanced_subset2000 /mydata/model-extraction/data/cifar10_balanced_subset1000
do
CUDA_VISIBLE_DEVICES=1 python main.py --model_name="wrn28_2" \
--seedset_dir=$seedset_dir \
                                       --output_type="one_hot" \
                                       --t_rand \
                                       --epochs=10 \
                                       --adjust_epochs=120 \
                                       --phi=3 \
                                       --eps=0.01 \
                                       --encoder_arch_name=$encoder_name \
                                       --encoder_suffix=$encoder_suffix \
                                       --encoder_margin=3.2 \
                                       --thresh=$thresh \
                                       --log_suffix="less_seed" \
                                       --device_id=1
done                                       

#for blinder in get_uniform_noise get_gaussian_noise get_random_brightness get_random_rotate get_random_contrast get_random_translate get_random_scale get_random_crop
#for blinder in get_random_contrast get_random_translate get_random_brightness
#do
#
##blinder="get_random_translate"
#CUDA_VISIBLE_DEVICES=1 python main.py --model_name="wrn28_2" \
#                                      --encoder_arch_name=$encoder_name \
#                                      --encoder_suffix=$encoder_suffix \
#                                      --encoder_margin=3.2 \
#                                      --k=1 \
#                                      --thresh=$thresh \
#                                      --blinders_dir=$blinder \
#                                      --log_suffix={$blinder}_xaug \
#                                      --r="low" \
#                                      --device_id=1
#
#
#CUDA_VISIBLE_DEVICES=1 python main.py --model_name="wrn28_2" \
#                                      --encoder_arch_name=$encoder_name \
#                                      --encoder_suffix=$encoder_suffix \
#                                      --encoder_margin=3.2 \
#                                      --k=1 \
#                                      --thresh=$thresh \
#                                      --blinders_dir=$blinder \
#                                      --log_suffix={$blinder}_xaug \
#                                      --r="high" \
#                                      --device_id=1
#
#done
#CUDA_VISIBLE_DEVICES=1 python main.py --model_name=wrn28_2 \
#                                      --encoder_arch_name=resnet34 \
#                                      --k=1 \
#                                      --phi=1 \
#                                      --epochs=150 \
#                                      --random_adv \
#                                      --thresh=0.054156411439180374 \
#                                      --log_suffix=distilled_2.0 \
#                                      --device_id=1
#                                      
#CUDA_VISIBLE_DEVICES=1 python main.py --model_name=wrn28 \
#                                      --encoder_arch_name=resnet34 \
#                                      --k=5 \
#                                      --thresh=0.08149350184947253 \
#                                      --log_suffix=auto_cifar \
#                                      --blinders_dir=/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/autoencoder_blind/phase2_ \
#                                      --device_id=1                                     
#CUDA_VISIBLE_DEVICES=0 python main.py --model_name=wrn28 \
#                                      --k=10 \
#                                      --thresh=0.13513331776857376 \
#                                      --log_suffix=conf_wrn28_k10 \
#                                      --return_conf_max \
#                                      --device_id=0

#CUDA_VISIBLE_DEVICES=0 python main.py --model_name=wrn40 \
#                                      --k=5 \
#                                      --thresh=0.028577305126935244 \
#                                      --log_suffix=conf_wrn40 \
#                                      --return_conf_max                                   