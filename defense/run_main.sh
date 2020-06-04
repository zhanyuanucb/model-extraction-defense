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
                                     
#CUDA_VISIBLE_DEVICES=0 python main.py --model_name=wrn28 \
#                                      --log_suffix=conf_wrn28_k5 \
#                                      --return_conf_max \
#                                      --device_id=1

#CUDA_VISIBLE_DEVICES=0 python main.py --model_name=wrn28 \
#                                      --k=10 \
#                                      --thresh=0.06685449682176113 \
#                                      --log_suffix=conf_wrn28_k10 \
#                                      --return_conf_max \
#                                      --device_id=0

CUDA_VISIBLE_DEVICES=0 python main.py --model_name=wrn28 \
                                      --encoder_arch_name=simnet \
                                      --k=5 \
                                      --thresh=0.007808731311466545 \
                                      --log_suffix=simnet \
                                      --device_id=0

#CUDA_VISIBLE_DEVICES=0 python main.py --model_name=wrn40 \
#                                      --k=5 \
#                                      --thresh=0.028577305126935244 \
#                                      --log_suffix=conf_wrn40 \
#                                      --return_conf_max                                   