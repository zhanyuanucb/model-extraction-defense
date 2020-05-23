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

# OOD attack
CUDA_VISIBLE_DEVICES=0 python main.py --model_name=wrn28 \
                                      --phi=1 \
                                      --epochs=50 \
                                      --log_suffix=cinic_maxquery\
                                      --random_adv