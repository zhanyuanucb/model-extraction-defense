#model_name="wrn28_2"
#margin=10
#model_suffix="_xAE_ep30"
#CUDA_VISIBLE_DEVICES=1 python train.py \
#                --load_pretrained \
#                --ckp_dir=/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28_2/ \
#                --sim_epochs=30 \
#                --sim_norm \
#                --model_name=$model_name \
#                --model_suffix=$model_suffix$sim_epochs \
#                --margin=$margin \
#                -d 1
               
               
#model_name="simnet"
#margin=10
#model_suffix="_ep30"
#CUDA_VISIBLE_DEVICES=0 python train.py \
#                --load_pretrained \
#                --ckp_dir=/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding \
#                --ckpt_suffix=".feat_simnet" \
#                --sim_epochs=30 \
#                --sim_norm \
#                --model_name=$model_name \
#                --model_suffix=$model_suffix \
#                --margin=$margin \
#                -d 0              
                
model_name="simnet"
margin=10
model_suffix="_random_start"
#CUDA_VISIBLE_DEVICES=0 python train.py \
#                --sim_epochs=30 \
#                --sim_norm \
#                --model_name=$model_name \
#                --model_suffix=$model_suffix \
#                --margin=$margin \
#                -d 0                             
                
CUDA_VISIBLE_DEVICES=1 python pick_thresh.py \
                       --dataset_name=CIFAR10 \
                       --model_name=$model_name --up_to_K \
                       --model_suffix=$model_suffix \
                       --margins=3.2 \
                       --norm

for k in 1 2 3 4 5 6 7 8 9 10 20 30 40 50 100 150 200
do
   python lookup_threshold.py --K=$k --margin=3.2 \
    --dataset=CIFAR10 \
    --encoder_arch=$model_name \
    --encoder_suffix=$model_suffix \

done