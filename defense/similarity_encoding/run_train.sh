model_name="wrn28_2"
margin=10
model_suffix="_victim_xtrained"
#for sim_epochs in 5 10 30
#do
#CUDA_VISIBLE_DEVICES=0 python train.py \
#                --load_pretrained \
#                --ckp_dir=/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28_2/ \
#                --sim_epochs=$sim_epochs \
#                --sim_norm \
#                --model_name=$model_name \
#                --model_suffix=$model_suffix$sim_epochs \
#                --margin=$margin \
#                -d 0
               
CUDA_VISIBLE_DEVICES=0 python pick_thresh.py \
                       --dataset_name=CIFAR10 \
                       --model_name=$model_name --up_to_K \
                       --model_suffix=$model_suffix \
                       --margins=3.2 \
                       --activation="sigmoid" \
                       --norm

for k in 1 2 3 4 5 6  8 9 10 20 30 40 50 100 150 200
do
   python lookup_threshold.py --K=$k --margin=3.2 \
    --dataset=CIFAR10 \
    --encoder_arch=$model_name \
    --encoder_suffix="_victim_xtrained" \

done