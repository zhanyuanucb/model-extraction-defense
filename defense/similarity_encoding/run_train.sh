#CUDA_VISIBLE_DEVICES=1 python train.py --ckpt_suffix=.cifar10_feat --margin=10
#CUDA_VISIBLE_DEVICES=1 python train.py --ckpt_suffix=.cifar10_feat --load_pretrained --margin=100
#CUDA_VISIBLE_DEVICES=1 python train.py --ckpt_suffix=.cifar10_feat --load_pretrained --margin=1000

#python train.py --dataset_name=MNIST --ckpt_suffix=.mnist_feat --load_pretrained=0 --train_epochs=5 --sim_epochs=30 --margin=1
#python train.py --dataset_name=MNIST --ckpt_suffix=.mnist_feat --load_pretrained=1 --train_epochs=5 --sim_epochs=30 --margin=10
#python train.py --dataset_name=MNIST --ckpt_suffix=.mnist_feat --load_pretrained=1 --train_epochs=5 --sim_epochs=30 --margin=100
#python train.py --dataset_name=MNIST --ckpt_suffix=.mnist_feat --load_pretrained=1 --train_epochs=5 --sim_epochs=30 --margin=1000

#CUDA_VISIBLE_DEVICES=1 python train.py --load_pretrained --ckp_dir=/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28/ --model_name=wrn28 --sim_epochs=10 --margin=1
#CUDA_VISIBLE_DEVICES=1 python train.py --load_pretrained --ckp_dir=/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28/ --model_name=wrn28 --sim_epochs=30 --margin=10
#CUDA_VISIBLE_DEVICES=1 python train.py --load_pretrained --ckp_dir=/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28/ --model_name=wrn28 --sim_epochs=10 --margin=100
#CUDA_VISIBLE_DEVICES=1 python train.py --load_pretrained --ckp_dir=/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28/ --model_name=wrn28 --sim_epochs=10 --margin=1000

#python train.py --load_pretrained --ckp_dir=/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28/ \
#                --model_name=wrn28 \
#                --sim_norm \
#                --sim_epochs=30 \
#                --margin=10 \
#                -d 0


#CUDA_VISIBLE_DEVICES=1 python train.py --model_name=wrn28 \
#                --ckpt_suffix=.feat_wrn28_xnorm \
#                --sim_epochs=30 \
#                --margin=10 \
#                -d 1
                
#CUDA_VISIBLE_DEVICES=0 python train.py --load_pretrained \
#                --ckp_dir=/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding/ \
#                --sim_epochs=30 \
#                --ckpt_suffix=.feat_simnet_xnorm \
#                --adv_train \
#                --margin=10 \
#                -d 0
#                
#CUDA_VISIBLE_DEVICES=0 python ../pick_thresh.py --dataset_name=CIFAR10 --model_name=simnet --up_to_K
#
#for k in 1 2 3 4 5 6 7 8 9 10 20 30 40 50 100 150 200
#do
#   python ../lookup_threshold.py --K=$k --margin=3.2 --dataset=CIFAR10 --encoder_arch=simnet
#done               

model_name="resnet34"
CUDA_VISIBLE_DEVICES=1 python train.py \
                --load_pretrained \
                --ckp_dir=/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding/ \
                --sim_epochs=10 \
                --sim_norm \
                --model_name=$model_name \
                --ckpt_suffix=.feat_$model_name \
                --margin=10 \
                -d 1
                
CUDA_VISIBLE_DEVICES=0 python pick_thresh.py \
                       --dataset_name=CIFAR10 \
                       --model_name=$model_name --up_to_K \
                       --margins=3.2

for k in 1 2 3 4 5 6 7 8 9 10 20 30 40 50 100 150 200
do
   python lookup_threshold.py --K=$k --margin=3.2 \
    --dataset=CIFAR10 \
    --encoder_arch=$model_name
done               