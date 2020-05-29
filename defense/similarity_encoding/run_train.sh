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
                
python train.py --load_pretrained --ckp_dir=/mydata/model-extraction/model-extraction-defense/defense/similarity_encoding/ \
                --sim_epochs=30 \
                --ckpt_suffix=.feat_simnet_norm \
                --sim_norm \
                --adv_train \
                --margin=10 \
                -d 0