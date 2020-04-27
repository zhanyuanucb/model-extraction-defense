CUDA_VISIBLE_DEVICES=1 python train.py --ckpt_suffix=.cifar10_feat --load_pretrained --margin=10
#CUDA_VISIBLE_DEVICES=1 python train.py --ckpt_suffix=.cifar10_feat --load_pretrained --margin=100
#CUDA_VISIBLE_DEVICES=1 python train.py --ckpt_suffix=.cifar10_feat --load_pretrained --margin=1000

#python train.py --dataset_name=MNIST --ckpt_suffix=.mnist_feat --load_pretrained=0 --train_epochs=5 --sim_epochs=30 --margin=1
#python train.py --dataset_name=MNIST --ckpt_suffix=.mnist_feat --load_pretrained=1 --train_epochs=5 --sim_epochs=30 --margin=10
#python train.py --dataset_name=MNIST --ckpt_suffix=.mnist_feat --load_pretrained=1 --train_epochs=5 --sim_epochs=30 --margin=100
#python train.py --dataset_name=MNIST --ckpt_suffix=.mnist_feat --load_pretrained=1 --train_epochs=5 --sim_epochs=30 --margin=1000

#CUDA_VISIBLE_DEVICES=1 python train.py --load_pretrained --ckp_dir=/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28/ --model_name=wrn28 --sim_epochs=10 --margin=1
#CUDA_VISIBLE_DEVICES=1 python train.py --load_pretrained --ckp_dir=/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28/ --model_name=wrn28 --sim_epochs=10 --margin=100
#CUDA_VISIBLE_DEVICES=1 python train.py --load_pretrained --ckp_dir=/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28/ --model_name=wrn28 --sim_epochs=10 --margin=1000