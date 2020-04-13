#python train.py --ckpt_suffix=feat_callback76 --callback=76.0 --load_pretrained=0 --margin=10
#python train.py --ckpt_suffix=feat_callback76 --callback=76.0 --load_pretrained=1 --margin=100
#python train.py --ckpt_suffix=feat_callback76 --callback=76.0 --load_pretrained=1 --margin=1000

#python train.py --ckpt_suffix=feat_callback --load_pretrained=0 --margin=10
#python train.py --ckpt_suffix=feat_callback --load_pretrained=1 --margin=100
#python train.py --ckpt_suffix=feat_callback --load_pretrained=1 --margin=1000

#python train.py --ckpt_suffix=feat --load_pretrained=0 --margin=10
#python train.py --ckpt_suffix=feat --load_pretrained=1 --margin=100
#python train.py --ckpt_suffix=feat --load_pretrained=1 --margin=1000

python train.py --dataset_name=MNIST --ckpt_suffix=mnist_feat --load_pretrained=1 --train_epochs=5 --sim_epochs=50 --margin=1
python train.py --dataset_name=MNIST --ckpt_suffix=mnist_feat --load_pretrained=1 --train_epochs=5 --sim_epochs=50 --margin=10
python train.py --dataset_name=MNIST --ckpt_suffix=mnist_feat --load_pretrained=1 --train_epochs=5 --sim_epochs=50 --margin=100
python train.py --dataset_name=MNIST --ckpt_suffix=mnist_feat --load_pretrained=1 --train_epochs=5 --sim_epochs=50 --margin=1000