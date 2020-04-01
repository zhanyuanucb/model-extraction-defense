#python train.py --ckpt_suffix=feat_callback76 --callback=76.0 --load_pretrained=0 --margin=10
#python train.py --ckpt_suffix=feat_callback76 --callback=76.0 --load_pretrained=1 --margin=100
#python train.py --ckpt_suffix=feat_callback76 --callback=76.0 --load_pretrained=1 --margin=1000

#python train.py --ckpt_suffix=feat_callback --load_pretrained=0 --margin=10
#python train.py --ckpt_suffix=feat_callback --load_pretrained=1 --margin=100
#python train.py --ckpt_suffix=feat_callback --load_pretrained=1 --margin=1000

python train.py --ckpt_suffix=feat --load_pretrained=1 --margin=10
python train.py --ckpt_suffix=feat --load_pretrained=1 --margin=100
python train.py --ckpt_suffix=feat --load_pretrained=1 --margin=1000