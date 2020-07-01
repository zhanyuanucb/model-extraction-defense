#CUDA_VISIBLE_DEVICES=1 python pick_thresh.py --dataset_name=CIFAR10 --model_name=wrn28 --up_to_K

#for k in 1 2 3 4 5 6 7 8 9 10 20 30 40 50 100 150 200
#do
#   python lookup_threshold.py --K=$k --margin=3.2 --dataset=CIFAR10 --encoder_arch=wrn28
#done

CUDA_VISIBLE_DEVICES=0 python pick_thresh.py --dataset_name=CIFAR10 --model_name=simnet --up_to_K

for k in 1 2 3 4 5 6 7 8 9 10 20 30 40 50 100 150 200
do
   python lookup_threshold.py --K=$k --margin=1.0 --dataset=CIFAR10 --encoder_arch=simnet
done