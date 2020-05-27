#python lookup_threshold.py --K=50 --margin=1.0 --dataset=CIFAR10
#python lookup_threshold.py --K=200 --margin=1.0 --dataset=CIFAR10
#python lookup_threshold.py --K=50 --margin=3.2 --dataset=CIFAR10
#python lookup_threshold.py --K=200 --margin=3.2 --dataset=CIFAR10
#python lookup_threshold.py --K=50 --margin=10.0 --dataset=CIFAR10
#python lookup_threshold.py --K=200 --margin=10.0 --dataset=CIFAR10
#python lookup_threshold.py --K=50 --margin=31.6 --dataset=CIFAR10
#python lookup_threshold.py --K=200 --margin=31.6 --dataset=CIFAR10

#python lookup_threshold.py --K=10 --margin=3.2 --dataset=CIFAR10
#python lookup_threshold.py --K=50 --margin=3.2 --dataset=CIFAR10
#python lookup_threshold.py --K=200 --margin=3.2 --dataset=CIFAR10

#for k in 1 2 3 4 5 6 7 8 9 10 20 30 40 50 100 150 200
#do
#   python lookup_threshold.py --K=$k --margin=3.2 --dataset=CIFAR10 --encoder_arch=wrn28
#done

for k in 1 2 3 4 5 6 7 8 9 10 20 30 40 50 100 150 200
do
   python lookup_threshold.py --K=$k --margin=3.2 --dataset=CIFAR10 --encoder_arch=simnet
done