#python train.py MNIST pnet -o /mydata/model-extraction/model-extraction-defense/models/mnist_testing -b 128 -e 50 --lr 0.01 --log-interval 10
python train.py CIFAR10 resnet18 -o /mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wo_normalization -b 128 -e 50 --lr 0.01 --log-interval 10