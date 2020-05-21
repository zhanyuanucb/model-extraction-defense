#python train.py MNIST pnet -o /mydata/model-extraction/model-extraction-defense/models/mnist_testing -b 128 -e 50 --lr 0.01 --log-interval 10
#python train.py CIFAR10 resnet18 -o /mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/resnet18 -b 128 -e 50 --lr 0.01 --log-interval 10
#python train.py MNIST pnet -o /mydata/model-extraction/model-extraction-defense/attack/victim/models/mnist/pnet -b 128 -e 50 --lr 0.01 --log-interval 10
#python train.py CIFAR10 wrn28 -o /mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28 -b 256 -e 120 --lr 0.01 --log-interval 10
#python train.py CIFAR10 wrn28 -o /mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/f -b 256 -e 50 --train_subset=5000 --lr 0.01 --log-interval 10
python train.py ImageNet32 wrn40 -o /mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/f_imagenet32 -b 128 -e 120 --lr 0.01 --num_classes 1000 --log-interval 1000