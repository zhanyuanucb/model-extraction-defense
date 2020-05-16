#python getseed.py /mydata/model-extraction/model-extraction-defense/attack/victim/models/mnist --out_dir /mydata/model-extraction/model-extraction-defense/attack/adversary/models/mnist --seedset_name=MNIST --seedset_dir=/mydata/model-extraction/data/mnist_balanced_subset100 --batch_size=128
#python getseed.py /mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/resnet18 --out_dir /mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10 --seedset_name=CIFAR10 --seedset_dir=/mydata/model-extraction/data/cifar10_balanced_subset1000 --batch_size=128
#python getseed.py /mydata/model-extraction/model-extraction-defense/attack/victim/models/mnist/pnet --out_dir /mydata/model-extraction/model-extraction-defense/attack/adversary/models/mnist --seedset_name=CIFAR10 --seedset_dir=/mydata/model-extraction/data/mnist_balanced_subset100 --batch_size=128
#python getseed.py /mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/resnet18 --out_dir /mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10 --seedset_name=CIFAR10 --seedset_dir=/mydata/model-extraction/data/cifar10_balanced_subset1000 --batch_size=128
#python getseed.py /mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28 --out_dir /mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10 --seedset_name=CIFAR10 --seedset_dir=/mydata/model-extraction/data/cifar10_balanced_subset6400 --batch_size=128

python getseed.py /mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28 \
                  --out_dir /mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10 \
                  --seedset_name=CIFAR10 \
                  --seedset_dir=/mydata/model-extraction/data/cifar10_balanced_subset5000 \
                  --batch_size=128