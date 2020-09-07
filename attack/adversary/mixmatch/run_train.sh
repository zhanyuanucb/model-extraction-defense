python train.py --gpu=1 \
                --lr=1e-4 \
                --dataset_name="CINIC10" \
                --seed_dir=/mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10/2020-08-31_08:03:54_baseline/substitute_set.pt \
                --resume=/mydata/model-extraction/model-extraction-defense/attack/adversary/models/cifar10/2020-09-02_21:08:56_standard/checkpoint.budget35000.pth.tar \
                --out="standard"