#python train.py --train_epochs=10 --ckp_path=/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28
#python val_autoencoder_acc.py  --ckp_path=/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28/ --model_name=wrn28 --num_classes=10 --auto_path=/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/autoencoder_blind/phase2 --dataset_name=CIFAR10
#python inspect_autoencoder.py --ckp_dir="/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/autoencoder_blind/phase2_" --dataset_name=CIFAR10

#python train.py --train_epochs=10
#
#python val_autoencoder_acc.py  --ckp_path=/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/f --model_name=wrn28 \
#                               --num_classes=10 \
#                               --auto_path=/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/autoencoder_blind/phase2 \
#                               --dataset_name=CIFAR10
#
#python inspect_autoencoder.py --ckp_dir=/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/autoencoder_blind/phase2 \
#                              --dataset_name=CIFAR10


for i in 1 2 #3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
python train.py --train_epochs=10 --out_dir=/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/autoencoder_blind \
                                  --attempt=$i

python val_autoencoder_acc.py  --ckp_path=/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/f --model_name=wrn28 \
                               --num_classes=10 \
                               --auto_path=/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/autoencoder_blind/phase2_$i \
                               --dataset_name=CIFAR10

python inspect_autoencoder.py --ckp_dir=/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/autoencoder_blind/phase2_$i \
                              --dataset_name=CIFAR10
done