python train.py --train_epochs=10
python val_autoencoder_acc.py  --ckp_path=/mydata/model-extraction/model-extraction-defense/attack/victim/models/cifar10/wrn28/ --model_name=wrn28 --num_classes=10 --auto_path=/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/autoencoder_blind/phase2 --dataset_name=CIFAR10

#python train.py --train_epochs=10 --optimizer_name=sgd 
#python train.py --train_epochs=20 --resume=/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/autoencoder_blind/checkpoint.blind.pth.tar
#python train.py --train_epochs=100 --out_dir=/mydata/model-extraction/model-extraction-defense/attack/adversary/query_blinding/autoencoder