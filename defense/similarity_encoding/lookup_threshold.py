import pickle
import argparse
import os
import os.path as osp


def main():
    parser = argparse.ArgumentParser(description='Given K and margin, look up threshold.')
    parser.add_argument('--K', type=int)
    parser.add_argument('--encoder_arch', type=str, default="simnet")
    parser.add_argument('--margin', type=str)
    parser.add_argument('--dataset', type=str, default="CIFAR10")

    args = parser.parse_args()
    params = vars(args)

    margin = params["margin"]
    dataset_name = params["dataset"]
    encoder_arch = params["encoder_arch"]
    K = params["K"]
    with open(osp.join(f"{encoder_arch}/{dataset_name}-margin-{margin}", "k_n_thresh.pkl"), 'rb') as file:
        ks, thresholds = pickle.load(file)
    assert len(thresholds) > K, f"K({K}) should < len(thresholds){len(thresholds)}"

    msg = [str(margin), str(K), str(dict(zip(ks, thresholds))[K])]

    out_dir = "./margin_k_thresh.tsv"
    if not osp.exists(out_dir):
        with open(out_dir, 'w') as record:
            title = ["Margin", "K", "Threshold"]
            record.write('\t'.join(title) + '\n')

    with open(out_dir, 'a') as record:
        record.write(f"{encoder_arch} " + '\n')

    with open(out_dir, 'a') as record:
        record.write('\t'.join(msg) + '\n')

    print(msg)

if __name__ == '__main__':
    main()