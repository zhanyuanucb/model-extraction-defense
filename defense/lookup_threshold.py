import pickle
import argparse
import os
import os.path as osp


def main():
    parser = argparse.ArgumentParser(description='Given K and margin, look up threshold.')
    parser.add_argument('--K', type=int)
    parser.add_argument('--margin', type=str)
    parser.add_argument('--dataset', type=str)

    args = parser.parse_args()
    params = vars(args)

    margin = params["margin"]
    dataset_name = params["dataset"]
    K = params["K"]
    with open(osp.join(f"similarity_encoding/{dataset_name}-margin-{margin}", "k_n_thresh.pkl"), 'rb') as file:
        ks, thresholds = pickle.load(file)
    assert len(thresholds) > K, f"K({K}) should < len(thresholds){len(thresholds)}"

    msg = [str(margin), str(K), str(dict(zip(ks, thresholds))[K])]

    out_dir = "./margin_k_thresh.tsv"
    if not osp.exists(out_dir):
        with open(out_dir, 'w') as record:
            title = ["Margin", "K", "Threshold"]
            record.write('\t'.join(title) + '\n')
    with open(out_dir, 'a') as record:
        record.write('\t'.join(msg) + '\n')

    print(msg)

if __name__ == '__main__':
    main()