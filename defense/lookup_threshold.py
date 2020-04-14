import pickle
import argparse
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
    
    print(f"K = {K} -> Threshold = {dict(zip(ks, thresholds))[K]}")

if __name__ == '__main__':
    main()