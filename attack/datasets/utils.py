# Reference: https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949/13

import argparse
from PIL import ImageStat

class Stats(ImageStat.Stat):
    def __add__(self, other):
        return Stats(list(map(add, self.h, other.h)))

def cal_stats(dataset):
    stats = None
    for image, label in dataset:
        if stats is None:
            stats = Stats(image)
        else:
            stats += Stats(image)
    print(f"=> mean: {stats.mean}, std: {stats.stddev}")
    return stats

def main():
    parser = argparse.ArgumentParser(description='Simulate model extraction')
    parser.add_argument("--model_name", metavar="TYPE", type=str, default="resnet18")
    parser.add_argument("--device_id", metavar="TYPE", type=int, default=1)
