import argparse
import modelzoo.zoo as zoo
from attack import datasets
from defense import utils as defense_utils 

def main():
    parser = argparse.ArgumentParser(description='Count model parameters')
    parser.add_argument('--dataset_name', metavar='TYPE', type=str, help='Name of validation dataset', default="CIFAR10")
    parser.add_argument('--model_name', metavar='TYPE', type=str, help='Model name', default="resnet18")
    parser.add_argument('--num_classes', metavar='TYPE', type=int, help='Number of classes', default=10)

    args = parser.parse_args()
    params = vars(args)

    model_name = params['model_name']
    num_classes = params['num_classes']
    dataset_name = params['dataset_name']
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    model = zoo.get_net(model_name, modelfamily, num_classes=num_classes)

    num_params = defense_utils.count_parameters(model)

    print(f"=> {model_name} ({modelfamily}): {num_params}")

if __name__ == '__main__':
    main()