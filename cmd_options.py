# -*- coding: utf-8 -*-

import json
import os.path as osp
import argparse
import logging
import os
import time


def create_log(args):
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename=os.path.join(args.dir, 'options.txt'), level=logging.DEBUG)
    logging.info(args)


def create_result_dir(dir):
    if not os.path.exists('results'):
        os.mkdir('results')
    if dir:
        result_dir = os.path.join('results', dir)
    else:
        result_dir = os.path.join(
            'results', time.strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    checkpoints = osp.join(result_dir, "checkpoints")
    if not os.path.exists(checkpoints):
        os.makedirs(checkpoints)

    return result_dir


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpus', type=int, default=1,
                        help='how many gpus will be used for training. 0 for cpu')
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    # optimize related
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Default is 64.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate. Default is 0.1')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Default is 1e-4.')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--n_epochs', type=int, default=300,
                        help="how many epochs will the model be trained")

    # network related
    parser.add_argument('--arch', type=str, default='resnet')
    parser.add_argument('--depth', type=int, default=40,
                        help='Depth of DenseNet, which is defined as L in '
                        'the paper. Default is 40.')
    parser.add_argument('--growth_rate', type=int, default=12,
                        help='Growth rate which is defined as k in the paper.'
                        ' Default is 12.')
    parser.add_argument('--bottleneck', type=str2bool, default=False,
                        help='''Whether to use bottleneck structure''')
    parser.add_argument('--drop_ratio', type=float, default=0.0,
                        help='Dropout ratio. The paper recommends 0 with data'
                        ' augmentation and 0.2 without data augmentation. '
                        'Default is 0')
    parser.add_argument('--fetch', type=str, default='linear', choices=['linear', 'exp'],
                        help='''The fetch style of layer aggregation
                                    `linear` for DenseNet , `exp` for SparseNet ''')
    parser.add_argument('--name', type=str, default=None)
    # parser.add_argument('--block', type=int, default=3,
    #                     help='Number of dense block. Default is 3.')

    parser.add_argument('--dir', type=str, default='work',
                        help='Directory name to save logs.')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'SVHN'],
                        help='Dataset name. Default \'cifar10\'.')
    parser.add_argument('--augment', type=str2bool, default=True,
                        help='Whether to use data augmentation. '
                             'Default is \'t\'')
    args = parser.parse_args()

    if args.gpus >= 0:
        args.gpus = list(range(args.gpus))
    else:
        args.gpus = None

    if args.name is None:
        args.name = "ResNet152"

    args.save = osp.join(args.dir, args.name)




    return args

if __name__ == "__main__":
    args = get_arguments()
    print(json.dumps(vars(args), indent=4))
