import os
import shutil
import time
import pprint

import scipy
from scipy import interpolate
from scipy.stats import t
import torch
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, f1_score


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if indices.is_cuda:
        encoded_indicies = encoded_indicies.cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('setting gpu:', x)
    print('multi_gpu counts: ', torch.cuda.device_count())


def ensure_path(dir_path, scripts_to_save=None):
    if os.path.exists(dir_path):
        if input('{} exists, remove? ([y]/n)'.format(dir_path)) != 'n':
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
    else:
        os.mkdir(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for src_file in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(src_file))
            print('copy {} to {}'.format(src_file, dst_file))
            if os.path.isdir(src_file):
                shutil.copytree(src_file, dst_file)
            else:
                shutil.copyfile(src_file, dst_file)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def roc_area_score(args, scores, targets, descending):
    # sort 默认对最后一维的数据排序，返回(values, indices)
    y_score, p = scores.sort(descending=descending)
    y_score = y_score.detach().cpu()
    # closed 也按照排序完的dist
    y_true = targets[p]
    # print(y_true)
    y_true = y_true.detach().cpu()

    # metric 1: auroc
    area_score = roc_auc_score(y_true, y_score)

    # metric 2: aupr
    aupr = average_precision_score(y_true, y_score)  # **********

    # metric 3: f1score
    y_pred = np.where(y_score >= np.sort(y_score)[args.way*args.query], 1, 0)
    f_score = f1_score(y_true, y_pred, average="binary")  # **********

    # metric 4: fpr95
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))  # **********

    return area_score, aupr, f_score, fpr95


def count_acc(args, logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def mean_confidence_interval(data, confidence=0.95):
    a = 100.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    m = np.round(m, 3)
    h = np.round(h, 3)
    return m, h


def postprocess_args(args):
    args.device = 'cpu'
    if torch.cuda.device_count() > 0:
        args.device = 'cuda'
    args.n_task = args.tasks_per_episode

    save_path1 = '-'.join([args.dataset, args.backbone_class,
                           '{:02d}w{:02d}s{:02}q{:02}o'.format(args.way, args.shot, args.query,
                                                               args.open)])  # ‘-’ is the connection flag between each element
    save_path2 = '_'.join([str('_'.join(args.step_size.split(','))), str(args.gamma),  # '_' is the same
                           'lr{:.2g}mul{:.2g}'.format(args.lr, args.lr_mul),
                           str(args.lr_scheduler),
                           'bsz{:03d}'.format(args.way * (args.shot + args.query) + args.way_open * args.open),
                           str(time.strftime('%Y%m%d_%H%M%S'))
                           ])

    if args.init_weights is not None:
        save_path1 += '-Pre'


    if args.test_only or args.resume:
        args.save_path = args.checkpoints_path
    else:
        if not os.path.exists(os.path.join(args.save_dir, save_path1)):
            os.makedirs(os.path.join(args.save_dir, save_path1))
        if not os.path.exists(os.path.join(args.save_dir, save_path1, save_path2)):
            os.makedirs(os.path.join(args.save_dir, save_path1, save_path2))
        args.save_path = os.path.join(args.save_dir, save_path1, save_path2)
    return args


def get_command_line_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=80, help='80 for miniImageNet and TireredImageNet, 120 for Cifar-FS')
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--tasks_per_episode', type=int, default=1)
    parser.add_argument('--num_eval_episodes', type=int, default=600)
    parser.add_argument('--num_test_episodes', type=int, default=600)
    parser.add_argument('--backbone_class', type=str, default='Res12',
                        choices=['ConvNet', 'Res12', 'Res18', 'WRN'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'TieredImageNet', 'CIFAR-FS'])

    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--way_open', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--open', type=int, default=15)
    parser.add_argument('--eval_way', type=int, default=5)
    parser.add_argument('--eval_way_open', type=int, default=5)
    parser.add_argument('--eval_shot', type=int, default=1)
    parser.add_argument('--eval_query', type=int, default=15)
    parser.add_argument('--eval_open', type=int, default=15)

    # optimization parameters
    parser.add_argument('--orig_imsize', type=int,
                        default=-1)  # -1 for no cache, and -2 for no resize, only for MiniImageNet and CUB
    parser.add_argument('--lr', type=float, default=0.0002)  #
    parser.add_argument('--lr_mul', type=float, default=10)
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='40')
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', default='0')

    # usually untouched parameters
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)  # we find this weight decay value works the best
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=2)
    parser.add_argument('--init_weights', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')

    # choose the only test mode for model, no need to train and only return the evaluation result
    parser.add_argument('--checkpoints_path', type=str, default='')
    parser.add_argument('--max_checkpoint_type', type=str, default='max_auroc', choices=['max_auroc', 'max_acc'])
    parser.add_argument('--test_only', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_weights', type=str, default=None)

    parser.add_argument('--sigma', type=float, default=10000, help='10000, 1000, 100')
    parser.add_argument('--vector', type=int, default=48, help='24, 12, 96')

    # --------------------------------desc of version updating------------------------------------------------
    parser.add_argument('--version_desc', type=str,
                        default='')

    return parser

