import collections
import os
import pickle

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import tqdm
from torch.utils.data import DataLoader
from model.dataloader.samplers import CategoriesSampler
from model.models.tnpnet import TNPNet



class MultiGPUDataloader:
    def __init__(self, dataloader, num_device):
        self.dataloader = dataloader
        self.num_device = num_device

    def __len__(self):
        return len(self.dataloader) // self.num_device

    def __iter__(self):
        data_iter = iter(self.dataloader)
        done = False

        while not done:
            try:
                output_batch = ([], [])
                for _ in range(self.num_device):
                    batch = next(data_iter)
                    for i, v in enumerate(batch):
                        output_batch[i].append(v[None])
                
                yield ( torch.cat(_, dim=0) for _ in output_batch )
            except StopIteration:
                done = True
        return


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def save_pickle(file, data):
    with open(file, 'wb') as f:
        return pickle.dump(data, f)


def get_dataloader(args):
    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'TieredImageNet':
        from model.dataloader.tiered_imagenet import tieredImageNet as Dataset
    elif args.dataset == 'CIFAR-FS':
        from model.dataloader.cifar import CifarFs as Dataset
    else:
        raise ValueError('Non-supported Dataset.')

    num_device = torch.cuda.device_count()
    num_episodes = args.episodes_per_epoch*num_device if args.multi_gpu else args.episodes_per_epoch
    num_workers=args.num_workers*num_device if args.multi_gpu else args.num_workers
    trainset = Dataset('train', args)
    args.num_class = trainset.num_class
    train_sampler = CategoriesSampler(trainset.label,
                                      num_episodes,  # n_batch
                                      args.n_task,
                                      args.way,  # n_cls_closed
                                      args.shot + args.query,  # n_per_close(support + query)
                                      args.way_open,  # n_cls_open
                                      args.open)  # n_per

    train_loader = DataLoader(dataset=trainset,
                                  num_workers=num_workers,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label,
                            args.num_eval_episodes,
                            1,
                            args.eval_way,
                            args.eval_shot + args.eval_query,  # n_per_close(support + query)
                            args.eval_way_open,  # n_cls_open
                            args.eval_open)  # n_per
    val_loader = DataLoader(dataset=valset,
                            batch_sampler=val_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)
    
    
    testset = Dataset('test', args)
    test_sampler = CategoriesSampler(testset.label,
                            args.num_test_episodes,  # args.num_eval_episodes,
                            1,
                            args.eval_way,
                            args.eval_shot + args.eval_query,  # n_per_close(support + query)
                            args.eval_way_open,  # n_cls_open
                            args.eval_open)  # n_per
    test_loader = DataLoader(dataset=testset,
                            batch_sampler=test_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)    

    return train_loader, val_loader, test_loader


def prepare_model(args):
    """
    return:
        model: the checkpoint model
    """
    model = TNPNet(args)  # should import the model_class object, or error: name 'model_class' is not defined
    device = args.device  # config the device before loading the checkpoint

    # load pre-trained model (no FC weights)
    model_dict = model.state_dict()
    if args.init_weights is not None and args.resume == 0:
        if torch.cuda.device_count() == 0:
            pretrained_dict = torch.load(args.init_weights, map_location=torch.device(device))['params']  # if use CPU, config map_location
        else:
            pretrained_dict = torch.load(args.init_weights)['params']
        if args.backbone_class == 'ConvNet':
            # +k ?  the '+' is the connector for characters
            pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # only load the params in model
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if args.multi_gpu:
        model.encoder = nn.DataParallel(model.encoder, dim=0)
    if torch.cuda.device_count() > 0:
        model = model.cuda()

    return model


def prepare_optimizer(model, args):
    top_para = [v for k, v in model.named_parameters() if 'encoder' not in k]  # load optimizer except encoder
    # as in the literature, we use ADAM for ConvNet and SGD for other backbones

    if args.backbone_class == 'ConvNet':
        optimizer = optim.Adam(  # different params can have different initial lr
            [{'params': model.encoder.parameters()},
             {'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            # weight_decay=args.weight_decay, do not use weight_decay here
        )                
    else:
        optimizer = optim.SGD(  # different params can have different initial lr
            [{'params': model.encoder.parameters()},
             {'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            momentum=args.mom,
            nesterov=True,
            weight_decay=args.weight_decay
        )

    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(  # scheduler, Decays the learning rate of each parameter
                            optimizer,  # group by gamma every step_size epochs.
                            step_size=int(args.step_size),
                            gamma=args.gamma
                        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones=[int(_) for _ in args.step_size.split(',')],
                            gamma=args.gamma,
                        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            args.max_epoch,
                            eta_min=0   # a tuning parameter
                        )
    else:
        raise ValueError('No Such Scheduler')

    return optimizer, lr_scheduler


