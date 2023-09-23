import torch
import torch.nn as nn
import numpy as np

class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
            self.hdim = 64
        elif args.backbone_class == 'Res12':
            self.hdim = 640
            from model.networks.res12 import ResNet
            if args.dataset in ['CIFAR-FS', 'FC100']:
                dropblock_size = 2
            else:
                dropblock_size = 5
            self.encoder = ResNet(dropblock_size=dropblock_size)
        elif args.backbone_class == 'Res18':
            self.hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'WRN':
            self.hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10, 0.5)  # we set the dropout=0.5 directly here, it may achieve better results by tunning the dropout
        else:
            raise ValueError('')

    def split_instances(self, data):
        args = self.args
        if self.training:
            return  (torch.Tensor(np.arange(args.n_task*args.way*   args.shot))
                     .long().view(args.n_task, args.shot, args.way),
                     torch.Tensor(np.arange(args.n_task*args.way*   args.shot, args.n_task*args.way * (args.shot + args.query)))
                     .long().view(args.n_task, args.query, args.way),
                     torch.Tensor(np.arange(args.n_task*args.way * (args.shot + args.query),
                                            args.n_task*(args.way * (args.shot + args.query) + args.way_open * args.open)))
                     .long().view(args.n_task, args.open, args.way_open))
        else: # the n_task is fixed to 1 when evaluating or testing, not consider.
            return  (torch.Tensor(np.arange(args.eval_way*args.eval_shot))
                     .long().view(1, args.eval_shot, args.eval_way),
                     torch.Tensor(np.arange(args.eval_way*args.eval_shot, args.eval_way * (args.eval_shot + args.eval_query)))
                     .long().view(1, args.eval_query, args.eval_way),
                     torch.Tensor(np.arange(args.eval_way * (args.eval_shot + args.eval_query),
                                            args.eval_way * (args.eval_shot + args.eval_query) + args.eval_way_open * args.eval_open))
                     .long().view(1, args.eval_open, args.eval_way_open))

    def forward(self, data):
        x, self.labels = data[0], data[1]
        # feature extraction
        x = x.squeeze(0)  # guarantee the four size (B, C, H, W), some may have only 3
        instance_embs = self.encoder(x)  # (B, d)
        # split support query set for few-shot data
        # (n_task, shot, way) permutation inverses
        support_idx, query_idx, open_idx = self.split_instances(x)
        klogits, ulogits = self._forward(instance_embs, support_idx, query_idx, open_idx)
        return klogits, ulogits

    def _forward(self, x, support_idx, query_idx, open_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')