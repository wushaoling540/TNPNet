import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel

class TNPNet(FewShotModel):
    def __init__(self, args):
        super().__init__(args)

        self.init_sigma = torch.FloatTensor(1).fill_(args.sigma)
        # scale and bias for similarity_metric
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(1), requires_grad=True)


    def _forward(self, instance_embs, support_idx, query_idx, open_idx):
        instance_flatten, _ = instance_embs
        instance_embs = instance_flatten

        emb_dim = instance_embs.size(1)
        support = instance_embs[support_idx.contiguous().view(-1)].contiguous().view(*(support_idx.shape + (-1,)))
        query = instance_embs[query_idx.contiguous().view(-1)].contiguous().view(*(query_idx.shape + (-1,)))
        open = instance_embs[open_idx.contiguous().view(-1)].contiguous().view(*(open_idx.shape + (-1,)))

        n_task = support_idx.shape[0]
        label_s, label_q, label_o = self.labels[0].view(n_task, -1), \
                                    self.labels[1].view(n_task, -1), self.labels[2].view(n_task, -1)

        # get mean of the support
        kproto = support.mean(dim=1)  # Nt x Ns x Nw x d -> Nt x Nw x d
        n_kproto = kproto.shape[1]

        support = support.view(n_task, -1, emb_dim).unsqueeze(-2)
        query = query.view(n_task, -1, emb_dim).unsqueeze(-2)  # (Nt, Nq*Nw, 1, d)
        open = open.view(n_task, -1, emb_dim).unsqueeze(-2)  # (Nt, No*Nw, 1, d)
        query_open = torch.cat((query, open), dim=1)  # (Nt, Nq*Nw + No*Nw, 1, d)

        # transductive learning:
        uproto = self.transductive_learning(support, label_s, query_open, kproto)

        # classifier:
        kproto_u = uproto[:, :-1, :]
        kkproto = F.normalize(kproto_u, dim=-1)  # normalize for cosine distance
        uuproto = F.normalize(uproto, dim=-1)
        query_open = query_open.squeeze(-2)

        klogits = self.scale_cls * torch.baddbmm(self.bias, query_open, kkproto.permute([0, 2, 1]))  # bmm
        ulogits = self.scale_cls * torch.baddbmm(self.bias, query_open, uuproto.permute([0, 2, 1]))  # bmm

        klogits = klogits.view(-1, *klogits.shape[2:])
        ulogits = ulogits.view(-1, *ulogits.shape[2:])

        return klogits, ulogits

