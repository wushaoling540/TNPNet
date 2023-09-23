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

    def compute_logs_sigma(self, cluster_centers, data, sigma):
        cluster_centers = cluster_centers.unsqueeze(1)
        sigma = sigma.unsqueeze(1)
        neg_dist = - torch.sum((data - cluster_centers) ** 2, -1)

        logits = neg_dist / 2.0 / (sigma**2)
        pi = torch.log(2 * torch.tensor([np.pi]))
        if torch.cuda.device_count() > 0:
            pi = pi.cuda()
        norm_constant = 0.5 * pi + torch.log(sigma)
        logits -= norm_constant
        return logits

    def compute_positive_logs_sigma(self, cluster_centers, data, sigma):
        cluster_centers = cluster_centers.unsqueeze(1)
        sigma = sigma.unsqueeze(1)
        neg_dist = torch.sum((data - cluster_centers) ** 2, -1)
        logits = neg_dist / 2.0 / (sigma**2)
        pi = torch.log(2 * torch.tensor([np.pi]))
        if torch.cuda.device_count() > 0:
            pi = pi.cuda()
        norm_constant = 0.5 * pi + torch.log(sigma)
        logits += norm_constant
        return logits

    def assign_cluster_sigma(self, cluster_centers, data, sigma):
        logits = self.compute_logs_sigma(cluster_centers, data, sigma)
        prob = F.softmax(logits, dim=-1)
        return prob

    def update_cluster(self, data, prob):
        prob_sum = prob.sum(1, keepdim=True)
        prob_sum += (prob_sum == 0.0).float() # 保证分母非0
        prob2 = prob / prob_sum
        cluster_centers = (data * prob2.unsqueeze(-1)).sum(1) # Nt*N*1*(Ndim) x Nt*N*1*(Nw+1)
        return cluster_centers

    def context_score_learning(self, dis, n_proto):
        dis_except_min = torch.zeros(dis.shape)
        if torch.cuda.device_count() > 0:
            dis_except_min = dis_except_min.cuda()
        # get the second min value except the current proto
        for i in range(n_proto):
            if i == 0:
                dis_except_min[:, :, i], _ = dis[:, :, i + 1:].min(dim=-1)
            else:
                dis_except_min[:, :, i], _ = torch.cat((dis[:, :, :i - 1], dis[:, :, i + 1:]), dim=-1).min(dim=-1)
        delta = 1e-8
        dis_ctx = dis / (dis_except_min + delta)
        scale_pro = torch.ones_like(dis_ctx)
        scale_pro[:, :, -1] = self.args.vector * scale_pro[:, :, -1]
        similarity_score = torch.exp(-dis_ctx * scale_pro)
        return similarity_score

    def augment_proto(self, protos, support, label_s, query_open, pseudo_label):
        n_task, n_proto, emb_dim = protos.shape
        for i in range(n_proto):
            current_supports = support[torch.where(label_s == i)].view(n_task, -1, emb_dim)
            selected_samples = query_open[torch.where(pseudo_label == i)].view(n_task, -1, emb_dim)
            if selected_samples.shape[1] == 0 and current_supports.shape[1] == 0:
                selected_samples = torch.zeros_like(protos[:, 0:1, :])
            protos[:, i] = torch.cat((current_supports, selected_samples), dim=1).mean(1)
        return protos

    def context_learning(self, query_open, protos, support, label_s):
        n_task, n_proto, emb_dim = protos.shape
        dis = torch.sum((query_open - protos) ** 2, -1)
        similarity_score = self.context_score_learning(dis, n_proto)
        pseudo_score, pseudo_label = similarity_score.max(dim=-1)
        # augment proto
        protos = self.augment_proto(protos, support, label_s, query_open, pseudo_label)
        return protos

    def transductive_learning(self, support, label_s, query_open, kproto):
        n_task, n_kproto, emb_dim = kproto.shape
        n_uproto = n_kproto + 1
        # prepare label negative_proto can be case1:0, case2: means, case3: random
        init_uproto = torch.zeros(n_task, 1, emb_dim)
        if torch.cuda.device_count() > 0:
            init_uproto = init_uproto.cuda()

        sqo_feats = torch.cat((support, query_open), dim=1)
        uproto = torch.cat((kproto, init_uproto), dim=1)

        labeled_ = F.one_hot(label_s).float()
        known_sigma = torch.ones((n_task, n_kproto)).float()
        distractor_sigma = torch.exp(torch.log(self.init_sigma))  # self.init_sigma
        distractor_sigma = distractor_sigma.repeat(n_task, 1)
        if torch.cuda.device_count() > 0:
            labeled_ = labeled_.cuda()
            known_sigma = known_sigma.cuda()
            distractor_sigma = distractor_sigma.cuda()

        # init sigma
        sigma = torch.cat((known_sigma, distractor_sigma), dim=-1)
        # prepare for support one-hot label
        open_label = torch.zeros_like(labeled_[:, :, 0:1])
        prob_support = torch.cat((labeled_, open_label), dim=-1)
        # local similarity context
        dis = self.compute_positive_logs_sigma(uproto, query_open, sigma)
        # global similarity context
        similarity_score = self.context_score_learning(dis, n_uproto)
        # enhance uproto
        prob_unlabel = F.softmax(similarity_score, dim=-1)
        prob_all = torch.cat((prob_support, prob_unlabel), dim=1)
        uproto = self.update_cluster(sqo_feats, prob_all)

        return uproto

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

