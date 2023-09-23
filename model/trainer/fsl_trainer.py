import time
import math
import os.path as osp
import time

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import NTXentLoss
from tqdm import tqdm

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.utils import (
    Averager, count_acc, roc_area_score, compute_confidence_interval,
)

class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model = prepare_model(args) # train_weight
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)
        if args.resume:
            resume_path = osp.join(args.save_path, args.resume_weights)
            if torch.cuda.device_count() == 0:
                checkpoints = torch.load(resume_path,
                                     map_location=torch.device(args.device))  # if use cpu, config map_location
            else:
                checkpoints = torch.load(resume_path)
            self.model.load_state_dict(checkpoints['model_state_dict'])
            self.optimizer.load_state_dict(checkpoints['optim_state_dict'])
            self.trlog['max_acc_results'] = checkpoints['max_acc_results']
            self.trlog['max_auroc_results'] = checkpoints['max_auroc_results']
            self.trlog['max_acc'] = self.trlog['max_acc_results'][1]
            self.trlog['max_auroc'] = self.trlog['max_auroc_results'][3]
            self.train_epoch = checkpoints['epoch']
            num_device = torch.cuda.device_count()
            num_episodes = args.episodes_per_epoch * num_device if args.multi_gpu else args.episodes_per_epoch
            self.train_step = self.train_epoch * num_episodes

    def prepare_label(self):
        """
        change the label to continuous small numbers
        """
        args = self.args
        # prepare one-hot label
        if self.model.training:
            label_s = torch.arange(args.way).repeat(args.n_task * args.shot)
            label_q = torch.arange(args.way).repeat(args.n_task * args.query)
            label_o = args.way * torch.ones(args.n_task * args.way_open * args.open)
            label_2cls = torch.cat((torch.zeros(args.way * args.query), torch.ones(args.way_open * args.open)), dim=0).repeat(args.n_task)
        else: # the n_task is fixed to 1 when evaluating or testing, not consider.
            label_s = torch.arange(args.eval_way).repeat(args.eval_shot)
            label_q = torch.arange(args.eval_way).repeat(args.eval_query)
            label_o = args.eval_way * torch.ones(args.n_task * args.eval_way_open * args.eval_open)
            label_2cls = torch.cat(
                (torch.zeros(args.eval_way * args.eval_query), torch.ones(args.eval_way_open * args.eval_open)), dim=0)

        if torch.cuda.device_count() > 0:
            label_s = label_s.type(torch.LongTensor).cuda()
            label_q = label_q.type(torch.LongTensor).cuda()
            label_o = label_o.type(torch.LongTensor).cuda()
            label_2cls = label_2cls.type(torch.LongTensor).cuda()
        return label_s, label_q, label_o, label_2cls

    def loss(self, klogits, ulogits, labels):
        args = self.args
        label_s, label_q, label_o = labels
        logits_closed, logits_open = klogits[:self.num_query], klogits[self.num_query:]
        ulogits_closed, ulogits_open = ulogits[:self.num_query], ulogits[self.num_query:]
        cls_labels = torch.cat([label_q, label_o], dim=0)

        loss_aux, loss_neg = torch.Tensor([0.0]), torch.Tensor([0.0])
        if torch.cuda.device_count() > 0:
            loss_aux, loss_neg = loss_aux.cuda(), loss_neg.cuda()

        # main loss with  weight:
        weight = torch.ones(self.args.n_task, ulogits.shape[1])
        for i in range(ulogits.shape[1]):
            proportion = torch.where(cls_labels.view(self.args.n_task, -1) == i)[0].shape[0]
            if proportion == 0:
                weight[:, i] = torch.ones(1)
            else:
                weight[:, i] = torch.ones(1) / proportion
        weight = F.softmax(weight, dim=-1)
        weight = weight.view(-1)
        if torch.cuda.device_count() > 0:
            weight = weight.cuda()
        loss = F.cross_entropy(ulogits, cls_labels, weight)

        # auxiliary loss:
        dummpylogits = ulogits_closed.clone()
        for i in range(len(ulogits_closed)):
            nowlabel = label_q[i]
            dummpylogits[i][nowlabel] = -1e9
        dummytargets = args.way * torch.ones_like(label_q)
        loss_aux = F.cross_entropy(dummpylogits, dummytargets)

        # negative loss:
        loss_neg = F.softmax(logits_open, dim=-1) * F.log_softmax(logits_open, dim=-1)  # (N, Nw)
        loss_neg = F.cross_entropy(logits_closed, label_q)+ loss_neg.sum(-1).mean()

        # eta and mu
        loss_scale_aux_lut = [[20000], [0.5]]
        loss_scale_neg_lut = [[2000, 5000, 10000, 20000], [0.5, 0.3, 0.1, 0]]
        for i in range(len(loss_scale_aux_lut[0])):
            if self.train_step < loss_scale_aux_lut[0][i]:
                loss_scale_aux = loss_scale_aux_lut[1][i]
                break
        for i in range(len(loss_scale_neg_lut[0])):
            if self.train_step < loss_scale_neg_lut[0][i]:
                loss_scale_neg = loss_scale_neg_lut[1][i]
                break

        loss_total = loss + loss_scale_aux * loss_aux + loss_scale_neg * loss_neg

        return loss_total, loss, loss_aux, loss_neg

    def train(self):
        args = self.args
        # prepare truth label
        self.num_query = args.n_task * args.way * args.query
        self.num_open  = args.n_task * args.way * args.open
        label_s, label_q, label_o, label_2cls = self.prepare_label()
        labels = label_s, label_q, label_o
        # start FSL training
        for epoch in range(self.train_epoch + 1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train_epoch = self.train_epoch
            self.model.train()

            tl1, tl2, tl3, tl4 = Averager(), Averager(), Averager(), Averager()
            ta = Averager()
            troc, taupr, tf1_score, tfpr95 = Averager(), Averager(), Averager(), Averager()

            start_tm = time.time()
            for batch in self.train_loader:
                self.train_step += 1
                data, _ = batch
                if torch.cuda.device_count() > 0:
                    data = data.cuda()

                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                # forward...get saved centers
                klogits, ulogits = self.model((data, labels))
                # get loss
                loss_total, loss_main, loss_aux, loss_neg = self.loss(klogits, ulogits, labels)

                tl2.add(loss_main.item())
                tl3.add(loss_aux.item())
                tl4.add(loss_neg.item())

                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)

                # close-set accuracy
                acc = count_acc(args, ulogits[:self.num_query], label_q)
                # 有softmax
                logits_predict = F.softmax(ulogits, dim=1)[:, -1]
                # open-set auroc
                auroc, aupr, f_score, fpr95 = roc_area_score(args, logits_predict, label_2cls, True)

                tl1.add(loss_total.item())
                ta.add(acc)
                troc.add(auroc)
                taupr.add(aupr)
                tf1_score.add(f_score)
                tfpr95.add(fpr95)

                # backward....
                self.optimizer.zero_grad()
                loss_total.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)

                # refresh start_tm
                start_tm = time.time()

            self.lr_scheduler.step()
            self.try_logging(tl1, tl2, tl3, tl4, ta, troc)
            self.try_evaluate(epoch)

            self.logger.writer_log('ETA:{}/{}'.format(  # Estimated time of arrival
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )

        self.save_model('%s-epoch-last' % epoch)

    def evaluate(self, data_loader, mode='val'):
        args = self.args
        # evaluation mode
        if mode == 'test':
            model_dict = self.model.state_dict()
            max_checkpoint_type = args.max_checkpoint_type
            if torch.cuda.device_count() == 0:
                checkpoints = torch.load(osp.join(self.args.save_path, '%s.pth' % max_checkpoint_type),
                                         map_location=args.device)
            else:
                checkpoints = torch.load(osp.join(self.args.save_path, '%s.pth' % max_checkpoint_type))
            self.trlog['max_acc_results'] = checkpoints['max_acc_results']
            self.trlog['max_auroc_results'] = checkpoints['max_auroc_results']
            model_save_dict = checkpoints['model_state_dict']
            saved_dict = {k: v for k, v in model_save_dict.items() if k in model_dict}
            model_dict.update(saved_dict)
            self.model.load_state_dict(model_dict)
            record = np.zeros((args.num_test_episodes, 6))
        else:
            record = np.zeros((args.num_eval_episodes, 6))
        self.model.eval()

        # start evaluate...
        self.num_query = args.eval_way * args.eval_query
        label_s, label_q, label_o, label_2cls = self.prepare_label()
        labels = label_s, label_q, label_o
        with torch.no_grad():
            i = 0
            for batch in tqdm(data_loader, ncols=64, position=0):
                data, _ = batch
                if torch.cuda.device_count() > 0:
                    data = data.cuda()
                i += 1
                # predict
                klogits, ulogits = self.model((data, labels))
                # get loss for recording
                loss_total, loss_main, loss_aux, loss_neg = self.loss(klogits, ulogits, labels)
                # close-set accuracy
                acc = count_acc(args, ulogits[:self.num_query], label_q)
                logits_predict = F.softmax(ulogits, dim=1)[:, -1]
                # open-set accuracy
                auroc, aupr, f1_score, fpr95 = roc_area_score(args, logits_predict, label_2cls, True)

                record[i - 1, 0] = loss_total.item()
                record[i - 1, 1] = acc
                record[i - 1, 2] = auroc
                record[i - 1, 3] = aupr
                record[i - 1, 4] = f1_score
                record[i - 1, 5] = fpr95

        assert (i == record.shape[0])
        va_list, vap_list = [None]*record.shape[1], [None]*record.shape[1]
        for j in range(record.shape[1]):
            va_list[j], vap_list[j] = compute_confidence_interval(record[:, j])

        # train mode
        self.model.train()
        if mode == 'test':
            self.logger.writer_log('\nTest ETA:{}'.format(self.timer.measure()))
        return va_list, vap_list

