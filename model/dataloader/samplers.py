import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_task, n_cls, n_per, n_cls_open, n_per_open):
        self.n_batch = n_batch
        self.n_task = n_task
        self.n_cls = n_cls  # the number of sampling's category
        self.n_per = n_per  # the number of samples for per closed
        self.n_cls_open = n_cls_open  # the number of sampling's open category
        self.n_per_open = n_per_open  # the number of samples for per open sampling's category

        label = np.array(label)
        self.m_ind = []  # group the indices by category for sampling, miniImage:64
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch_task = []
            for i_task in range(self.n_task):
                batch_closed, batch_open = [], []
                classes_perm = torch.randperm(len(self.m_ind))
                classes = classes_perm[:self.n_cls]
                classes_open = classes_perm[self.n_cls: self.n_cls + self.n_cls_open]
                for c in classes:
                    l = self.m_ind[c]
                    pos = torch.randperm(len(l))[:self.n_per]
                    batch_closed.append(l[pos])
                # stack the support and query data by position, e.g., the index of 0 in each list will stack together.
                batch_closed = torch.stack(batch_closed).t().reshape(-1)
                for c in classes_open:
                    l = self.m_ind[c]
                    pos = torch.randperm(len(l))[:self.n_per_open]
                    batch_open.append(l[pos])
                # batch = torch.stack(batch).t().reshape(-1)
                batch_open = torch.stack(batch_open).t().reshape(-1)
                batch_task.append(torch.cat((batch_closed, batch_open), dim=0))
            batch = torch.cat(batch_task, dim=0)
            yield batch


class RandomSampler():

    def __init__(self, label, n_batch, n_per):
        self.n_batch = n_batch
        self.n_per = n_per
        self.label = np.array(label)
        self.num_label = self.label.shape[0]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = torch.randperm(self.num_label)[:self.n_per]
            yield batch
            
            
# sample for each class
class ClassSampler():

    def __init__(self, label, n_per=None):
        self.n_per = n_per
        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return len(self.m_ind)

    def __iter__(self):
        classes = torch.arange(len(self.m_ind))
        for c in classes:
            l = self.m_ind[int(c)]
            if self.n_per is None:
                pos = torch.randperm(len(l))
            else:
                pos = torch.randperm(len(l))[:self.n_per]
            yield l[pos]
            
            
# for ResNet Fine-Tune, which output the same index of task examples several times
class InSetSampler():

    def __init__(self, n_batch, n_sbatch, pool): # pool is a tensor
        self.n_batch = n_batch
        self.n_sbatch = n_sbatch
        self.pool = pool
        self.pool_size = pool.shape[0]

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = self.pool[torch.randperm(self.pool_size)[:self.n_sbatch]]
            yield batch