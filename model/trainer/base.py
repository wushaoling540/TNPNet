import abc
import torch
import os.path as osp

from model.utils import (
    ensure_path,
    Averager, Timer, count_acc,
    compute_confidence_interval,
)
from model.logger import Logger


class Trainer(object, metaclass=abc.ABCMeta):
    """
    init the logger, step, epoch, max_step, several time for cost record,
             record the optimal metric[max_acc, max_acc_epoch, max_acc_interval]
    """

    def __init__(self, args):
        self.args = args
        # ensure_path(
        #     self.args.save_path,
        #     scripts_to_save=['model/models', 'model/networks', __file__],
        # )
        self.logger = Logger(args, osp.join(args.save_path))

        self.train_step = 0
        self.train_epoch = 0
        self.max_steps = torch.cuda.device_count() * args.episodes_per_epoch * args.max_epoch if args.multi_gpu else args.episodes_per_epoch * args.max_epoch
        self.dt, self.ft = Averager(), Averager()  # data cost time, forward cost time
        self.bt, self.ot = Averager(), Averager()  # backward cost time, optimizer cost time
        self.timer = Timer()

        # train statistics, [0916]should add the open set result record
        self.trlog = {}
        self.trlog['max_acc'] = 0.0
        self.trlog['max_auroc'] = 0.0
        # [maxa_epoch, maxa_acc, maxa_acc_interval, maxa_acc_interval, maxa_roc_interval]
        self.trlog['max_acc_results'] = [0, 0.0, 0.0, 0.0, 0.0]
        # [maxr_epoch, maxr_acc, maxr_acc_interval, maxr_acc_interval, maxr_roc_interval]
        self.trlog['max_auroc_results'] = [0, 0.0, 0.0, 0.0, 0.0]

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate(self, data_loader):
        pass

    def logger_best_result(self):
        trlog = self.trlog
        # Display the current training best result...
        self.logger.writer_log('\tbest ACC   epoch {:03d}, VAL acc={:.2f}+{:.2f}, auroc={:.2f}+{:.2f}'.format(
            trlog['max_acc_results'][0], trlog['max_acc_results'][1] * 100, trlog['max_acc_results'][2] * 100,
                                         trlog['max_acc_results'][3] * 100, trlog['max_acc_results'][4] * 100))
        self.logger.writer_log('\tbest AUROC epoch {:03d}, VAL acc={:.2f}+{:.2f}, auroc={:.2f}+{:.2f}'.format(
            trlog['max_auroc_results'][0], trlog['max_auroc_results'][1] * 100, trlog['max_auroc_results'][2] * 100,
                                           trlog['max_auroc_results'][3] * 100, trlog['max_auroc_results'][4] * 100))

    def try_evaluate(self, epoch):
        args = self.args
        trlog = self.trlog
        if epoch % args.eval_interval == 0:
            # Validating...
            vas, vaps = self.evaluate(self.val_loader, 'val')
            va, vap, va_open, vap_open = vas[1], vaps[1], vas[2], vaps[2]

            # Display the current training best result...
            self.logger_best_result()

            self.logger.add_scalar('val_loss', float(vas[0]), epoch)
            self.logger.add_scalar('val_acc', float(vas[1]), epoch)
            self.logger.add_scalar('val_auroc', float(vaps[2]), epoch)

            # Display the validating result...
            self.logger.writer_log('\tepoch {:03d}, val, loss={:.4f} | acc={:.2f}+{:.2f} | auroc={:.2f}+{:.2f}'
                                   ' | aupr={:.2f}+{:.2f} | f1={:.2f}+{:.2f} | fpr95={:.2f}+{:.2f}'.format(
                epoch, vas[0], va * 100, vap * 100, va_open * 100, vap_open * 100,
                               vas[3] * 100, vaps[3] * 100, vas[4] * 100, vaps[4] * 100, vas[5] * 100, vaps[5] * 100))

            # save two optimal models
            if va >= trlog['max_acc']:
                trlog['max_acc'] = va
                trlog['max_acc_results'] = [epoch, va, vap, va_open, vap_open]
                self.save_model('max_acc')
            if va_open >= self.trlog['max_auroc']:
                trlog['max_auroc'] = va_open
                trlog['max_auroc_results'] = [epoch, va, vap, va_open, vap_open]
                self.save_model('max_auroc')
            if epoch == args.eval_interval:
                self.try_test('train_test')

    def try_test(self, mode='test'):
        vas, vaps = self.evaluate(self.test_loader, mode)
        va, vap, va_open, vap_open = vas[1], vaps[1], vas[2], vaps[2]

        self.logger_best_result()

        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_auroc'] = va_open
        self.trlog['test_auroc_interval'] = vap_open

        self.logger.writer_log('【Test】 acc={:.2f}+{:.2f} | auroc={:.2f}+{:.2f} '
                               '| aupr={:.2f}+{:.2f} | f1={:.2f}+{:.2f} | fpr95={:.2f}+{:.2f}'.format(
            va * 100, vap * 100, va_open * 100, vap_open * 100,
            vas[3] * 100, vaps[3] * 100, vas[4] * 100, vaps[4] * 100, vas[5] * 100, vaps[5] * 100))

    def try_logging(self, tl1, tl2, tl3, tl4, ta, troc, tg=None):
        args = self.args
        if self.train_step % args.log_interval == 0:
            self.logger.writer_log(
                'epoch {:03d}, train {:06g}/{:06g}'
                '\n\ttotal loss={:.4f}, loss_main={:.4f}, loss_aux={:.4f}, loss_neg={:.4f} '
                '| acc={:.4f}, roc={:.4f} | lr={:.4g}'.format(
                    self.train_epoch, self.train_step, self.max_steps,
                    tl1.item(), tl2.item(), tl3.item(), tl4.item(),
                    ta.item(), troc.item(), self.optimizer.param_groups[0]['lr']
                )
            )
            self.logger.add_scalar('train_total_loss', tl1.item(), self.train_step)
            self.logger.add_scalar('train_loss_main', tl2.item(), self.train_step)

            self.logger.add_scalar('train_loss_aux', tl3.item(), self.train_step)
            self.logger.add_scalar('train_loss_neg', tl4.item(), self.train_step)

            if tg is not None:
                self.logger.add_scalar('grad_norm', tg.item(), self.train_step)

            self.logger.dump()

    def save_model(self, name):
        file_to_save = {
            'epoch': self.train_epoch,
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'max_acc_results': self.trlog['max_acc_results'],
            'max_auroc_results': self.trlog['max_auroc_results']
        }
        torch.save(
            file_to_save,  # dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, name + '.pth')
        )

    def final_record(self):
        # save the best performance in a txt file
        trlog = self.trlog
        if self.args.max_checkpoint_type == 'max_auroc':
            max_epoch = trlog['max_auroc_results'][0]
        else:
            max_epoch = trlog['max_acc_results'][0]
        with open(osp.join(self.args.save_path, 'ep{:03d}_{:.4f}_{:.4f}'.format(max_epoch, self.trlog['test_acc'],
                                                                                self.trlog['test_auroc'])),
                  'w') as f:
            f.write("Modify details: {:s}\n".format(self.args.version_desc))

            f.write("Max checkpoint type: {:s}, GPU: {:s}\n".format(self.args.max_checkpoint_type, self.args.gpu))

            f.write('best ACC   epoch {:03d}, Best acc={:.2f}+{:.2f} | auroc={:.2f}+{:.2f}\n'.format(
                trlog['max_acc_results'][0], trlog['max_acc_results'][1] * 100, trlog['max_acc_results'][2] * 100,
                                             trlog['max_acc_results'][3] * 100, trlog['max_acc_results'][4] * 100
            ))
            f.write('best AUROC epoch {:03d}, Best acc={:.2f}+{:.2f} | auroc={:.2f}+{:.2f}\n'.format(
                trlog['max_auroc_results'][0], trlog['max_auroc_results'][1] * 100, trlog['max_auroc_results'][2] * 100,
                                               trlog['max_auroc_results'][3] * 100, trlog['max_auroc_results'][4] * 100
            ))
            f.write('Test acc={:.2f} + {:.2f} | auroc={:.2f} + {:.2f}\n'.format(
                self.trlog['test_acc'] * 100, self.trlog['test_acc_interval'] * 100,
                self.trlog['test_auroc'] * 100, self.trlog['test_auroc_interval'] * 100))

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.model.__class__.__name__
        )
