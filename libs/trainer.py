import torch
import time, os
import numpy as np
from tensorboardX import SummaryWriter
from utils.timer import Timer, AverageMeter
from tqdm import tqdm
# import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self, args):
        # parameters
        self.max_epoch = args.max_epoch
        self.training_max_iter = args.training_max_iter
        self.val_max_iter = args.val_max_iter
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.gpu_mode = args.gpu_mode
        self.verbose = args.verbose

        self.model = args.model
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_interval = args.scheduler_interval
        self.snapshot_interval = args.snapshot_interval
        self.evaluate_interval = args.evaluate_interval
        self.evaluate_metric = args.evaluate_metric
        self.metric_weight = args.metric_weight
        self.transformation_loss_start_epoch = args.transformation_loss_start_epoch
        self.writer = SummaryWriter(log_dir=args.tboard_dir)

        self.train_loader = args.train_loader
        self.val_loader = args.val_loader

        if self.gpu_mode:
            self.model = self.model.cuda()

        if args.pretrain != '':
            self._load_pretrain(args.pretrain)

    def train(self):
        best_reg_recall = 0
        print('training start!!')
        start_time = time.time()

        self.model.train()
        res = self.evaluate(0)
        print(f'Evaluation: Epoch 0: SM Loss {res["sm_loss"]:.2f} Class Loss {res["class_loss"]:.2f} Trans Loss {res["trans_loss"]:.2f} Recall {res["reg_recall"]:.2f}')
        for epoch in range(self.max_epoch):
            self.train_epoch(epoch + 1)  # start from epoch 1

            if (epoch + 1) % self.evaluate_interval == 0 or epoch == 0:
                res = self.evaluate(epoch + 1)
                print(f'Evaluation: Epoch {epoch+1}: SM Loss {res["sm_loss"]:.2f} Class Loss {res["class_loss"]:.2f} Trans Loss {res["trans_loss"]:.2f} Recall {res["reg_recall"]:.2f}')
                if res['reg_recall'] > best_reg_recall:
                    best_reg_recall = res['reg_recall']
                    self._snapshot('best')

            if (epoch + 1) % self.scheduler_interval == 0:
                self.scheduler.step()

            if (epoch + 1) % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)

        # finish all epoch
        print("Training finish!... save training results")

    def train_epoch(self, epoch):
        # create meters and timers
        meter_list = ['class_loss', 'trans_loss', 'sm_loss', 'reg_recall', 're', 'te', 'precision', 'recall', 'f1']
        meter_dict = {}
        for key in meter_list:
            meter_dict[key] = AverageMeter()
        data_timer, model_timer = Timer(), Timer()

        num_iter = int(len(self.train_loader.dataset) / self.batch_size)
        num_iter = min(self.training_max_iter, num_iter)
        trainer_loader_iter = self.train_loader.__iter__()
        for iter in range(num_iter):
            data_timer.tic()
            (corr_pos, src_keypts, tgt_keypts, gt_trans, gt_labels) = trainer_loader_iter.next()
            if self.gpu_mode:
                corr_pos, src_keypts, tgt_keypts, gt_trans, gt_labels = \
                    corr_pos.cuda(), src_keypts.cuda(), tgt_keypts.cuda(), gt_trans.cuda(), gt_labels.cuda()
            data = {
                'corr_pos': corr_pos,
                'src_keypts': src_keypts,
                'tgt_keypts': tgt_keypts,
            }
            data_timer.toc()

            model_timer.tic()
            # forward
            self.optimizer.zero_grad()
            res = self.model(data)
            pred_trans, pred_labels = res['final_trans'], res['final_labels']
            # classification loss
            class_stats = self.evaluate_metric['ClassificationLoss'](pred_labels, gt_labels)
            class_loss = class_stats['loss']
            # spectral matching loss
            sm_loss = self.evaluate_metric['SpectralMatchingLoss'](res['M'], gt_labels)
            # transformation loss
            trans_loss, reg_recall, re, te, rmse = self.evaluate_metric['TransformationLoss'](pred_trans, gt_trans, src_keypts, tgt_keypts, pred_labels)

            loss = self.metric_weight['ClassificationLoss'] * class_loss + self.metric_weight['SpectralMatchingLoss'] * sm_loss
            if epoch > self.transformation_loss_start_epoch and self.metric_weight['TransformationLoss'] > 0.0:
                loss += self.metric_weight['TransformationLoss'] * trans_loss
            
            stats = {
                'class_loss': float(class_loss),
                'sm_loss': float(sm_loss),
                'trans_loss': float(trans_loss),
                'reg_recall': float(reg_recall),
                're': float(re),
                'te': float(te),
                'precision': class_stats['precision'],
                'recall': class_stats['recall'],
                'f1': class_stats['f1'],
            }

            # backward
            loss.backward()
            do_step = True
            for param in self.model.parameters():
                if param.grad is not None:
                    if (1 - torch.isfinite(param.grad).long()).sum() > 0:
                        do_step = False
                        break
            if do_step is True:
                self.optimizer.step()
            model_timer.toc()

            if not np.isnan(float(loss)):
                for key in meter_list:
                    if not np.isnan(stats[key]):
                        meter_dict[key].update(stats[key])

            else:  # debug the loss calculation process.
                import pdb
                pdb.set_trace()

            if (iter + 1) % 100 == 0 and self.verbose:
                curr_iter = num_iter * (epoch - 1) + iter
                for key in meter_list:
                    self.writer.add_scalar(f"Train/{key}", meter_dict[key].avg, curr_iter)

                print(f"Epoch: {epoch} [{iter+1:4d}/{num_iter}] "
                      f"sm_loss: {meter_dict['sm_loss'].avg:.2f} "
                      f"class_loss: {meter_dict['class_loss'].avg:.2f} "
                      f"trans_loss: {meter_dict['trans_loss'].avg:.2f} "
                      f"reg_recall: {meter_dict['reg_recall'].avg:.2f}% "
                      f"re: {meter_dict['re'].avg:.2f}degree "
                      f"te: {meter_dict['te'].avg:.2f}cm "
                      f"data_time: {data_timer.avg:.2f}s "
                      f"model_time: {model_timer.avg:.2f}s "
                      )

    def evaluate(self, epoch):
        self.model.eval()

        # create meters and timers
        meter_list = ['class_loss', 'trans_loss', 'sm_loss', 'reg_recall', 're', 'te', 'precision', 'recall', 'f1']
        meter_dict = {}
        for key in meter_list:
            meter_dict[key] = AverageMeter()
        data_timer, model_timer = Timer(), Timer()

        num_iter = int(len(self.val_loader.dataset) / self.batch_size)
        num_iter = min(self.val_max_iter, num_iter)
        val_loader_iter = self.val_loader.__iter__()
        for iter in range(num_iter):
            data_timer.tic()
            (corr_pos, src_keypts, tgt_keypts, gt_trans, gt_labels) = val_loader_iter.next()
            if self.gpu_mode:
                corr_pos, src_keypts, tgt_keypts, gt_trans, gt_labels = \
                    corr_pos.cuda(), src_keypts.cuda(), tgt_keypts.cuda(), gt_trans.cuda(), gt_labels.cuda()
            data = {
                'corr_pos': corr_pos,
                'src_keypts': src_keypts,
                'tgt_keypts': tgt_keypts,
            }
            data_timer.toc()

            model_timer.tic()
            # forward
            res = self.model(data)
            pred_trans, pred_labels = res['final_trans'], res['final_labels']
            # classification loss
            class_stats = self.evaluate_metric['ClassificationLoss'](pred_labels, gt_labels)
            class_loss = class_stats['loss']
            # spectral matching loss
            sm_loss = self.evaluate_metric['SpectralMatchingLoss'](res['M'], gt_labels)
            # transformation loss
            trans_loss, reg_recall, re, te, rmse = self.evaluate_metric['TransformationLoss'](pred_trans, gt_trans, src_keypts, tgt_keypts, pred_labels)
            model_timer.toc()

            stats = {
                'class_loss': float(class_loss),
                'sm_loss': float(sm_loss),
                'trans_loss': float(trans_loss),
                'reg_recall': float(reg_recall),
                're': float(re),
                'te': float(re),
                'precision': class_stats['precision'],
                'recall': class_stats['recall'],
                'f1': class_stats['f1'],
            }
            for key in meter_list:
                if not np.isnan(stats[key]):
                    meter_dict[key].update(stats[key])

        self.model.train()
        res = {
            'sm_loss': meter_dict['sm_loss'].avg,
            'class_loss': meter_dict['class_loss'].avg,
            'reg_recall': meter_dict['reg_recall'].avg,
            'trans_loss': meter_dict['trans_loss'].avg,
        }
        for key in meter_list:
            self.writer.add_scalar(f"Val/{key}", meter_dict[key].avg, epoch)

        return res

    def _snapshot(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f"model_{epoch}.pkl"))
        print(f"Save model to {self.save_dir}/model_{epoch}.pkl")

    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        self.model.load_state_dict(state_dict)
        print(f"Load model from {pretrain}.pkl")
