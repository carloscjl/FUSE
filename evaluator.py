import time

import models
import torch
import torch.optim as optim
import util
from torch.autograd import Variable
from scipy.fftpack import dct, idct
import numpy as np
import cv2
import torch.nn.functional as F

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    




class Evaluator():
    def __init__(self, data_loader, logger, config, args):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.data_loader = data_loader
        self.logger = logger
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 100
        self.config = config
        self.current_acc = 0
        self.current_acc_top5 = 0
        self.confusion_matrix = torch.zeros(config.num_classes, config.num_classes)
        self.args = args
        self.collect_features = self.args.tsne
        # return

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.confusion_matrix = torch.zeros(self.config.num_classes, self.config.num_classes)
        return

    def eval(self, epoch, model):
        model.eval()
        collect_feats = getattr(self, "collect_features", True)

        if collect_feats:
            self._feat_buffer = []
            self._label_buffer = []
        for i, (images, labels) in enumerate(self.data_loader["test_dataset"]):
            start = time.time()
            log_payload = self.eval_batch(images=images, labels=labels, model=model, batch_idx=i, collect_feats=collect_feats)
            end = time.time()
            time_used = end - start
        display = util.log_display(epoch=epoch,
                                   global_step=i,
                                   time_elapse=time_used,
                                   **log_payload)
        if self.logger is not None:
            self.logger.info(display)
            
            
        if collect_feats:
            feats = torch.cat(self._feat_buffer, dim=0).cpu().numpy()   # [N, D]
            labels = torch.cat(self._label_buffer, dim=0).cpu().numpy() # [N]

            
            feats = StandardScaler().fit_transform(feats)
            tsne = TSNE(n_components=2, perplexity=30, random_state=0, init="pca", learning_rate="auto")
            Z = tsne.fit_transform(feats)  # [N, 2]

            
            plt.figure(figsize=(8, 7))
            scatter = plt.scatter(Z[:,0], Z[:,1], c=labels, cmap="tab20", s=5, alpha=0.6)
            plt.colorbar(scatter, ticks=range(len(set(labels))))
            # acc = log_payload["acc_avg"] * 100
            for key, value in log_payload.items():
                if key == "acc_avg":
                    acc = value * 100
                    break
            plt.title(f"t-SNE of features (epoch {epoch}) Acc: {acc:.2f}%")

            
            if "Poison" not in self.args.train_data_type: method = "clean"
            elif 'EMN' in self.args.generator_filepath: method = "emn"
            elif 'GUE' in self.args.generator_filepath: method = "gue"
            elif 'LSP' in self.args.generator_filepath: method = "lsp"
            elif 'TUE' in self.args.generator_filepath: method = "tue"
            elif 'certified-data-learnability' in self.args.generator_filepath: method = "pue"
            else: method = "fuse"
            # method = "tue2"
            save_dir = os.path.join("visualization_tsne", "{}_all".format(method))
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"tsne_epoch{epoch}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            if self.logger is not None:
                self.logger.info(f"t-SNE saved to {save_path}")
        return

    def eval_batch(self, images, labels, model, batch_idx=0, collect_feats=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        with torch.no_grad():
            features, pred = model(images)
            loss = self.criterion(pred, labels)
            acc, acc5 = util.accuracy(pred, labels, topk=(1, 5))
            _, preds = torch.max(pred, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                self.confusion_matrix[t.long(), p.long()] += 1
                
            if collect_feats:
                if features is None:
                    
                    features = pred
                self._feat_buffer.append(features.detach())
                self._label_buffer.append(labels.detach())

        self.loss_meters.update(loss.item(), n=images.size(0))
        self.acc_meters.update(acc.item(), n=images.size(0))
        self.acc5_meters.update(acc5.item(), n=images.size(0))
        payload = {"acc": acc.item(),
                   "acc_avg": self.acc_meters.avg,
                   "acc5": acc5.item(),
                   "acc5_avg": self.acc5_meters.avg,
                   "loss": loss.item(),
                   "loss_avg": self.loss_meters.avg}
        return payload

    def _pgd_whitebox(self, model, X, y, random_start=True, epsilon=0.031, num_steps=20, step_size=0.003):
        model.eval()
        out = model(X)
        acc = (out.data.max(1)[1] == y.data).float().sum()
        X_pgd = Variable(X.data, requires_grad=True)
        if random_start:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()

            with torch.enable_grad():
                loss = torch.nn.CrossEntropyLoss()(model(X_pgd), y)
            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        acc_pgd = (model(X_pgd).data.max(1)[1] == y.data).float().sum()
        return acc.item(), acc_pgd.item()
