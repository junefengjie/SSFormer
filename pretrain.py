import pandas as pd
import logging
import numpy as np
import torch
import math, os, shutil
import functools
import matplotlib.pyplot as plt
from typing import Tuple
from torch import nn, Tensor
from torch._C import TensorType
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import optimizer
from torch.utils.data import dataset, DataLoader
from torch.nn import CrossEntropyLoss
# from transformers.utils.dummy_vision_objects import ImageFeatureExtractionMixin
from model.transformer import TransformerPretrain
from model.utils import *
from torch.utils.tensorboard import SummaryWriter
#from model.dataset import MOF_pretrain_Dataset
from dataset.dataset_multiview import CIFData,collate_pool,get_train_val_test_loader
from datetime import datetime, timedelta
from time import time
from model.mlm_pytorch_new import MLM
from loss.barlow_twins import BarlowTwinsLoss
import yaml
from tqdm import tqdm
from model.cgcnn_pretrain import CrystalGraphConvNet
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.simplefilter("ignore")
warnings.warn("deprecated", UserWarning)
warnings.warn("deprecated", FutureWarning)

warnings.simplefilter("ignore")
warnings.warn("deprecated", UserWarning)
warnings.warn("deprecated", FutureWarning)

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config_multiview.yaml', os.path.join(model_checkpoints_folder, 'config_multiview.yaml'))

class Multiview(object):
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time
        log_dir = os.path.join('runs_multiview', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dual_criterion = BarlowTwinsLoss(self.device, config['batch_size'], **config['barlow_loss'])
        self.dataset = CIFData(**self.config['graph_dataset'])

        collate_fn = collate_pool
        self.train_loader, self.valid_loader = get_train_val_test_loader(
            dataset=self.dataset,
            collate_fn=collate_fn,
            pin_memory=self.config['gpu'],
            batch_size=self.config['batch_size'],
            **self.config['dataloader']
        )

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            self.config['cuda'] = True
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
            self.config['cuda'] = False
        print("Running on:", device)
        return device

    def _step(self, model_t, model_g, data_i, data_j, epsilon=0):
        zjs = model_g(*data_j)  # [N,C]
        zis = model_t(data_i)
        loss_barlow = self.dual_criterion(zis, zjs)
        return loss_barlow

    def train(self):
        structures, _, _ = self.dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]

        model_t = TransformerPretrain(**self.config["Transformer"]).to(self.device)
        model_g = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len, **self.config['model_cgcnn']).to(self.device)
        model_t, model_g = self._load_pre_trained_weights(model_t, model_g)

        optimizer = optim.Adam(list(model_t.parameters()) + list(model_g.parameters()), lr=self.config['optim']['init_lr'], weight_decay=eval(self.config['optim']['weight_decay']))
        scheduler = CosineAnnealingLR(optimizer, T_max=len(self.train_loader), eta_min=0, last_epoch=-1)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = float('inf')

        for epoch_counter in range(self.config['epochs']):
            # 使用 tqdm 显示训练进度
            for bn, (input_1, input_2, _) in enumerate(tqdm(self.train_loader, desc=f'Epoch {epoch_counter+1}/{self.config["epochs"]}')):
                if self.config['cuda']:
                    input_graph = (Variable(input_1[0].to(self.device, non_blocking=True)),
                                   Variable(input_1[1].to(self.device, non_blocking=True)),
                                   input_1[2].to(self.device, non_blocking=True),
                                   [crys_idx.to(self.device, non_blocking=True) for crys_idx in input_1[3]])
                    input_transformer = input_2.to(self.device, non_blocking=True)
                else:
                    input_graph = (Variable(input_1[0]),
                                   Variable(input_1[1]),
                                   input_1[2],
                                   input_1[3])
                    input_transformer = input_2

                loss = self._step(model_t, model_g, input_transformer, input_graph)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss.item(), global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    print(f"epoch: {epoch_counter}, 批次: {bn}, 训练损失: {loss.item()}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_iter += 1

            torch.cuda.empty_cache()

            # 验证模型
            if (epoch_counter + 1) % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model_t, model_g, self.valid_loader)
                print(f"epoch: {epoch_counter}, 验证损失: {valid_loss}")
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model_t.state_dict(), os.path.join(model_checkpoints_folder, 'model_t.pth'))
                    torch.save(model_g.state_dict(), os.path.join(model_checkpoints_folder, 'model_g.pth'))

                self.writer.add_scalar('valid_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            if (epoch_counter + 1) % 1 == 0:
                torch.save(model_t.state_dict(), os.path.join(model_checkpoints_folder, f'model_transformer_{epoch_counter}.pth'))
                torch.save(model_g.state_dict(), os.path.join(model_checkpoints_folder, f'model_graph_{epoch_counter}.pth'))

            if epoch_counter >= 5:
                scheduler.step()

    def _load_pre_trained_weights(self, model_t, model_g):
        try:
            checkpoints_folder = os.path.join('./runs_multiview', self.config['fine_tune_from'], 'checkpoints')
            state_dict_t = torch.load(os.path.join(checkpoints_folder, 'model_transformer_11.pth'), map_location=self.config['gpu'])
            model_t.load_state_dict(state_dict_t)

            state_dict_g = torch.load(os.path.join(checkpoints_folder, 'model_graph_11.pth'), map_location=self.config['gpu'])
            model_g.load_state_dict(state_dict_g)

            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model_t, model_g

    def _validate(self, model_t, model_g, valid_loader):
        with torch.no_grad():
            model_t.eval()
            model_g.eval()

            loss_total = 0.0
            total_num = 0
            # 使用 tqdm 显示验证进度
            for input_1, input_2, batch_cif_ids in tqdm(valid_loader, desc='Validation'):
                if self.config['cuda']:
                    input_graph = (Variable(input_1[0].to(self.device, non_blocking=True)),
                                   Variable(input_1[1].to(self.device, non_blocking=True)),
                                   input_1[2].to(self.device, non_blocking=True),
                                   [crys_idx.to(self.device, non_blocking=True) for crys_idx in input_1[3]])
                    input_transformer = input_2.to(self.device, non_blocking=True)
                else:
                    input_graph = (Variable(input_1[0]),
                                   Variable(input_1[1]),
                                   input_1[2],
                                   input_1[3])
                    input_transformer = input_2

                loss = self._step(model_t, model_g, input_transformer, input_graph)
                loss_total += loss.item() * len(batch_cif_ids)
                total_num += len(batch_cif_ids)

            loss_total /= total_num
        torch.cuda.empty_cache()
        model_t.train()
        model_g.train()
        return loss_total

if __name__ == "__main__":
    config = yaml.load(open("config_multiview.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    mof_multiview = Multiview(config)
    mof_multiview.train()