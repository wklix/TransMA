# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function
from ast import Load

import logging
import copy
import os
import pandas as pd
import numpy as np
import csv
import torch

import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch.optim import Adam,AdamW
from torch.nn.utils import clip_grad_norm_
from transformers.optimization import get_linear_schedule_with_warmup
from utils import Metrics
from utils import logger
from .split import Splitter
from .fusion import MyModule
from .triple_loss import BatchAllTtripletLoss_multi_module_version
from tqdm import tqdm
from unicore.utils import get_activation_fn
from itertools import chain
import time
import sys

class Trainer(object):
    def __init__(self, save_path=None, **params):
        self.save_path = save_path
        self.task = params.get('task', None)

        if self.task != 'repr':
            self.metrics_str = params['metrics']
            self.metrics = Metrics(self.task, self.metrics_str)
        self._init_trainer(**params)


    def _init_trainer(self, **params):
        ### init common params ###
        self.split_method = params.get('split_method', '5fold_random')
        self.split_seed = params.get('split_seed', 42)###原始是42
        self.seed = params.get('seed', 42)
        self.set_seed(self.seed)
        self.splitter = Splitter(self.split_method, self.split_seed)
        self.logger_level = int(params.get('logger_level', 1))
        ### init NN trainer params ###
        self.learning_rate = float(params.get('learning_rate', 1e-4))
        self.batch_size = params.get('batch_size', 32)
        self.max_epochs = params.get('epochs', 100)
        self.warmup_ratio = params.get('warmup_ratio', 0.1)
        self.patience = params.get('patience', 10)
        self.max_norm = params.get('max_norm', 1.0)
        self.cuda = params.get('cuda', False)
        self.amp = params.get('amp', False)
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.cuda else "cpu")
        self.scaler = torch.cuda.amp.GradScaler(
        ) if self.device.type == 'cuda' and self.amp == True else None
        self.regressor = nn.Linear(1024, 512).to(self.device)  
        self.regressor1 = nn.Linear(512, 1).to(self.device)
        self.gate = nn.Parameter(torch.rand(4, 512)).to(self.device)
        self.MyModule = MyModule(self.device)
        self.triple_loss=BatchAllTtripletLoss_multi_module_version(margin=0.1)
    def decorate_batch(self, batch, feature_name=None):
        return self.decorate_torch_batch(batch)

    def decorate_graph_batch(self, batch):
        net_input, net_target = {'net_input': batch.to(
            self.device)}, batch.y.to(self.device)
        if self.task in ['classification', 'multiclass', 'multilabel_classification']:
            net_target = net_target.long()
        else:
            net_target = net_target.float()
        return net_input, net_target

    def decorate_torch_batch(self, batch):
        """function used to decorate batch data
        """
        net_input, net_target = batch
        #print(net_target)
        if isinstance(net_input, dict):
            net_input, net_target = {
                k: v.to(self.device) for k, v in net_input.items()}, net_target.to(self.device)
        else:
            net_input, net_target = {'net_input': net_input.to(
                self.device)}, net_target.to(self.device)
        if self.task == 'repr':
            net_target = None
        elif self.task in ['classification', 'multiclass', 'multilabel_classification']:
            net_target = net_target.long()
        else:
            net_target = net_target.float()
            #weight= weight.float()
        return net_input, net_target

    def fit_predict(self, Unimol, traindataset, validdataset, model, train_dataset, valid_dataset, loss_func, activation_fn, dump_dir, fold, target_scaler, feature_name=None):
        #weights = weighted_mse_loss()
        model = model.to(self.device)
        Unimol = Unimol.to(self.device)
        train_dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=None,
            drop_last=True,
        )
        Unimol_dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=traindataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=Unimol.batch_collate_fn,
            drop_last=True,
        )
        #for i in range(10):
           #print(train_dataloader.dataset.label[i])
           #print(Unimol_dataloader.dataset.label[i])
        # remove last batch, bs=1 can not work on batchnorm1d
        min_val_loss = float("inf")
        max_score = float("-inf")
        wait = 0
        ### init optimizer ###
        num_training_steps = len(train_dataloader) * self.max_epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        parameters_to_optimize = chain(model.parameters(), Unimol.parameters(),self.MyModule.parameters())
        optimizer = AdamW(parameters_to_optimize, lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        #optimizer1 = Adam(Unimol.parameters(), lr=self.learning_rate, eps=1e-6)
        #scheduler1 = get_linear_schedule_with_warmup(
        #    optimizer1, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        for epoch in range(self.max_epochs):
            model = model.train()
            Unimol = Unimol.train()
            self.MyModule = self.MyModule.train()
            # Progress Bar
            start_time = time.time()
            batch_bar = tqdm(total=len(train_dataloader), dynamic_ncols=True,
                             leave=False, position=0, desc='Train', ncols=5)
            trn_loss = []
            encodings = []
            labels1=[]
            #for i, batch in enumerate(Unimol_dataloader):
            #print(train_dataset.data[0])
            for i, (batch, unimol_batch) in enumerate(zip(train_dataloader, Unimol_dataloader)):
                #print(list(zip(train_dataloader.dataset.label, Unimol_dataloader.dataset.label)))
                net_input, net_target= self.decorate_batch(
                    batch, feature_name)
                unimol_input, unimol_target = self.decorate_batch(
                    unimol_batch, feature_name)
                optimizer.zero_grad()  # Zero gradients
                #optimizer1.zero_grad()  # Zero gradients
                if self.scaler and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        net_input_tensor = net_input['net_input']
                        net_input_tensor = torch.stack(list(net_input_tensor))
                        outputs,cls_repr1,out1 = model(net_input_tensor)
                        unimol_outputs,unimol_cls_repr1,out2 = Unimol(**unimol_input,labels=unimol_target, epoch=epoch)
                        merged_features=[]
                        for ii in range(len(unimol_cls_repr1)):
                            merged_tensor = torch.cat([unimol_cls_repr1[ii], cls_repr1[ii]], dim=1)
                            merged_features.append(merged_tensor)
                        preds = []
                        for feature in merged_features:
                            value,attention_weights,_ = self.MyModule(feature)
                            preds.append(value)
                        pred = torch.stack(preds, dim=0)
                        # non_zero_counts = torch.tensor([torch.nonzero(row).size(0) for row in net_input_tensor])
                        # last_non_zero_indices = []
                        # jj = 0
                        # for ii in non_zero_counts:
                        #     last_zero_index = cls_repr1[jj,0:ii,:]
                        #     last_zero_index = last_zero_index.mean(dim=0)
                        #     jj= jj+1
                        #     last_non_zero_indices.append(last_zero_index)
                        # cls_repr1=torch.stack(last_non_zero_indices)
                        encodings.extend(unimol_outputs.data.cpu().numpy())
                        labels1.extend(net_target.data.squeeze(1).cpu().numpy())
                        tripleloss=self.triple_loss(out1,out2)
                        loss = torch.mean(((pred - net_target) ** 2))
                        loss =loss+3*tripleloss
                else:
                    with torch.set_grad_enabled(True):
                        print(11111111111)
                        outputs,cls_repr1 = model(**net_input,labels=net_target, epoch=epoch)
                        loss = loss_func(outputs, net_target)
                trn_loss.append(float(loss.data))
                # tqdm lets you add some details so you can monitor training as you train.
                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                    loss="{:.04f}".format(float(sum(trn_loss) / (i + 1))),
                    lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])))
                if self.scaler and self.device.type == 'cuda':
                    # This is a replacement for loss.backward()

                    self.scaler.scale(loss).backward()
                    # unscale the gradients of optimizer's assigned params in-place
                    self.scaler.unscale_(optimizer)
                    #self.scaler.unscale_(optimizer1)
                    # Clip the norm of the gradients to max_norm.
                    clip_grad_norm_(model.parameters(), self.max_norm)
                    clip_grad_norm_(Unimol.parameters(), self.max_norm)
                    # This is a replacement for optimizer.step()
                    self.scaler.step(optimizer)
                    #self.scaler.step(optimizer1)
                    self.scaler.update()
                else:
                    loss.backward()
                    clip_grad_norm_(model.parameters(), self.max_norm)
                    clip_grad_norm_(Unimol.parameters(), self.max_norm)
                    optimizer.step()
                    #optimizer1.step()
                scheduler.step()
                #scheduler1.step()
                batch_bar.update()

            batch_bar.close()
            total_trn_loss = np.mean(trn_loss)
            encodings, labels1 = torch.from_numpy(np.vstack(encodings)).cuda(0), \
                               torch.from_numpy(np.hstack(labels1)).cuda(0)
            #model.FDS.update_last_epoch_stats(epoch)
            #model.FDS.update_running_stats(encodings, labels1, epoch)
            y_preds, val_loss, metric_score ,_,_,_= self.predict(
                 self.MyModule,Unimol, validdataset, model, valid_dataset, loss_func, activation_fn, dump_dir, fold, target_scaler, epoch, load_model=False, feature_name=feature_name)
            end_time = time.time()
            total_val_loss = np.mean(val_loss)
            _score = list(metric_score.values())[0]
            _metric = list(metric_score.keys())[0]
            message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, val_{}: {:.4f}, lr: {:.6f}, ' \
                '{:.1f}s'.format(epoch+1, self.max_epochs,
                                 total_trn_loss, total_val_loss,
                                 _metric, _score,
                                 optimizer.param_groups[0]['lr'],
                                 (end_time - start_time))
            logger.info(message)
            is_early_stop, min_val_loss, wait, max_score = self._early_stop_choice(
                wait, total_val_loss, min_val_loss, metric_score, max_score, Unimol,model, self.MyModule, dump_dir, fold, self.patience, epoch)
            if is_early_stop:
                break

        y_preds, _, _ ,_,_,_= self.predict(self.MyModule,Unimol, validdataset,model, valid_dataset, loss_func, activation_fn,
                                     dump_dir, fold, target_scaler, epoch, load_model=True, feature_name=feature_name)
        return y_preds

    def _early_stop_choice(self, wait, loss, min_loss, metric_score, max_score, Unimol,model, MyModule, dump_dir, fold, patience, epoch):
        ### hpyerparameter need to tune if you want to use early stop, currently find use loss is suitable in benchmark test. ###
        if not isinstance(self.metrics_str, str) or self.metrics_str in ['loss', 'none', '']:
            # loss 作为早停 直接用trainer里面的早停函数
            is_early_stop, min_val_loss, wait = self._judge_early_stop_loss(
                wait, loss, min_loss, Unimol,model, MyModule, dump_dir, fold, patience, epoch)
        else:
            # 到metric进行判断
            is_early_stop, min_val_loss, wait, max_score = self.metrics._early_stop_choice(
                wait, min_loss, metric_score, max_score, Unimol,model,  MyModule, dump_dir, fold, patience, epoch)
        return is_early_stop, min_val_loss, wait, max_score

    def _judge_early_stop_loss(self, wait, loss, min_loss, Unimol,model, MyModule, dump_dir, fold, patience, epoch):
        is_early_stop = False
        if loss <= min_loss:
            min_loss = loss
            wait = 0
            info = {'mamba_state_dict': model.state_dict(),'Unimol_state_dict': Unimol.state_dict(),'MyModule_state_dict': MyModule.state_dict(),}
            #info1 = {'model_state_dict': Unimol.state_dict()}
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(info, os.path.join(dump_dir, f'model_{fold}.pth'))
            #torch.save(info1, os.path.join(dump_dir, f'Unimol_{fold}.pth'))
        elif loss >= min_loss:
            wait += 1
            if wait == self.patience:
                logger.warning(f'Early stopping at epoch: {epoch+1}')
                is_early_stop = True
        return is_early_stop, min_loss, wait

    def predict(self, MyModule,Unimol, validdataset, model, dataset, loss_func, activation_fn, dump_dir, fold, target_scaler=None, epoch=1, load_model=False, feature_name=None):
        model = model.to(self.device)
        Unimol = Unimol.to(self.device)
        if load_model == True:
            load_model_path = os.path.join(dump_dir, f'model_{fold}.pth')
            #model_dict = torch.load(load_model_path, map_location=self.device)[
            #    "model_state_dict"]
            info = torch.load(load_model_path, map_location=self.device)
            model.load_state_dict(info['mamba_state_dict'])
            Unimol.load_state_dict(info['Unimol_state_dict'])
            MyModule.load_state_dict(info['MyModule_state_dict'])
            #model.load_state_dict(model_dict)
            logger.info("load model success!")
        dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=None,
        )
        Unimol_dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=validdataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=Unimol.batch_collate_fn,
        )
        model = model.eval()
        Unimol = Unimol.eval()
        MyModule = MyModule.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True,
                         position=0, leave=False, desc='val', ncols=5)
        val_loss = []
        y_preds = []
        y_truths = []
        encodings = []
       #for i, batch in enumerate(dataloader):
        for i, (batch, unimol_batch) in enumerate(zip(dataloader,Unimol_dataloader)):
            net_input, net_target = self.decorate_batch(batch, feature_name)
            unimol_input, unimol_target = self.decorate_batch(
                    unimol_batch, feature_name)
            # Get model outputs
            with torch.no_grad():
                net_input_tensor = net_input['net_input']
                net_input_tensor = torch.stack(list(net_input_tensor))
                outputs,cls_repr1,out1 = model(net_input_tensor)
                unimol_outputs,unimol_cls_repr1,out2 = Unimol(**unimol_input,labels=unimol_target, epoch=epoch)
                merged_features=[]
                for ii in range(len(unimol_cls_repr1)):
                    merged_tensor = torch.cat([unimol_cls_repr1[ii], cls_repr1[ii]], dim=1)
                    merged_features.append(merged_tensor)
                preds = []
                attention_weights = []
                for feature in merged_features:
                    value,attention_weight,x_pool = MyModule(feature)
                    preds.append(value)
                    attention_weights.append(attention_weight.cpu().numpy())
                    ##fusion encoding
                    encodings.append(x_pool.cpu().detach().numpy() )
                pred = torch.stack(preds, dim=0)
                # non_zero_counts = torch.tensor([torch.nonzero(row).size(0) for row in net_input_tensor])
                # last_non_zero_indices = []
                # jj = 0
                # for ii in non_zero_counts:
                #     last_zero_index = cls_repr1[jj,0:ii,:]
                #     last_zero_index = last_zero_index.mean(dim=0)
                #     jj= jj+1
                #     last_non_zero_indices.append(last_zero_index)
                # cls_repr1=torch.stack(last_non_zero_indices)
                if not load_model:
                    loss = loss_func(pred, net_target)
                    tripleloss=self.triple_loss(out1,out2)
                    loss =loss+3*tripleloss
                    val_loss.append(float(loss.data))
            y_preds.append(activation_fn(pred).cpu().numpy())
            y_truths.append(net_target.detach().cpu().numpy())
            ##mamba encoding
            #encodings.extend(out2.cpu().detach().numpy())
            ##trans encoding
            #encodings.extend(out1.cpu().detach().numpy())
            #featuresss=torch.tensor(merged_feature for merged_feature in merged_features).cuda(0)
           
            if not load_model:
                batch_bar.set_postfix(
                    Epoch="Epoch {}/{}".format(epoch+1, self.max_epochs),
                    loss="{:.04f}".format(float(np.sum(val_loss) / (i + 1))))

            batch_bar.update()
        y_preds = np.concatenate(y_preds)
        attention_weights = np.concatenate(attention_weights)
        y_truths = np.concatenate(y_truths)
        encodings = np.array(encodings)
        #print(encodings.shape)

        try:
            label_cnt = model.output_dim
        except:
            label_cnt = None

        if target_scaler is not None:
            inverse_y_preds = target_scaler.inverse_transform(y_preds)
            inverse_y_truths = target_scaler.inverse_transform(y_truths)
            metric_score = self.metrics.cal_metric(
                inverse_y_truths, inverse_y_preds, label_cnt=label_cnt) if not load_model else None
        else:
            metric_score = self.metrics.cal_metric(
                y_truths, y_preds, label_cnt=label_cnt) if not load_model else None
        batch_bar.close()
        #return y_preds, val_loss, metric_score
        return y_preds, val_loss, metric_score, encodings,y_truths,attention_weights

    def inference(self, model, dataset, feature_name=None, return_repr=True):
        model = model.to(self.device)
        dataloader = NNDataLoader(
            feature_name=feature_name,
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=model.batch_collate_fn,
        )
        model = model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True,
                         position=0, leave=False, desc='val', ncols=5)
        repr_dict = {"cls_repr": [], "atomic_reprs": []}
        for i, batch in enumerate(dataloader):
            net_input, _ = self.decorate_batch(batch, feature_name)
            with torch.no_grad():
                outputs = model(return_repr=return_repr, **net_input)
                assert isinstance(outputs, dict)
                for key, value in outputs.items():
                    if isinstance(value, list):
                        value_list = [item.cpu().numpy() for item in value]
                        repr_dict[key].extend(value_list)
                    else:
                        repr_dict[key].extend([value.cpu().numpy()])
        repr_dict["cls_repr"] = np.concatenate(repr_dict["cls_repr"]).tolist()
        return repr_dict

    def set_seed(self, seed):
        """function used to set a random seed
        Arguments:
            seed {int} -- seed number, will set to torch and numpy
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def gate_layer_fusion(self,feature1, feature2):

        input_size = feature1.size(1)
        sigmoid = nn.Sigmoid()
        combined_feature = torch.cat((feature1, feature2), dim=1)
        gate = sigmoid(self.regressor(combined_feature))
        fused_feature = feature1 * gate + feature2 * (1 - gate)

        return fused_feature
    
    def classification_head(self,features, pooler_dropout):
        activation_fn = nn.Tanh()  # 修改为ReLU激活函数
        dropout = nn.Dropout(p=pooler_dropout)

        # 将特征向量输入到分类头部模块中
        x = features
        x = dropout(x)
        x = activation_fn(x)
        x = dropout(x)
        x = self.regressor1(x)
        

        return x


def NNDataLoader(feature_name=None, dataset=None, batch_size=None, shuffle=False, collate_fn=None, drop_last=False):

    dataloader = TorchDataLoader(dataset=dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 collate_fn=collate_fn,
                                 drop_last=drop_last,
                                 )
    return dataloader

