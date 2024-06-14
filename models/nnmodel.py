# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import logging
import copy
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import joblib
from torch.utils.data import Dataset
import numpy as np
from utils import logger
from .unimol import UniMolModel,Mamba,ModelArgs
from .loss import GHMC_Loss, FocalLossWithLogits, myCrossEntropyLoss
from  .weight import weighted_mse_loss
from tasks.fusion import MyModule
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import warnings
NNMODEL_REGISTER = {
    'unimolv1': UniMolModel,
}

LOSS_RREGISTER = {
    'classification': myCrossEntropyLoss,
    'multiclass': myCrossEntropyLoss,
    'regression': nn.MSELoss(),
    'multilabel_classification': {
        'bce': nn.BCEWithLogitsLoss(),
        'ghm': GHMC_Loss(bins=10, alpha=0.5),
        'focal': FocalLossWithLogits,
    },
    'multilabel_regression': nn.MSELoss(),
}
ACTIVATION_FN = {
    # predict prob shape should be (N, K), especially for binary classification, K equals to 1.
    'classification': lambda x: F.softmax(x, dim=-1)[:, 1:],
    # softmax is used for multiclass classification
    'multiclass': lambda x: F.softmax(x, dim=-1),
    'regression': lambda x: x,
    # sigmoid is used for multilabel classification
    'multilabel_classification': lambda x: F.sigmoid(x),
    # no activation function is used for multilabel regression
    'multilabel_regression': lambda x: x,
}
OUTPUT_DIM = {
    'classification': 2,
    'regression': 1,
}


class NNModel(object):
    def __init__(self, data, trainer, **params):
        self.data = data
        self.num_classes = self.data['num_classes']
        self.target_scaler = self.data['target_scaler']
        self.features = data['unimol_input']
        self.model_name = params.get('model_name', 'unimolv1')
        self.data_type = params.get('data_type', 'molecule')
        self.loss_key = params.get('loss_key', None)
        self.trainer = trainer
        self.splitter = self.trainer.splitter
        self.model_params = params.copy()
        self.task = params['task']
        if self.task in OUTPUT_DIM:
            self.model_params['output_dim'] = OUTPUT_DIM[self.task]
        elif self.task == 'multiclass':
            self.model_params['output_dim'] = self.data['multiclass_cnt']
        else:
            self.model_params['output_dim'] = self.num_classes
        self.model_params['device'] = self.trainer.device
        self.cv = dict()
        self.metrics = self.trainer.metrics
        if self.task == 'multilabel_classification':
            if self.loss_key is None:
                self.loss_key = 'focal'
            self.loss_func = LOSS_RREGISTER[self.task][self.loss_key]
        else:
            self.loss_func = LOSS_RREGISTER[self.task]
        self.activation_fn = ACTIVATION_FN[self.task]
        self.save_path = self.trainer.save_path
        self.trainer.set_seed(self.trainer.seed)
        self.model = self._init_model(**self.model_params)
        self.model_args = ModelArgs()  # 根据需要进行参数设置
        self.Mamba_model = Mamba(self.model_args)
        self.MyModule = MyModule(self.trainer.device)
    def _init_model(self, model_name, **params):
        if model_name in NNMODEL_REGISTER:
            model = NNMODEL_REGISTER[model_name](**params)
        else:
            raise ValueError('Unknown model: {}'.format(self.model_name))
        return model

    def collect_data(self, X, y, idx):
        assert isinstance(y, np.ndarray), 'y must be numpy array'
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X[idx]).float(), torch.from_numpy(y[idx])
        elif isinstance(X, list):
            return {k: v[idx] for k, v in X.items()}, torch.from_numpy(y[idx])
        else:
            raise ValueError('X must be numpy array or dict')

    def run(self):
        train_data=self.data['raw_data']
        data = train_data[['SMILES', 'TARGET']]
        tokenizer = AutoTokenizer.from_pretrained("/home/wk/3_paper/mamba_Unimol_loss/models/ChemBERTa-77M-MTR")
        #weights = [111111]
        logger.info("start training Uni-Mol:{}".format(self.model_name))
        X = np.asarray(self.features)
        y = np.asarray(self.data['target'])
        #weights= np.asarray(#).reshape(-1, 1)
        scaffold = np.asarray(self.data['scaffolds'])
        if self.task == 'classification':
            y_pred = np.zeros_like(
                y.reshape(y.shape[0], self.num_classes)).astype(float)
        else:
            y_pred = np.zeros((y.shape[0], self.model_params['output_dim']))

        for fold, (tr_idx, te_idx) in enumerate(self.splitter.split(X, y, scaffold)):
            Mamba_train_list = []
            Mamba_valid_list = []
            #X_train, y_train, weight = X[tr_idx], y[tr_idx], weights[tr_idx]
            X_train, y_train = X[tr_idx], y[tr_idx]
            #print(tr_idx)
            X_valid, y_valid = X[te_idx], y[te_idx]
            traindataset = NNDataset(X_train, y_train)
            # print("tr_idx",tr_idx)
            # print("te_idx",te_idx)
            for idx in tr_idx:
                train_smiles = data.iloc[idx]["SMILES"]
                Mamba_train = tokenizer(train_smiles, return_tensors="pt", padding='max_length', truncation=True, max_length=100)
                Mamba_train = Mamba_train["input_ids"].squeeze(0).tolist()
                Mamba_train_list.append(Mamba_train)

            for idx in te_idx:
                test_smiles = data.iloc[idx]["SMILES"]
                Mamba_test = tokenizer(test_smiles, return_tensors="pt", padding='max_length', truncation=True, max_length=100)
                Mamba_test = Mamba_test["input_ids"].squeeze(0).tolist()
                Mamba_valid_list.append(Mamba_test)

            Mamba_train_tensors = [torch.tensor(item) for item in Mamba_train_list]
            Mamba_valid_tensors = [torch.tensor(item) for item in Mamba_valid_list]
            Mamba_traindataset = NNDataset(Mamba_train_tensors, y_train)
            Mamba_validdataset = NNDataset(Mamba_valid_tensors, y_valid)
            
            #print(Mamba_traindataset.data)
            #scalar = self.data['target_scaler']
            #y_true = scalar.inverse_transform(traindataset.label)
            #import matplotlib.pyplot as plt
            #plt.hist(y_true, bins=20, edgecolor='black')  # 调整 bins 的数量以适应你的数据
            #plt.title('Histogram of y_train')
            #plt.xlabel('Label Values')
            #plt.ylabel('Frequency')
            #plt.savefig('histogram_1.png')  # 文件名可以根据需要进行调整

            validdataset = NNDataset(X_valid, y_valid)
            if fold > 0:
                # need to initalize model for next fold training
                self.model = self._init_model(**self.model_params)
                #model_args = ModelArgs()  # 根据需要进行参数设置
                #Mamba_model = Mamba(model_args)
            #_y_pred = self.trainer.fit_predict(
            #    self.model, traindataset, validdataset, self.loss_func, self.activation_fn, self.save_path, fold, self.target_scaler)
            _y_pred = self.trainer.fit_predict(
                self.model, traindataset, validdataset,self.Mamba_model, Mamba_traindataset, Mamba_validdataset, self.loss_func, self.activation_fn, self.save_path, fold, self.target_scaler)
            y_pred[te_idx] = _y_pred

            if 'multiclass_cnt' in self.data:
                label_cnt = self.data['multiclass_cnt']
            else:
                label_cnt = None

            logger.info("fold {0}, result {1}".format(
                fold,
                self.metrics.cal_metric(
                        self.data['target_scaler'].inverse_transform(y_valid),
                        self.data['target_scaler'].inverse_transform(_y_pred),
                        label_cnt=label_cnt
                        )
            )
            )
        
        self.cv['pred'] = y_pred
        self.cv['metric'] = self.metrics.cal_metric(self.data['target_scaler'].inverse_transform(
            y), self.data['target_scaler'].inverse_transform(self.cv['pred']))
        self.dump(self.cv['pred'], self.save_path, 'cv.data')
        self.dump(self.cv['metric'], self.save_path, 'metric.result')
        logger.info("Uni-Mol metrics score: \n{}".format(self.cv['metric']))
        logger.info("Uni-Mol & Metric result saved!")

    def dump(self, data, dir, name):
        path = os.path.join(dir, name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        joblib.dump(data, path)

    def evaluate(self, trainer=None,  checkpoints_path=None, visual=True):
        logger.info("start predict NNModel:{}".format(self.model_name))
        testdataset = NNDataset(self.features, np.asarray(self.data['target']))
        #train_data = pd.read_csv(data)
        train_data=self.data['raw_data']
        y = np.asarray(self.data['target'])
        data = train_data[['SMILES', 'TARGET']]
        tokenizer = AutoTokenizer.from_pretrained("/home/wk/3_paper/mamba_Unimol_loss/models/ChemBERTa-77M-MTR")
        Mamba_test_list = []
        tr_idx = range(len(data))
        for idx in tr_idx:
            test_smiles = data.iloc[idx]["SMILES"]
            Mamba_test = tokenizer(test_smiles, return_tensors="pt", padding='max_length', truncation=True, max_length=100)
            Mamba_test = Mamba_test["input_ids"].squeeze(0).tolist()
            Mamba_test_list.append(Mamba_test)

        Mamba_test_tensors = [torch.tensor(item) for item in Mamba_test_list]
        Mamba_testdataset = NNDataset(Mamba_test_tensors, y)
        for fold in range(self.splitter.n_splits):
            model_path = os.path.join(checkpoints_path, f'model_{fold}.pth')

            #self.Mamba_model.load_state_dict(torch.load(
            #    model_path, map_location=self.trainer.device)['model_state_dict'])

            info = torch.load(model_path, map_location=self.trainer.device)
            self.Mamba_model.load_state_dict(info['mamba_state_dict'])
            self.model.load_state_dict(info['Unimol_state_dict'])
            self.MyModule.load_state_dict(info['MyModule_state_dict'])
            #model.load_state_dict(model_dict)
            _y_pred, _, __,encodings,y_truths,attention_weights  = trainer.predict(self.MyModule,self.model, testdataset,self.Mamba_model, Mamba_testdataset, self.loss_func, self.activation_fn,
                            self.save_path, fold, self.target_scaler, epoch=1, load_model=True)
            if fold == 0:
                y_pred = np.zeros_like(_y_pred)
                weight_keshi = np.zeros_like(attention_weights)
            y_pred += _y_pred
            weight_keshi += attention_weights
        y_pred /= self.splitter.n_splits
        weight_keshi /= self.splitter.n_splits
        #print(weight_keshi)
        scalar = self.data['target_scaler']
        y_prediction = scalar.inverse_transform(y_pred)
        df = self.data['raw_data'].copy()
        self.cv['test_pred'] = y_pred
        if visual == True:
            return y_prediction, encodings,y_truths,weight_keshi

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def NNDataset(data, label=None):
    return TorchDataset(data, label)


class TorchDataset(Dataset):
    def __init__(self, data, label=None):
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))
        #self.weight = weight if weight is not None else np.zeros((len(data), 1))

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)
