"""Trainer for TSCAN."""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torchvision.transforms import RandomHorizontalFlip
import torch.nn as nn

from evaluation.metrics import calculate_metrics_ev
from neural_methods.model.NTSCAN import NTSCAN
from neural_methods.trainer.BaseTrainer import BaseTrainer

from tqdm import tqdm

import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags

import math

class NtscanTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.device_ids = config.DEVICE_IDS
        
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.frame_depth = config.MODEL.TSCAN.FRAME_DEPTH
        self.num_train_batches = len(data_loader["train"])
        self.batch_size = config.TRAIN.BATCH_SIZE
        
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        
        self.base_len = self.frame_depth * self.num_of_gpu

        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config 
        self.min_valid_loss = None
        self.best_epoch = 0
        self.img_size = config.TRAIN.DATA.PREPROCESS.H

        self.model = NTSCAN(frame_depth=self.frame_depth, img_size=self.img_size).to(self.device)
        print(self.model)
        self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        
        self.criterion = torch.nn.MSELoss()
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)

        # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        self.tran = RandomHorizontalFlip(p=0.5)
        
    
    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_losses = []
            valid_losses = []
            self.model.train()
            
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                
                #data, label, period, filename, chunk_id
                data, labels = batch[0].to(self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                
                #apply random flip
                data = self.tran(data)

                labels = labels.view(-1, 1)
                
                data = data[:(N * D) // self.base_len * self.base_len]
                labels = labels[:(N * D) // self.base_len * self.base_len]
                
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                
                pred_ppg = (pred_ppg - torch.mean(pred_ppg)) / torch.std(pred_ppg)  # normalize
                
                loss = self.criterion(pred_ppg, labels)
                loss.backward()
                
                self.optimizer.step()
                self.scheduler.step()
                
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')                    
                    running_loss = 0.0
                    
                train_losses.append(loss.item())
                tbar.set_postfix(loss=loss.item())

            #record train loss
            mean_training_losses.append(sum(train_losses)/len(train_losses))
            self.save_model(epoch)
            
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
            print("train_losses= ", mean_training_losses)
            print("valid_losses= ", mean_valid_losses)

        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
    
    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print("===Validating===")
        print("Testing validation")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data, labels = valid_batch[0].to(self.device), valid_batch[1].to(self.device)

                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)

                data = data[:(N * D) // self.base_len * self.base_len]
                labels = labels[:(N * D) // self.base_len * self.base_len]

                pred_ppg = self.model(data)

                pred_ppg = (pred_ppg - torch.mean(pred_ppg)) / torch.std(pred_ppg)  # normalize

                loss = self.criterion(pred_ppg, labels)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print("Model path: ", self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.device)
        self.model.eval()

        predictions = dict()
        labels = dict()
        
        with torch.no_grad():
            batch_s = 0
            running_loss = 0
            
            for _, batch in enumerate(data_loader['test']):
                batch_size = batch[0].shape[0] 
                data_test = batch[0].to(self.device)
                labels_test = batch[1].to(self.device)
                
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                data_test = data_test[:(N * D) // self.base_len * self.base_len]
                
                labels_test = labels_test.view(-1, 1)
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
                
                pred_ppg_test = self.model(data_test)
                loss = self.criterion(pred_ppg_test, labels_test)
                running_loss+=loss
                batch_s+=1
                
                for idx in range(batch_size):
                    subj_index = batch[2][idx]
                    sort_index = int(batch[3][idx])
                    
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                        
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        calculate_metrics_ev(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)