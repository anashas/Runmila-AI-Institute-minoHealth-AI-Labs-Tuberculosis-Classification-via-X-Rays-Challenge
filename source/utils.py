import torch
import pandas as pd
import numpy as np
import os
from PIL import Image
import albumentations
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import pretrainedmodels
from torch.nn import functional as F
from torch.cuda import amp
from PIL import ImageFile
from sklearn import metrics

ImageFile.LOAD_TRUNCATED_IMAGES = True
# Build Custom Dataset

class TB_Dataset(Dataset):
    def __init__(self,image_path,data,targets,transform=None):
        self.image_path = image_path
        self.data = data
        self.targets = targets
        self.image_id = self.data['ID'].unique()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self,item):
        img = self.image_id[item]
        image = Image.open(f"{self.image_path}/{img}")
        image = image.convert("RGB")
        image = np.array(image)
        if self.transform is not None:
          image = self.transform(image=image)["image"]
        image = np.transpose(image,(2,0,1))
        return {
            'x': torch.tensor(image,dtype=torch.float),
            'y': torch.tensor(self.targets[item],dtype=torch.float)
            }
    
class TB_Test_Dataset(Dataset):
    def __init__(self,image_path,data,transform=None):
        self.image_path = image_path
        self.data = data
        self.image_id = self.data['ID'].unique()
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self,item):
        img = self.image_id[item]
        image = Image.open(f"{self.image_path}/{img}")
        image = image.convert("RGB")
        image = np.array(image)
        if self.transform is not None:
          image = self.transform(image=image)["image"]
          
        image = np.transpose(image,(2,0,1))
        return {
            'x': torch.tensor(image,dtype=torch.float),
            'img':img
            }      

# Build Engine 

class Engine:
    def __init__(self,model,optimizer,device):
        self.model =model
        self.optimizer =optimizer
        self.device = device
    def loss_fn(self,outputs,targets):
        return nn.BCEWithLogitsLoss()(outputs,targets)

    def acc_fn(self,outputs,targets):
        return metrics.roc_auc_score(outputs.detach().cpu().numpy(),targets.detach().cpu().numpy()) 

    def train(self,data_loader,scaler):
        self.model.train()
        final_loss = 0
        final_targets = []
        final_outputs = []
        for data in data_loader:
            self.optimizer.zero_grad()
            with amp.autocast(): 
                inputs = data['x'].to(self.device,dtype=torch.float)
                targets = data['y'].to(self.device,dtype=torch.float)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs,targets)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update( )
            final_loss += loss.item()
            targets = targets.detach().cpu().numpy().tolist()
            outputs = outputs.detach().cpu().numpy().tolist()

            final_targets.extend(targets)
            final_outputs.extend(outputs)
        return final_loss/len(data_loader), final_outputs,final_targets

    def validate(self,data_loader):
        self.model.eval()
        final_loss = 0
        final_targets = []
        final_outputs = []
        for data in data_loader:
            inputs = data['x'].to(self.device,dtype=torch.float)        
            targets = data['y'].to(self.device,dtype=torch.float)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs,targets)
            final_loss += loss.item()

            targets = targets.detach().cpu().numpy().tolist()
            outputs = outputs.detach().cpu().numpy().tolist()

            final_targets.extend(targets)
            final_outputs.extend(outputs)
        return final_loss/len(data_loader),final_outputs, final_targets

# Define Model

class resnet152(nn.Module):
    def __init__(self,pretrained):
        super(resnet152,self).__init__()
        if pretrained is True:
          self.model = pretrainedmodels.__dict__['resnet152'](pretrained="imagenet")
        else:
          self.model = pretrainedmodels.__dict__['resnet152'](pretrained=None)

        self.last_linear = nn.Linear(2048, 1)

    def forward(self, x):
        bsize, _ , _ , _ = x.shape  
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(bsize,-1)
        fn = self.last_linear(x)
        return fn    
        
