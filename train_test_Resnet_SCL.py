# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 22:19:35 2022

@author: FaresBougourzi
"""


# Bougourzi Fares
#from Letters_data_loader import Letter_loader,  Letter_loader2

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision


import torchvision.transforms as transforms
import numpy as np
import scipy.io as sio

from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import os
from sklearn.metrics import f1_score, balanced_accuracy_score


from torchvision.datasets import MNIST



class Letter_loader(MNIST):

    def __init__(self, root, train, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set
        self.data, self.y1, self.y2 = torch.load(os.path.join(self.root, self.train))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, y1, y2 = self.data[index], self.y1[index], self.y2[index]        

        img1 = np.array(img)
        # img1 = img.transpose((1, 2, 0))       
        
        img1 = img1.astype(np.uint8)        
              
        if self.transform is not None:
            img1 = self.transform(img1)
            
        if self.target_transform is not None:
            y1 = self.target_transform(y1)
            
        return   img1, y1, y2

    def __len__(self):
        return len(self.data)
    
#########################################
    
class Letter_loader2(MNIST):

    def __init__(self, root, train, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set
        self.data, self.y1, self.y2 = torch.load(os.path.join(self.root, self.train))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, y1, y2 = self.data[index], self.y1[index], self.y2[index]        

        img1 = np.array(img)
        # img1 = img.transpose((1, 2, 0))       
        
        img1 = img1.astype(np.uint8)        
              
        if self.transform is not None:
            img1 = self.transform(img1)
            
        if self.target_transform is not None:
            y1 = self.target_transform(y1)
            
        return   img1, y1, y2

    def __len__(self):
        return len(self.data)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

##########################################


train_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize((224,224)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),        
        transforms.RandomRotation(degrees = (-10,10)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),      
])    

test_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),          
]) 

torch.set_printoptions(linewidth=120)

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
    
train_set = Letter_loader2(
        root='./'
        ,train = 'Training_Ch.pt'
        ,transform = TwoCropTransform(train_transform)
)
       

model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, 128)            



if not os.path.exists('./ModelsCh/'):
    os.makedirs('./ModelsCh/')  


device = torch.device("cuda:0") 
model = model.to(device)     
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 128, shuffle = True)
criterion = SupConLoss()

nameSCL = './ModelsCh/Resnet50SCL.pt'

LR = 0.0001
lossSCL = []
##################################   20 0.0001  ############################################################    
for epoch in range(100):
    print(epoch)
    lr = LR
    if epoch > 30:
        lr = LR / 2
    if epoch > 50:
        lr = LR / 2 / 2
    if epoch > 70:
        lr = LR / 2 / 2 / 2
    if epoch > 80:
        lr = LR / 2 / 2 / 2 / 5
    if epoch > 90:
        lr = LR / 2 / 2 / 2 / 5 / 5          
        
    optimizer = optim.Adam(model.parameters(), lr = lr)    
      
    
    for batch in tqdm(train_loader):        
        images, labels1, labels2 = batch
        
        # images2 = images2.float().to(device)
        labels1 = labels1.long().to(device)
        images = torch.cat([images[0], images[1]], dim=0)
        images = images.float().to(device)
        torch.set_grad_enabled(True)
        model.train()
        bsz = labels1.shape[0]
        features = F.normalize(model(images), dim=1)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(features, labels1) 
        
        lossSCL.append(np.array(loss.cpu().detach()))              
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()                              

        del images; del labels1; del labels2
      
    torch.save(model.state_dict(), nameSCL)
    
    
    
    
    
    
    
    
    
    
    


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 13:09:52 2021

@author: bougourzi
"""



train_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize((224,224)),        
        transforms.RandomRotation(degrees = (-10,10)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),      
])    

test_transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize((224,224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),          
]) 

torch.set_printoptions(linewidth=120)

train_set = Letter_loader(
        root='./'
        ,train = 'Training_Ch.pt'
        ,transform = train_transform
)

validate_set = Letter_loader(
        root='./'
        ,train = 'Validation_Ch.pt'
        ,transform = test_transform
)

test_set = Letter_loader(
        root='./'
        ,train = 'Testing_Ch.pt'
        ,transform = test_transform
)

#######
class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, gamma=3.5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        pt = torch.exp(-BCE_loss)
        F_loss = (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
    
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()     


#

# model = torchvision.models.resnet50(pretrained=True) 
# model.fc = nn.Linear(2048, 624)
model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 624)
)

for param in model.parameters():
    param.requires_grad = False 
    
for param in model.fc.parameters():
    param.requires_grad = True 
# nn.BatchNorm1d(1024)
torch.set_grad_enabled(True)
torch.set_printoptions(linewidth=120)    


def MAE_distance(preds, labels):
    return torch.sum(torch.abs(preds - labels))

def Adaptive_loss(preds, labels, sigma):
    mse = (1+sigma)*((preds - labels)**2)
    mae = sigma + (torch.abs(preds - labels))
    return torch.mean(mse/mae)

def PC_mine(preds, labels):
    dem = np.sum((preds - np.mean(preds))*(labels - np.mean(labels)))
    mina = (np.sqrt(np.sum((preds - np.mean(preds))**2)))*(np.sqrt(np.sum((labels - np.mean(labels))**2)))
    return dem/mina 


if not os.path.exists('./ModelsCh/'):
    os.makedirs('./ModelsCh/')  


device = torch.device("cuda:0") 
model = model.to(device)     
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 16, shuffle = True)
validate_loader = torch.utils.data.DataLoader(validate_set, batch_size = 64)       
test_loader  = torch.utils.data.DataLoader(test_set, batch_size = 64)
criterion = FocalLoss()
sigma = 2

epoch_count = []
Accracy_tr = []
Accracy_vl = []
Accracy_ts = []

AccracyRA_tr = []
AccracyRA_vl = []
AccracyRA_ts = []

MiF1_tr = []
MiF1_vl = []
MiF1_ts = []

MaF1_tr = []
MaF1_vl = []
MaF1_ts = []

WeF1_tr = []
WeF1_vl = []
WeF1_ts = []

label_tr = []
pred_tr = []

Acc_best = -2
name = './ModelsCh/Resnet50_SCL_focal_best.pt'
nameR = './ModelsCh/Re_Resnet50_SCL_focal.pt'

LR = 0.0001

##################################   20 0.0001  ############################################################    
for epoch in range(100):
    epoch_count.append(epoch)
    lr = LR
    if epoch > 30:
        lr = LR / 2
    if epoch > 50:
        lr = LR / 2 / 2
    if epoch > 70:
        lr = LR / 2 / 2 / 2
    if epoch > 80:
        lr = LR / 2 / 2 / 2 / 5
    if epoch > 90:
        lr = LR / 2 / 2 / 2 / 5 / 5          
        
    optimizer = optim.Adam(model.parameters(), lr = lr)
    train_loss = 0
    validation_loss = 0
    total_correct_tr = 0
    total_correct_val = 0
    total_correct_ts = 0    
     
    label_f1tr = []
    pred_f1tr = []  
    
    for batch in tqdm(train_loader):        
        images, labels1, labels2 = batch
        images = images.float().to(device)
        labels1 = labels1.long().to(device)
      
        torch.set_grad_enabled(True)
        model.train()
        preds = model(images)
        loss = criterion(preds, labels1)   
        
        label_f1tr.extend(labels1.cpu().numpy().tolist())
        pred_f1tr.extend(preds.argmax(dim=1).tolist())        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()            
             
        train_loss += loss.item()                   
        total_correct_tr +=  get_num_correct(preds, labels1)      

        del images; del labels1; del labels2

    label_f1vl = []
    pred_f1vl = []        

    for batch in tqdm(validate_loader): 
        images, labels1, labels2 = batch
        images = images.float().to(device)
        labels1 = labels1.long().to(device)
        
        model.eval()
        with torch.no_grad():
            preds = model(images)
            
        label_f1vl.extend(labels1.cpu().numpy().tolist())
        pred_f1vl.extend(preds.argmax(dim=1).tolist())              
            
        loss = criterion(preds, labels1)                 
        validation_loss += loss.item()                   
        total_correct_val +=  get_num_correct(preds, labels1)        
        del images; del labels1; del labels2 

    label_f1ts = []
    pred_f1ts = []
        
    for batch in tqdm(test_loader): 
        images, labels1, labels2 = batch
        images = images.float().to(device)
        labels1 = labels1.long().to(device)
        
        model.eval()
        with torch.no_grad():
            preds = model(images)
            
        label_f1ts.extend(labels1.cpu().numpy().tolist())
        pred_f1ts.extend(preds.argmax(dim=1).tolist())              
            
        loss = criterion(preds, labels1)                 
        validation_loss += loss.item()                   
        total_correct_ts +=  get_num_correct(preds, labels1)        
        del images; del labels1; del labels2         
   
    
    print('Ep: ', epoch, 'AC_tr: ', total_correct_tr/len(train_set), 'AC_vl: ', total_correct_val/len(validate_set), 'AC_ts: ', total_correct_ts/len(test_set), 'Loss_tr: ', train_loss/len(train_set), 'Loss_ts: ', validation_loss/len(validate_set))
    print('AcMa_tr: ', balanced_accuracy_score(label_f1tr, pred_f1tr), 'AcMa_vl: ', \
          balanced_accuracy_score(label_f1vl, pred_f1vl), 'AcMa_ts: ', balanced_accuracy_score(label_f1ts, pred_f1ts))
    print('MaF1_tr: ', f1_score(label_f1tr, pred_f1tr , average='macro'), 'MaF1_vl: ', \
          f1_score(label_f1vl, pred_f1vl , average='macro'), 'MaF1_ts: ', f1_score(label_f1ts, pred_f1ts , average='macro'))

    Accracy_tr.append(total_correct_tr/len(train_set))
    Accracy_vl.append(total_correct_val/len(validate_set))
    Accracy_ts.append(total_correct_ts/len(test_set))
    
    AccracyRA_tr.append(balanced_accuracy_score(label_f1tr, pred_f1tr))
    AccracyRA_vl.append(balanced_accuracy_score(label_f1vl, pred_f1vl))
    AccracyRA_ts.append(balanced_accuracy_score(label_f1ts, pred_f1ts))    

    
    MaF1_tr.append(f1_score(label_f1tr, pred_f1tr , average='macro'))
    MaF1_vl.append(f1_score(label_f1vl, pred_f1vl , average='macro'))
    MaF1_ts.append(f1_score(label_f1ts, pred_f1ts , average='macro'))
    
    WeF1_tr.append(f1_score(label_f1tr, pred_f1tr , average='weighted'))
    WeF1_vl.append(f1_score(label_f1vl, pred_f1vl , average='weighted'))
    WeF1_ts.append(f1_score(label_f1ts, pred_f1ts , average='weighted')) 
    
    
  
    Acc_best2 = f1_score(label_f1vl, pred_f1vl, average='macro')       
    
    if Acc_best2 >=Acc_best: 
        Acc_best = Acc_best2
        torch.save(model.state_dict(), name)
        
print("Accuracy") 
print(Accracy_vl[MaF1_vl.index(np.max(MaF1_vl))]) 
print(Accracy_ts[MaF1_vl.index(np.max(MaF1_vl))])

print("AccuracyRA") 
print(AccracyRA_vl[MaF1_vl.index(np.max(MaF1_vl))]) 
print(AccracyRA_ts[MaF1_vl.index(np.max(MaF1_vl))])        
        
print("Macro F1")                        
print(np.max(MaF1_vl)) 
print(MaF1_ts[MaF1_vl.index(np.max(MaF1_vl))])

print("Weighted F1")                        
print(WeF1_vl[MaF1_vl.index(np.max(MaF1_vl))]) 
print(WeF1_ts[MaF1_vl.index(np.max(MaF1_vl))])


l = (epoch_count, Accracy_vl, Accracy_ts, AccracyRA_vl, AccracyRA_ts, MaF1_vl, MaF1_ts, WeF1_vl, WeF1_ts)
torch.save(l, nameR)    
    
    
    
    
    
    
    
    
        
