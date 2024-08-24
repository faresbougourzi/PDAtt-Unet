#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 09:52:49 2022

@author: Fares Bougourzi
"""


# Bougourzi Fares
from Letters_data_loader import Letter_loader, Letter_loaderSia2
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
import time

from sklearn.metrics import f1_score, balanced_accuracy_score




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

# #######
class ContrastiveLossl2(torch.nn.Module):

      def __init__(self, margin=1.0):
            super(ContrastiveLossl2, self).__init__()
            self.margin = margin

      def forward(self, output1, output2, label):
            # Find the pairwise distance or eucledian distance of two output feature vectors
            euclidean_distance = F.pairwise_distance(output1, output2)
            # perform contrastive loss calculation with the distance
            loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

            return loss_contrastive


class ContrastiveLosscos(torch.nn.Module):

      def __init__(self, margin=1.0):
            super(ContrastiveLosscos, self).__init__()
            self.margin = margin
            self.cossim = nn.CosineSimilarity()
            self.pi = 3.1415927410125732/2

      def forward(self, output1, output2, label):
            # Find the pairwise distance or eucledian distance of two output feature vectors
            euclidean_distance = self.cossim(output1, output2)
            # perform contrastive loss calculation with the distance1 -
            loss_contrastive = torch.mean((1-label) * torch.cos(torch.clamp(euclidean_distance, min=0.0)*self.pi) +
            (label) * (self.margin -  torch.cos(torch.clamp(euclidean_distance, min=0.0)*self.pi)))

            return loss_contrastive


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

def get_num_correct2(preds, labels):
    correct = 0
    for i in range(preds.shape[0]):
        correct += np.array((preds[i] == labels[i].cpu())).sum().item()        
    return correct 

#######
from torch.nn import Parameter
device = torch.device("cuda:0") 
    
def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t())
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1,w2).clamp(min=eps)


class MarginCosineProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(MarginCosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, label, train):
        cosine = cosine_sim(inputs, self.weight)
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507

        if train:
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1), 1.0)
            # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------            
            output = self.s * (cosine - one_hot * self.m)
        else:
            output = self.s * cosine

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


Z1 = 1800
Z2 = 1500
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True) 
        self.model.fc =  nn.Linear(2048, 128)
 
        
        self.name = './ModelsCh/Resnet50SCL.pt'
        self.model.load_state_dict(torch.load(self.name))

        self.model.fc = nn.Identity()      
        self.fc1 = nn.Sequential(
              nn.Linear(2048, Z1),
              nn.ReLU(),
              nn.Linear(Z1, Z2),
              nn.ReLU(),              
              nn.Linear(Z2, 624)
       )
        
        self.fc2 = nn.Sequential(
              nn.Linear(2048, Z1),
              nn.ReLU(),
              nn.Linear(Z1, Z2),
              nn.ReLU(),              
              nn.Linear(Z2, 624)
       )        
        
        self.classifier =  MarginCosineProduct(624, 624) 


    def forward_once(self, x, label,Train):        
        output = self.model(x)
        output1 = self.fc1(output)
        output2 = self.fc2(output)
        output3 = self.classifier(output2, label,Train)
        return output1, output2, output3

    def forward(self, input1, input2, label1, label2, Train):
        output1, out1, ou1 = self.forward_once(input1, label1,Train)
        output2, out2, ou2 = self.forward_once(input2, label2,Train)
        return output1, output2, out1, out2, ou1, ou2


###################################

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
import os 
if not os.path.exists('./ModelsChSi/'):
    os.makedirs('./ModelsChSi/')     

name = './ModelsChSi/Siamese_Resnet50_3head_neg4_SCL_ContCosF_best.pt'
nameR = './ModelsChSi/Re_Siamese_Resnet50_3head_neg4_SCL_ContCosF.pt'

device = torch.device("cuda:0") 
model = SiameseNetwork().to(device)      
    
    
validate_loader = torch.utils.data.DataLoader(validate_set, batch_size = 512)       
test_loader  = torch.utils.data.DataLoader(test_set, batch_size = 512)
criterion = FocalLoss()
# criterion2 = FocalLoss(gamma=2)
criterioncos = ContrastiveLosscos()
criterionl2 = ContrastiveLossl2()


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
# softmax = nn.Softmax(dim=1) 0.8434


LR = 0.0001
start = time.time()

import random
import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


train_path = '/splits images/train'
fine_files = sorted_alphanumeric(os.listdir(train_path))
ts_path = '/' 
n_epochs = 50

sigma_max2 = 1
sigma_min2 = 0.5

sigma_max = 1
sigma_min = 0.1


##################################   20 0.0001  ############################################################    
for epoch in range(25):
    if epoch<1:
        start2 = time.time()    

##################################    
    img1 = []
    img2 = []
    label1 = []
    label2 = []
    lab_cont = []
    
    for fine in range(len(fine_files)):    
        fine_dir = os.path.join(train_path, fine_files[fine])
        images = sorted_alphanumeric(os.listdir(fine_dir))
        for i in range(len(images)):
            for j in range(i+1,len(images)):
                img1.append(os.path.join(train_path, fine_files[fine], images[i]))
                label1.append(str(fine_files[fine]))
                img2.append(os.path.join(train_path, fine_files[fine], images[j]))
                label2.append(str(fine_files[fine]))            
                lab_cont.append(0)
                
                
            fine_files_negative = list(range(len(fine_files)))
            fine_files_negative.remove(fine) 
            nesel = random.sample(range(len(fine_files_negative)), 4)
            for ll in range(len(nesel)):
                images2 = sorted_alphanumeric(os.listdir(os.path.join(train_path, fine_files[nesel[ll]])))
                # class_img_len = len(images2)
                sample_list = random.sample(range(len(images2)), 1)
                img1.append(os.path.join(train_path, fine_files[fine], images[i]))
                label1.append(fine_files[fine])
                
                img2.append(os.path.join(train_path, fine_files[nesel[ll]], images2[sample_list[0]]))
                label2.append(fine_files[nesel[ll]])            
                lab_cont.append(1)                
                  
    data = []                
    for i in range(len(img1)):                
        data.append([img1[i], img2[i], label1[i], label2[i], lab_cont[i]])
    
    ts_indxs = list(range(len(data))) 

    
    
    train_set = Letter_loaderSia2(
            list_IDs = ts_indxs, 
            path = ts_path,
            data = data,
            transform = train_transform
    )    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 32, shuffle = True)     
    
   
################################# 

    lr = LR
    if epoch > 9:
        lr = LR *0.1
    if epoch > 19:
        lr = LR *0.01  

      
        
    optimizer = optim.Adam(model.parameters(), lr = lr)
    train_loss = 0
    validation_loss = 0
    total_correct_tr = 0
    total_correct_val = 0
    total_correct_ts = 0 
    total_correct_ts2 = 0
    correctcs = 0.
    totalcs = 0.
    
    label_f1tr = []
    pred_f1tr = []   
    
    for batch in tqdm(train_loader):        
        images1, images2, labels1, labels2, lab_cont = batch
        # img1, img2, lab1, lab2, pairspone, lab_inter, lab_uni, coarselab1, coarselab2
        images1 = images1.float().to(device)
        labels1 = labels1.long().to(device)
        images2 = images2.float().to(device)
        labels2 = labels2.long().to(device)
        lab_cont = lab_cont.long().to(device)
        lab_inter = (1-lab_cont).long().to(device)
                       
        torch.set_grad_enabled(True)
        model.train()
        preds1, preds2, pred1, pred2, pre1, pre2 = model(images1, images2, labels1, labels2, True)
        
        loss1 = criterion(preds1, labels1)  
        loss2 = criterion(preds2, labels2)
        loss11 = criterion(pre1, labels1)  
        loss22 = criterion(pre2, labels2)        
        # loss3 = criterion2(feas1, lab_inter)
        lossl2 = criterionl2(preds1, preds2, lab_cont)
        losscos = criterioncos(pred1, pred2, lab_cont)

        loss = loss1 + loss2+ loss11 + loss22+ 0.3*lossl2+ 0.3*losscos

        predds =  F.softmax(preds1, dim =1)  +  F.softmax(pre1, dim =1)   
        label_f1tr.extend(labels1.cpu().numpy().tolist())
        pred_f1tr.extend(predds.argmax(dim=1).tolist())  
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()            
             
        train_loss += loss.item()                   
        total_correct_tr +=  get_num_correct(predds, labels1)

        del images1; del images2; del labels1; del labels2
        
        
    label_f1vl = []
    pred_f1vl = []        

    for batch in tqdm(validate_loader): 
        images, labels1, labels2 = batch
        images = images.float().to(device)
        labels1 = labels1.long().to(device)
        
        model.eval()
        with torch.no_grad():
            preds1, _, preds2 =  model.forward_once(images, None, False)
            
        predds =  F.softmax(preds1, dim =1)  +  F.softmax(preds2, dim =1)     
            
        label_f1vl.extend(labels1.cpu().numpy().tolist())
        pred_f1vl.extend(predds.argmax(dim=1).tolist())              
            
        loss = criterion(predds, labels1)                 
        validation_loss += loss.item()                   
        total_correct_val +=  get_num_correct(predds, labels1)        
        del images; del labels1; del labels2 

    label_f1ts = []
    pred_f1ts = []
        
    for batch in tqdm(test_loader): 
        images, labels1, labels2 = batch
        images = images.float().to(device)
        labels1 = labels1.long().to(device)
        
        model.eval()
        with torch.no_grad():
            preds1, _, preds2 =  model.forward_once(images, None, False)
            
        predds =  F.softmax(preds1, dim =1)  +  F.softmax(preds2, dim =1) 
            
        label_f1ts.extend(labels1.cpu().numpy().tolist())
        pred_f1ts.extend(predds.argmax(dim=1).tolist())              
            
        loss = criterion(predds, labels1)                 
        validation_loss += loss.item()                   
        total_correct_ts +=  get_num_correct(predds, labels1)        
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
        
    if epoch<1:
        # start2 = time.time()
        time_elapsed2 = time.time() - start2
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed2 // 60, time_elapsed2 % 60))   

time_elapsed = time.time() - start
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
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


