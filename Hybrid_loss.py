# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 02:46:42 2022

@author: FaresBougourzi
"""

import torch
import torch.nn as nn
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2


import cv2


kernel = np.ones((5,5),np.uint8)
criterion =nn.BCEWithLogitsLoss()
TFtotensor = ToTensorV2()

def Hybrid_loss(GT_Inf, GT_Lung, Pred_Inf, Pred_Lung):
    alpha = 1.0
    Ed = GT_Inf*255.0
    Ed = Ed.astype(np.uint8)
    Edge = cv2.morphologyEx(Ed, cv2.MORPH_GRADIENT, kernel)
    Edge[Edge > 0.0] = 1.0
    Edge = TFtotensor(Edge)    
        
    loss1 = criterion(GT_Inf, Pred_Inf)
    loss2 = criterion(GT_Lung, Pred_Lung)
    loss3 = criterion(GT_Inf*Edge, Edge)
    loss = 0.7*loss1 + 0.3*loss2+ alpha*loss3
    
    return loss
