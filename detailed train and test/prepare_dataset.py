# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:08:04 2023

@author: FaresBougourzi
"""

import os
import numpy as np
import cv2
import nibabel as nib
from sklearn.model_selection import train_test_split

# The datasets could be donwloaded from: http://medicalsegmentation.com/covid19/

import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


database_path = './data/9 CT scans/rp_im'
database_pathh = './data/9 CT scans/'

# split the data into train and val
indicies = list(range(len(os.listdir(database_path))))
train_spt, val_spt = train_test_split(indicies, test_size=0.3) 

# train_spt = [0, 4, 5, 6, 7, 8]
# val_spt = [1, 2, 3]

# train_idx
# Out[2]: [0, 4, 5, 6, 7, 8]
# train_name
# Out[5]: ['1.nii.gz', '5.nii.gz', '6.nii.gz', '7.nii.gz', '8.nii.gz', '9.nii.gz']

# val_idx
# Out[3]: [1, 2, 3]
# val_name
# Out[4]: ['2.nii.gz', '3.nii.gz', '4.nii.gz']


#####################################

Ct_scans = sorted_alphanumeric(os.listdir(database_path))
sub = -1

Training_data = []
Validation_data = []

Training_lung = []
Validation_lung = []

Training_inf = []
Validation_inf = []
train_idx = []
train_name = []
val_idx = []
val_name = []
idx = -1

for Ct_scan in Ct_scans:
    idx += 1
    if idx in train_spt:
        train_idx.append(idx)
        train_name.append(Ct_scan)
    
        data_name = 'rp_im'
        slice_samples = os.path.join(database_pathh, data_name, Ct_scan)
        
        mask_name = 'rp_lung_msk'
        mask_samples = os.path.join(database_pathh, mask_name, Ct_scan)
        
        inf_name = 'rp_msk'
        inf_samples = os.path.join(database_pathh, inf_name, Ct_scan)   
        
        slices = nib.load(slice_samples)
        masks = nib.load(mask_samples)
        infs = nib.load(inf_samples)    
            
        slices = slices.get_fdata()
        masks = masks.get_fdata()
        infs = infs.get_fdata()
        sub += 1
    
        sub_lung_area = 0
        sub_inf_area = 0
        for i in range(infs.shape[2]):        
           
            slice1 = slices[:,:,i]
            slice1 = cv2.rotate(slice1, cv2.ROTATE_90_CLOCKWISE)
            mask1 = masks[:,:,i]
            mask1 = np.uint8(cv2.rotate(mask1, cv2.ROTATE_90_CLOCKWISE))
            inf1 = infs[:,:,i]
            inf1 =  np.uint8(cv2.rotate(inf1, cv2.ROTATE_90_CLOCKWISE))
            
            img = cv2.normalize(src=slice1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            Training_data.append(img)
            Training_lung.append(mask1)
            Training_inf.append(inf1)
        
    else:
        val_idx.append(idx)
        val_name.append(Ct_scan)        
    
        data_name = 'rp_im'
        slice_samples = os.path.join(database_pathh, data_name, Ct_scan)

        mask_name = 'rp_lung_msk'
        mask_samples = os.path.join(database_pathh, mask_name, Ct_scan)
        
        inf_name = 'rp_msk'
        inf_samples = os.path.join(database_pathh, inf_name, Ct_scan)   
        
        slices = nib.load(slice_samples)
        masks = nib.load(mask_samples)
        infs = nib.load(inf_samples)    
        
    
        slices = slices.get_fdata()
        masks = masks.get_fdata()
        infs = infs.get_fdata()
        sub += 1
    
        sub_lung_area = 0
        sub_inf_area = 0
        for i in range(infs.shape[2]):        
           
            slice1 = slices[:,:,i]
            slice1 = cv2.rotate(slice1, cv2.ROTATE_90_CLOCKWISE)
            mask1 = masks[:,:,i]
            mask1 = np.uint8(cv2.rotate(mask1, cv2.ROTATE_90_CLOCKWISE))
            inf1 = infs[:,:,i]
            inf1 =  np.uint8(cv2.rotate(inf1, cv2.ROTATE_90_CLOCKWISE))
            
            img = cv2.normalize(src=slice1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            Validation_data.append(img)
            Validation_lung.append(mask1)
            Validation_inf.append(inf1)        
 

import torch
X = [i for i in Training_data]
y = [i for i in Training_lung] 
y2 = [i for i in Training_inf]
training= (X, y, y2)
torch.save(training,'Training_data.pt') 


import torch
X = [i for i in Validation_data]
y = [i for i in Validation_lung] 
y2 = [i for i in Validation_inf]
training= (X, y, y2)
torch.save(training,'Validation_data.pt') 
