# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 02:36:14 2022
@author: FaresBougourzi
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


##### Double Convs
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
                
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )            

    def forward(self, x):
        return self.conv(x)
 
###### Attention Block
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
    
#### Unet #########################################################

class UNet(nn.Module):
    def __init__(self, input_channels=3, num_classes = 1, deep_supervision=False):
        super(UNet, self).__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])
        self.conv2_2 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv1_3 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv0_4 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output 
    


#### Unet++ #########################################################

class UNetplus(nn.Module):
    def __init__(self, input_channels=3, num_classes = 1, deep_supervision=False):
        super(UNetplus, self).__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv0_1 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2]*2+nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1]*3+nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0]*4+nb_filter[1], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

#### AttUNet #########################################################

class AttUNet(nn.Module):
    def __init__(self, input_channels=3, num_classes = 1, deep_supervision=False):
        super(AttUNet, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        # self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        nb_filter = [32, 64, 128, 256, 512]

        self.conv0_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])
        self.conv2_2 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv1_3 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv0_4 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])


        self.Att4 = Attention_block(F_g= nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[2])
        
        self.Att3 = Attention_block(F_g= nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[1])       

        self.Att2 = Attention_block(F_g= nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[0])
        
        self.Att1 = Attention_block(F_g= nb_filter[1], F_l=nb_filter[0], F_int= int(nb_filter[0]/2))       

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        
    def forward(self, input):
        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))                        
      
        # decoding + concat path        
        x3_1 = self.up(x4_0)
        x3_0 = self.Att4(g=x3_1, x=x3_0) 
        x3_1 = self.conv3_1(torch.cat((x3_0, x3_1),dim=1))
        

        x2_2 = self.up(x3_1)
        x2_0 = self.Att3(g=x2_2, x=x2_0) 
        x2_2 = self.conv2_2(torch.cat((x2_0, x2_2),dim=1)) 
        
        x1_3 = self.up(x2_2)
        x1_0 = self.Att2(g=x1_3, x=x1_0) 
        x1_3 = self.conv1_3(torch.cat((x1_0, x1_3),dim=1)) 

        x0_4 = self.up(x1_3)
        x0_0 = self.Att1(g=x0_4, x=x0_0) 
        x0_4 = self.conv0_4(torch.cat((x0_0, x0_4),dim=1))                
        
        output = self.final(x0_4)         

        return output


#### Pyramid Att-UNet #########################################################
   
class PAttUNet(nn.Module):
    def __init__(self, input_channels=3, num_classes = 1, deep_supervision=False):
        super(PAttUNet, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)                
        
        nb_filter = [32, 64, 128, 256, 512]
        self.nb_filter = nb_filter

        self.conv0_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0]*2, nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1]*2, nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2]*2, nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3]*2, nb_filter[4])
        

        self.conv11_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv12_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv13_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv14_0 = DoubleConv(input_channels, nb_filter[0])  
        
        self.conv22_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv23_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv24_0 = DoubleConv(nb_filter[0], nb_filter[1]) 

        self.conv33_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv34_0 = DoubleConv(nb_filter[1], nb_filter[2]) 

        self.conv44_0 = DoubleConv(nb_filter[2], nb_filter[3])
        
        self.Attdw1 = Attention_block(F_g= nb_filter[0], F_l=nb_filter[0], F_int= int(nb_filter[0]/2))
        self.Attdw2 = Attention_block(F_g= nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Attdw3 = Attention_block(F_g= nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])   
        self.Attdw4 = Attention_block(F_g= nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[2])        
                      
        self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])
        self.conv2_2 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv1_3 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv0_4 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])

        self.Att4 = Attention_block(F_g= nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[2])        
        self.Att3 = Attention_block(F_g= nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[1])       
        self.Att2 = Attention_block(F_g= nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[0])        
        self.Att1 = Attention_block(F_g= nb_filter[1], F_l=nb_filter[0], F_int= int(nb_filter[0]/2)) 
        

        
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        
    def forward(self, input):
        # Images
        image_size = input.shape
        Images = []
        divsize = [2,4,8,16]
        for i in range(len(self.nb_filter)-1):
            Images.append(TF.resize(input, size=[int(image_size[2]/divsize[i]) , int(image_size[3]/divsize[i])]))
        
        # encoding path
        x0_0 = self.conv0_0(input)
        
        x11_0 = self.conv11_0(Images[0])
        x1_0att = self.Attdw1(g=x11_0, x=self.pool(x0_0))        
        x1_0 = self.conv1_0(torch.cat((x1_0att, self.pool(x0_0)),dim=1))

        x12_0 = self.conv12_0(Images[1])
        x22_0 = self.conv22_0(x12_0)         
        x2_0att = self.Attdw2(g=x22_0, x=self.pool(x1_0))  
        x2_0 = self.conv2_0(torch.cat((x2_0att, self.pool(x1_0)),dim=1))        
        
        
        x13_0 = self.conv13_0(Images[2])
        x23_0 = self.conv23_0(x13_0)        
        x33_0 = self.conv33_0(x23_0) 
        x3_0att = self.Attdw3(g=x33_0, x=self.pool(x2_0))        
        x3_0 = self.conv3_0(torch.cat((x3_0att, self.pool(x2_0)),dim=1))
        
        x14_0 = self.conv14_0(Images[3])  
        x24_0 = self.conv24_0(x14_0)         
        x34_0 = self.conv34_0(x24_0)
        x44_0 = self.conv44_0(x34_0)
        x4_0att = self.Attdw4(g=x44_0, x=self.pool(x3_0))
        x4_0 = self.conv4_0(torch.cat((x4_0att, self.pool(x3_0)),dim=1)) 
       
                             
      
        # decoding + concat path        
        x3_1 = self.up(x4_0)
        x3_0 = self.Att4(g=x3_1, x=x3_0) 
        x3_1 = self.conv3_1(torch.cat((x3_0, x3_1),dim=1))
        

        x2_2 = self.up(x3_1)
        x2_0 = self.Att3(g=x2_2, x=x2_0) 
        x2_2 = self.conv2_2(torch.cat((x2_0, x2_2),dim=1)) 
        
        x1_3 = self.up(x2_2)
        x1_0 = self.Att2(g=x1_3, x=x1_0) 
        x1_3 = self.conv1_3(torch.cat((x1_0, x1_3),dim=1)) 

        x0_4 = self.up(x1_3)
        x0_0 = self.Att1(g=x0_4, x=x0_0) 
        x0_4 = self.conv0_4(torch.cat((x0_0, x0_4),dim=1))                
        
        output = self.final(x0_4)         

        return output 

#### Dual-Decoder Att-UNet #########################################################
class DAttUNet(nn.Module):
    def __init__(self, input_channels=3, num_classes = 1, deep_supervision=False):
        super(DAttUNet, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        nb_filter = [32, 64, 128, 256, 512]

        self.conv0_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])
        self.conv2_2 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv1_3 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv0_4 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])
        
        self.conv3_1_2 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])
        self.conv2_2_2 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv1_3_2 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv0_4_2 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])        


        self.Att4 = Attention_block(F_g= nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[2])        
        self.Att3 = Attention_block(F_g= nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[1])       
        self.Att2 = Attention_block(F_g= nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[0])        
        self.Att1 = Attention_block(F_g= nb_filter[1], F_l=nb_filter[0], F_int= int(nb_filter[0]/2))       
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.Att4_2 = Attention_block(F_g= nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[2])        
        self.Att3_2 = Attention_block(F_g= nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[1])       
        self.Att2_2 = Attention_block(F_g= nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[0])       
        self.Att1_2 = Attention_block(F_g= nb_filter[1], F_l=nb_filter[0], F_int= int(nb_filter[0]/2))       
        self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1) 
        
    def forward(self, input):
        # encoding path
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))                        
      
        # decoding + concat path 
        # Att1
        x3_1 = self.up(x4_0)
        x3_0_1 = self.Att4(g=x3_1, x=x3_0) 
        x3_1 = self.conv3_1(torch.cat((x3_0_1, x3_1),dim=1))
        
        x3_1_2 = self.up(x4_0)
        x3_0_2 = self.Att4_2(g=x3_1_2, x=x3_0) 
        x3_1_2 = self.conv3_1_2(torch.cat((x3_0_2, x3_1_2),dim=1))

        # Att2
        x2_2 = self.up(x3_1)
        x2_0_1 = self.Att3(g=x2_2, x=x2_0) 
        x2_2 = self.conv2_2(torch.cat((x2_0_1, x2_2),dim=1)) 

        x2_2_2 = self.up(x3_1_2)
        x2_0_2 = self.Att3_2(g=x2_2_2, x=x2_0) 
        x2_2_2 = self.conv2_2_2(torch.cat((x2_0_2, x2_2_2),dim=1)) 
 
        # Att3        
        x1_3 = self.up(x2_2)
        x1_0_1 = self.Att2(g=x1_3, x=x1_0) 
        x1_3 = self.conv1_3(torch.cat((x1_0_1, x1_3),dim=1)) 

        x1_3_2 = self.up(x2_2_2)
        x1_0_2 = self.Att2_2(g=x1_3_2, x=x1_0) 
        x1_3_2 = self.conv1_3_2(torch.cat((x1_0_2, x1_3_2),dim=1)) 

        # Att4
        x0_4 = self.up(x1_3)
        x0_0_1 = self.Att1(g=x0_4, x=x0_0) 
        x0_4 = self.conv0_4(torch.cat((x0_0_1, x0_4),dim=1))                

        x0_4_2 = self.up(x1_3_2)
        x0_0_2 = self.Att1_2(g=x0_4_2, x=x0_0) 
        x0_4_2 = self.conv0_4_2(torch.cat((x0_0_2, x0_4_2),dim=1)) 
        
        output = self.final(x0_4)  
        output2 = self.final2(x0_4_2)         

        return output, output2    
    
#### Pyramid Dual-Decoder Att-UNet #########################################################
class PYAttUNet(nn.Module):
    def __init__(self, input_channels=3, num_classes = 1, deep_supervision=False):
        super(PYAttUNet, self).__init__()
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        # self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        
        nb_filter = [32, 64, 128, 256, 512]
        self.nb_filter = nb_filter
        # Encoder
        self.conv0_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0]*2, nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1]*2, nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2]*2, nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3]*2, nb_filter[4])
        

        self.conv11_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv12_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv13_0 = DoubleConv(input_channels, nb_filter[0])
        self.conv14_0 = DoubleConv(input_channels, nb_filter[0])  
        
        self.conv22_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv23_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv24_0 = DoubleConv(nb_filter[0], nb_filter[1]) 

        self.conv33_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv34_0 = DoubleConv(nb_filter[1], nb_filter[2]) 

        self.conv44_0 = DoubleConv(nb_filter[2], nb_filter[3])
        
        self.Attdw1 = Attention_block(F_g= nb_filter[0], F_l=nb_filter[0], F_int= int(nb_filter[0]/2))
        self.Attdw2 = Attention_block(F_g= nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Attdw3 = Attention_block(F_g= nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])   
        self.Attdw4 = Attention_block(F_g= nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[2])        

        
        # Decoder
        self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])
        self.conv2_2 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv1_3 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv0_4 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])
        
        self.conv3_1_2 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])
        self.conv2_2_2 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv1_3_2 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv0_4_2 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])        


        self.Att4 = Attention_block(F_g= nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[2])        
        self.Att3 = Attention_block(F_g= nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[1])       
        self.Att2 = Attention_block(F_g= nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[0])        
        self.Att1 = Attention_block(F_g= nb_filter[1], F_l=nb_filter[0], F_int= int(nb_filter[0]/2))       
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self.Att4_2 = Attention_block(F_g= nb_filter[4], F_l=nb_filter[3], F_int=nb_filter[2])        
        self.Att3_2 = Attention_block(F_g= nb_filter[3], F_l=nb_filter[2], F_int=nb_filter[1])       
        self.Att2_2 = Attention_block(F_g= nb_filter[2], F_l=nb_filter[1], F_int=nb_filter[0])       
        self.Att1_2 = Attention_block(F_g= nb_filter[1], F_l=nb_filter[0], F_int= int(nb_filter[0]/2))       
        self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1) 
        
    def forward(self, input):

        # Images
        image_size = input.shape
        Images = []
        divsize = [2,4,8,16]
        for i in range(len(self.nb_filter)-1):
            Images.append(TF.resize(input, size=[int(image_size[2]/divsize[i]) , int(image_size[3]/divsize[i])]))
        
        # encoding path
        x0_0 = self.conv0_0(input)
        
        x11_0 = self.conv11_0(Images[0])
        x1_0att = self.Attdw1(g=x11_0, x=self.pool(x0_0))        
        x1_0 = self.conv1_0(torch.cat((x1_0att, self.pool(x0_0)),dim=1))

        x12_0 = self.conv12_0(Images[1])
        x22_0 = self.conv22_0(x12_0)         
        x2_0att = self.Attdw2(g=x22_0, x=self.pool(x1_0))  
        x2_0 = self.conv2_0(torch.cat((x2_0att, self.pool(x1_0)),dim=1))        
        
        
        x13_0 = self.conv13_0(Images[2])
        x23_0 = self.conv23_0(x13_0)        
        x33_0 = self.conv33_0(x23_0) 
        x3_0att = self.Attdw3(g=x33_0, x=self.pool(x2_0))        
        x3_0 = self.conv3_0(torch.cat((x3_0att, self.pool(x2_0)),dim=1))
        
        x14_0 = self.conv14_0(Images[3])  
        x24_0 = self.conv24_0(x14_0)         
        x34_0 = self.conv34_0(x24_0)
        x44_0 = self.conv44_0(x34_0)
        x4_0att = self.Attdw4(g=x44_0, x=self.pool(x3_0))
        x4_0 = self.conv4_0(torch.cat((x4_0att, self.pool(x3_0)),dim=1)) 

                       
      
        # decoding + concat path 
        # Att1
        x3_1 = self.up(x4_0)
        x3_0_1 = self.Att4(g=x3_1, x=x3_0) 
        x3_1 = self.conv3_1(torch.cat((x3_0_1, x3_1),dim=1))
        
        x3_1_2 = self.up(x4_0)
        x3_0_2 = self.Att4_2(g=x3_1_2, x=x3_0) 
        x3_1_2 = self.conv3_1_2(torch.cat((x3_0_2, x3_1_2),dim=1))

        # Att2
        x2_2 = self.up(x3_1)
        x2_0_1 = self.Att3(g=x2_2, x=x2_0) 
        x2_2 = self.conv2_2(torch.cat((x2_0_1, x2_2),dim=1)) 

        x2_2_2 = self.up(x3_1_2)
        x2_0_2 = self.Att3_2(g=x2_2_2, x=x2_0) 
        x2_2_2 = self.conv2_2_2(torch.cat((x2_0_2, x2_2_2),dim=1)) 
 
        # Att3        
        x1_3 = self.up(x2_2)
        x1_0_1 = self.Att2(g=x1_3, x=x1_0) 
        x1_3 = self.conv1_3(torch.cat((x1_0_1, x1_3),dim=1)) 

        x1_3_2 = self.up(x2_2_2)
        x1_0_2 = self.Att2_2(g=x1_3_2, x=x1_0) 
        x1_3_2 = self.conv1_3_2(torch.cat((x1_0_2, x1_3_2),dim=1)) 

        # Att4
        x0_4 = self.up(x1_3)
        x0_0_1 = self.Att1(g=x0_4, x=x0_0) 
        x0_4 = self.conv0_4(torch.cat((x0_0_1, x0_4),dim=1))                

        x0_4_2 = self.up(x1_3_2)
        x0_0_2 = self.Att1_2(g=x0_4_2, x=x0_0) 
        x0_4_2 = self.conv0_4_2(torch.cat((x0_0_2, x0_4_2),dim=1)) 
        
        output = self.final(x0_4)  
        output2 = self.final2(x0_4_2)         

        return output, output2 
