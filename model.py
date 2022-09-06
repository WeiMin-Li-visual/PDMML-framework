# -*- coding: utf-8 -*-
# The authors: Fangyu Liu, Shizhong Yuan, Weimin Li, Qun Xu, Bin Sheng
import numpy
import torch
from torch import nn
from torchvision import transforms
import nibabel as nib
from patch_range import *

#Set of pre-filtered patch locations, which contains the number of the patch at the Nth position, and the discriminability rank of that position
select_location = {'location1': rank1, 'location2': rank2, ..., 'locationN': rankN}

#The DenseResBlock module designed in this work is adapted to 24 x 24 x 24 image patch input, and each block contains 3 convolutional layers
class DenseResBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(DenseResBlock, self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.layers1 = nn.Sequential(
            nn.Conv3d(in_channels=self.channels_in, out_channels=self.channels_in, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=self.channels_in)
        )
        self.relu1 = nn.ReLU()
        self.layers2 = nn.Sequential(
            nn.Conv3d(in_channels=3 * self.channels_in, out_channels=self.channels_in, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=self.channels_in)
        )
        self.relu2 = nn.ReLU()
        self.layers3 = nn.Sequential(
            nn.Conv3d(in_channels=4 * self.channels_in, out_channels=self.channels_out, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(num_features=self.channels_out)
        )
        self.relu3 = nn.ReLU()
        self.w1_1 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w1_2 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2_1 = torch.nn.Parameter(torch.tensor([0.333]), requires_grad=True)
        self.w2_2 = torch.nn.Parameter(torch.tensor([0.333]), requires_grad=True)
        self.w2_3 = torch.nn.Parameter(torch.tensor([0.333]), requires_grad=True)

    def forward(self, x):
        x1 = self.layers1(x)
        x11 = self.relu1(torch.add(self.w1_1 * x, self.w1_2 * x1))
        x111 = torch.cat((x, x1, x11), dim=1)
        x2 = self.layers2(x111)
        x22 = self.relu2(torch.add(torch.add(self.w2_1 * x, self.w2_2 * x1), self.w2_3 * x2))
        x222 = torch.cat((x, x1, x2, x22), dim=1)
        x3 = self.layers3(x222)
        x3 = self.relu3(x3)
        return x3


#The sub-network "SubNet_Alone" designed in this work adapts the 24 x 24 x 24 image patch input before modality fusion
class SubNet_Alone(nn.Module):
    def __init__(self):
        super(SubNet_Alone, self).__init__()
        self.layers = nn.Sequential(
            #initial layer
            torch.nn.Conv3d(1, 16, kernel_size=(3, 3, 3), stride=1, padding=0), 
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),
            #DenseResBlock
            DenseResBlock(16, 16),
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0),
            DenseResBlock(16, 16),
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0),
        )

    def forward(self, input):
        output = self.layers(input)
        return output

#The sub-network "SubNet_Fusion" designed in this work adapts the 24 x 24 x 24 image patch input after modal fusion
class SubNet_Fusion(nn.Module):
    def __init__(self):
        super(SubNet_Fusion, self).__init__()
        self.layers = nn.Sequential(
            #The fused multimodal features are followed by a convolution operation
            DenseResBlock(16, 32),
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, padding=0),
        )

    def forward(self, input):
        output = self.layers(input)
        return output

#The PDMML framework designed in this work is adapted to 24 x 24 x 24 image patch input, in which we perform multimodal feature map fusion at the patch level
class PDMML(nn.Module):
    def __init__(self):
        super(PDMML, self).__init__()
        self.sub_mri_model = nn.ModuleDict(
            {index: SubNet_Alone() for index in select_location.keys()}
        )
        self.sub_gm_model = nn.ModuleDict(
            {index: SubNet_Alone() for index in select_location.keys()}
        )
        self.sub_wm_model = nn.ModuleDict(
            {index: SubNet_Alone() for index in select_location.keys()}
        )
        self.sub_csf_model = nn.ModuleDict(
            {index: SubNet_Alone() for index in select_location.keys()}
        )
        self.sub_pet_model = nn.ModuleDict(
            {index: SubNet_Alone() for index in select_location.keys()}
        )
        self.sub_fusion_model = nn.ModuleDict(
            {index: SubNet_Fusion() for index in select_location.keys()}
        )
        self.lstm = nn.LSTM(input_size=len(select_location), hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.MLP = nn.Sequential(
            nn.Linear(6, 64),
            nn.Linear(64, 32)
        )
        self.flatten = nn.Flatten()
        self.merge = nn.Sequential(
            nn.Linear(32 * 2 * 128 + 32, 256),#N image patches + an MLP module
            nn.Linear(256, 32),
            nn.Linear(32, 3)
        )
        self.w1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.w4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.w5 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, mul_mri_patch, mul_gm_patch, mul_wm_patch, mul_csf_patch, mul_pet_patch, batch_of_demographic_information):
        mul_output = 0.0
        flag = False
        for index in select_location.keys():
            mri_out = self.sub_mri_model[index](mul_mri_patch[index])
            gm_out = self.sub_gm_model[index](mul_gm_patch[index])
            wm_out = self.sub_wm_model[index](mul_wm_patch[index])
            csf_out = self.sub_csf_model[index](mul_csf_patch[index])
            pet_out = self.sub_pet_model[index](mul_pet_patch[index])
            fusion_out = self.w1 * mri_out + self.w2 * gm_out + self.w3 * wm_out + self.w4 * csf_out + self.w5 * pet_out
            if flag == False:
                # (B, channel=32, D=1, H=1, W=1)
                temp_out = self.sub_fusion_model[index](fusion_out)
                temp_shape = temp_out.shape
                # reshape -> (B, channel=32, L=D*H*W=1)
                mul_output = temp_out.reshape(temp_shape[0], temp_shape[1], -1)
                flag = True
            else:
                # (B, channel=32, D=1, H=1, W=1)
                temp_out = self.sub_fusion_model[index](fusion_out)
                temp_shape = temp_out.shape
                # reshape -> (B, channel=32, L=D*H*W，which is len(select_location))
                temp_out = temp_out.reshape(temp_shape[0], temp_shape[1], -1)
                mul_output = torch.cat((mul_output, temp_out), 2)
        # (B, channel=32, L=D*H_out=2*128)
        mul_output, (_, _) = self.lstm(mul_output)
        mul_output = self.flatten(mul_output)
        mul_output = torch.cat((mul_output, self.MLP(batch_of_demographic_information)), 1)
        mul_output = self.merge(mul_output)
        return mul_output

