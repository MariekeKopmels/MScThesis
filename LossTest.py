from os import listdir
import torch.nn as nn
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch import optim

import torch.nn.functional as F
import numpy as np
import os
import cv2
import torch

LR = 0.001
MOMENTUM = 0.9
NO_PIXELS = 224
NUM_EXAMPLES = 1
EXAMPLE = 0
# BATCH_SIZE = 2

def save_image(filename, image, bw=False):
    # print("Dims recieved for printing image: ", image.shape)
    if bw:
        image = image*225
    if type(image) != np.ndarray:
        cv2.imwrite(filename, image.numpy())
    else:
        cv2.imwrite(filename, image)
        

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Compute binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calculate the modulating factor (focal term)
        pt = torch.exp(-bce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma) * bce_loss

        # Calculate the average loss
        return torch.mean(focal_loss)

def load_data():
    print("Loading data...")
    base_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/visuAAL"
    img_dir_path = base_path + "/TrainImages"
    gt_dir_path = base_path + "/TrainGroundTruth"
    
    image_list = listdir(img_dir_path)
    gt_list = listdir(gt_dir_path)
    
    dir_list = [file for file in image_list if file in gt_list]
    dir_list = dir_list[:NUM_EXAMPLES]

    images = []
    gts = []
    
    directory = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Output/LossTest/"
    os.chdir(directory) 
        
    for file_name in dir_list:
        if file_name.startswith('.'):
            continue
        
        # Read the images
        img_path = img_dir_path + "/" + file_name
        gt_path = gt_dir_path + "/" + file_name
                
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path)
        
        # Resize images and ground truths to size 224*224
        img = cv2.resize(img, (NO_PIXELS,NO_PIXELS), interpolation=cv2.INTER_CUBIC)
        gt = cv2.resize(gt, (NO_PIXELS,NO_PIXELS), interpolation=cv2.INTER_CUBIC)
                        
        save_image("TestGTbefore.jpg", gt)
        # Convert Ground Truth from RGB to 1 channel (Black or White)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)  
        ret,gt = cv2.threshold(gt,127,1,0)
        # count_zeros = (gt == 0.0).sum().item()
        # count_ones = (gt == 1.0).sum().item()
        # count_white = (gt == 255.0).sum().item()
        # values = np.unique(gt)
        # print("Shape gt: ", np.shape(gt))
        # print("Count of 0's:", count_zeros)
        # print("Count of 1's:", count_ones)
        # print("Count of 225's:", count_white)
        # print("Unique values:", values)
        save_image("TestGTafter.jpg", gt*225)

        #Store in list
        images.append(img)
        gts.append(gt)
    
    images = torch.tensor(np.array(images))
    gts = torch.tensor(np.array(gts))
    
    return images.permute(0,3,1,2), gts

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x
     
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))
    
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        
        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
    
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        """ Bottleneck """
        self.b = conv_block(512, 1024)
        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        """ Classifier """
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)
    
    def forward(self, inputs):
        # For the last few items, the batch size may be smaller than BATCH_SIZE
        batch_size = len(inputs)
        
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p4)
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        x = self.outputs(d4)
        # x = self.softmax(x)
        
        """ Reformatting"""
        outputs = x.clone().reshape(batch_size, NO_PIXELS, NO_PIXELS)
        # Not correct, should be a sigmoid
        binary_outputs = (outputs>=0.5).float()
        return binary_outputs


if __name__ == '__main__':
    model = UNET()
    # optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    images, gts = load_data()
    images = images.float()
    gts = gts.float()
    model_output = model(images)
    
    # image = [[28.0, 2.0, 211.0],[24.0, 91.0, 197.0],[61.0, 38.0, 273.0]]
    # gt = [0.0, 0.0, 1.0, 1.0, 0.0]
    # model_output = model(torch.tensor(image))
    
    # print("GT contents summary")
    # count_zeros = (gts[EXAMPLE] == 0.0).sum().item()
    # count_ones = (gts[EXAMPLE] == 1.0).sum().item()
    # print("Count of 0's:", count_zeros)
    # print("Count of 1's:", count_ones)
    
    # print("Model contents summary")
    # count_zeros = (model_output[EXAMPLE] == 0.0).sum().item()
    # count_ones = (model_output[EXAMPLE] == 1.0).sum().item()
    # print("Count of 0's:", count_zeros)
    # print("Count of 1's:", count_ones)
    
    image_perm = images[EXAMPLE].permute(1,2,0)
    # output_perm = model_output[EXAMPLE].permute(1,2,0)
    save_image("TestImage.jpg", image_perm)
    save_image("TestGT.jpg", gts[EXAMPLE], bw=True)
    save_image("TestOutput.jpg", model_output[EXAMPLE], bw=True)
    # save_image("TestOutput.jpg", model_output[EXAMPLE])
    
    zero_output = torch.zeros(NUM_EXAMPLES, 224, 224, requires_grad=True)
    one_output = torch.ones(NUM_EXAMPLES, 224, 224, requires_grad=True)
    
    gt = gts[EXAMPLE]
    gt = gt.float()
    gt = (gt>0).to(torch.float32)
    
    # print("zero_output shape:", zero_output.shape)
    # print("gt shape:", gt.shape)
    # print("model_output shape:", model_output.shape)
    
    # print(f"Zero output: {zero_output[20]}")
    # print(f"GT: {gt[20]}")
    # print(f"Model: {model_output[0][20][:][:]}")
    
    print("CE loss")
    CE_loss = nn.CrossEntropyLoss()
    zero_loss = CE_loss(zero_output, gts)
    one_loss = CE_loss(one_output, gts)
    model_loss = CE_loss(model_output, gts)
    print(f"Zero's loss: {zero_loss}")
    print(f"One's loss: {one_loss}")
    print(f"Model's loss: {model_loss}")
        
    print("BCE loss")
    BCE_loss = nn.BCELoss()
    zero_loss = BCE_loss(zero_output, gts)
    one_loss = BCE_loss(one_output, gts)
    model_loss = BCE_loss(model_output, gts)
    print(f"Zero's loss: {zero_loss}")
    print(f"One's loss: {one_loss}")
    print(f"Model's loss: {model_loss}")
        
    print("Focal loss")
    focal_loss = FocalLoss()
    zero_loss = focal_loss(zero_output, gts)
    one_loss = focal_loss(one_output, gts)
    model_loss = focal_loss(model_output, gts)
    print(f"Zero's loss: {zero_loss}")
    print(f"One's loss: {one_loss}")
    print(f"Model's loss: {model_loss}")
    
    