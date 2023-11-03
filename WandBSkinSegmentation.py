#MSc Thesis Marieke Kopmels
import os
import random
import time
from os import listdir
import numpy as np

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchmetrics.classification import Dice
from torch.utils.data import DataLoader

import wandb

NO_PIXELS = 224 

config = dict(
    BATCH_SIZE=32,
    NUM_EPOCHS=25,
    LR=0.01,
    MOMENTUM=0.99,
    TRAIN_SIZE=128,
    TEST_SIZE=32,
    dataset="VisuAAL",
    architecture="UNet", 
    device=torch.device("mps"))

"""Returns images in a given directory
        Format of return: list of NumPy arrays.
        NumPy arrays are of shape (224,224,3) for images and (224,224) for ground truths 
"""
def load_images(config, images, gts, image_dir_path, gt_dir_path, test=False):
    # Load list of files and directories
    image_list = listdir(image_dir_path)
    gt_list = listdir(gt_dir_path)
    
    # Not all images have a ground truth, select those that do
    dir_list = [file for file in image_list if file in gt_list]
    dir_list = sorted(dir_list, key=str.casefold)
    
    if test:
        dir_list = dir_list[:config.TEST_SIZE]
    else:
        dir_list = dir_list[:config.TRAIN_SIZE]
    
    for file_name in dir_list:
        # Hidden files, irrelevant for this usecase
        if file_name.startswith('.'):
            continue
        # Read the images
        img_path = image_dir_path + "/" + file_name
        gt_path = gt_dir_path + "/" + file_name
        
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path)
        
        # Convert Ground Truth from RGB to 1 channel (Black or White)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        _,gt = cv2.threshold(gt,127,1,0)
        
        # Resize images and ground truths to size 224*224
        img = cv2.resize(img, (NO_PIXELS,NO_PIXELS), interpolation=cv2.INTER_CUBIC)
        gt = cv2.resize(gt, (NO_PIXELS,NO_PIXELS), interpolation=cv2.INTER_CUBIC)
        
        #Store in list
        images.append(img)
        gts.append(gt)
        
    return images, gts

"""Returns train and test, both being tensors with tuples of input images and corresponding ground truths.
        Format of return: Tensor of tuples containging an input tensor and a ground truth tensor
        Image tensors are of shape (batch_size, channels, height, width)
"""
def load_data(config, train, test):
    base_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/visuAAL"

    train_images, train_gts, test_images, test_gts = [], [], [], []
    print("Loading training data...")
    train_images, train_gts = load_images(config, train_images, train_gts, base_path + "/TrainImages", base_path + "/TrainGroundTruth")
    print("Loading testing data...")
    test_images, test_gts = load_images(config, test_images, test_gts, base_path + "/TestImages", base_path + "/TestGroundTruth", test = True)
    
    # TODO: fix de eerst naar numpy en daarna pas naar tensor (sneller dan vanaf een list direct naar tensor maar nu heel lelijk)
    train = torch.utils.data.TensorDataset(torch.as_tensor(np.array(train_images)).permute(0,3,1,2), torch.as_tensor(np.array(train_gts)).permute(0,1,2))
    test = torch.utils.data.TensorDataset(torch.as_tensor(np.array(test_images)).permute(0,3,1,2), torch.as_tensor(np.array(test_gts)).permute(0,1,2))

    return train, test

class IoULoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        # Flatten the inputs to 2D tensors (batch_size * height * width, channels)
        prediction = prediction.view(-1, prediction.size(1))
        target = target.view(-1, target.size(1))

        # Calculate intersection and union
        intersection = torch.sum(prediction * target, dim=0)
        union = torch.sum(prediction + target, dim=0) - intersection

        # Calculate IoU for each channel
        iou = (intersection + self.smooth) / (union + self.smooth)

        # Average IoU across all channels
        mean_iou = torch.mean(iou)

        # Return 1 - mean IoU as the loss (to be minimized)
        loss = 1 - mean_iou

        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, prediction, target):
        # Calculate binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(prediction, target, reduction='none')

        # Calculate the modulating factor (alpha * target + (1 - alpha) * (1 - target))^gamma
        modulating_factor = (self.alpha * target + (1 - self.alpha) * (1 - target)).pow(self.gamma)

        # Calculate the focal loss
        focal_loss = bce_loss * modulating_factor

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        

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
        self.sigmoid = nn.Sigmoid()
    
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
        x = self.sigmoid(x)
        
        """ Reformatting"""
        outputs = x.clone().reshape(batch_size, NO_PIXELS, NO_PIXELS)
        
        return outputs

def make(config):
    # Make the data
    train, test = load_data(config, [], [])
    train_loader = DataLoader(train, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test, batch_size=config.BATCH_SIZE, shuffle=False)

    # Make the model
    model = UNET().to(config.device)

    # Make the loss and optimizer 
    # TODO: Dit ook in de config fixen?
    # Define loss function and optimizer
    # loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.BCEWithLogitsLoss()
    # loss_function = nn.L1Loss()
    # loss_function = nn.BCELoss()
    loss_function = IoULoss()
    # loss_function = FocalLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=config.LR, momentum=config.MOMENTUM)
    # optimizer = optim.Adam(model.parameters(), lr=config.LR)
    
    return model, train_loader, test_loader, loss_function, optimizer

def train(config, model, train_loader, loss_function, optimizer):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, loss_function, log="all", log_freq=1)
    
    # Store the losses
    train_losses = []
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"-------------------------Starting Epoch {epoch+1}/{config.NUM_EPOCHS} epochs-------------------------")
        model.train()
        epoch_train_loss = 0.0
        # epoch_test_loss = 0.0
        # batch = 0
        for images, targets in train_loader:  
            loss = train_batch(config, images, targets, model, optimizer, loss_function)
            epoch_train_loss += loss.item()
        train_log(config, epoch_train_loss, epoch)
            
    train_losses.append(epoch_train_loss/config.TRAIN_SIZE)
    print(f"mean train loss: {epoch_train_loss/config.TRAIN_SIZE:.6f}")         
    
def train_batch(config, images, targets, model, optimizer, loss_function):
    images, targets = images.to(config.device), targets.to(config.device)
    images = images.float()
    targets = targets.float()
    
    outputs = model(images)
    
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
    
    return loss

def train_log(config, epoch_train_loss, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "mean_loss": epoch_train_loss})
    print(f"mean train loss: {epoch_train_loss/config.TRAIN_SIZE:.6f}") 
    # print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
    
def test(config, model, test_loader, loss_function):
    model.eval()
    test_losses = []
    
    # Run the model on some test examples
    with torch.no_grad():
        for images, targets in test_loader:
            
            images, targets = images.to(config.device), targets.to(config.device)
            images = images.float()
            targets = targets.float()
            outputs = model(images)
            
            loss = loss_function(outputs, targets)
            test_loss += loss.item()

        test_losses.append(test_loss/config.TEST_SIZE)
        wandb.log({"mean test loss": test_losses/config.TEST_SIZE})

    # Save the model in the exchangeable ONNX format
    torch.onnx.export(model, images, "model.onnx")
    wandb.save("model.onnx")

def model_pipeline(hyperparameters):
    # tell wandb to get started
    with wandb.init(project="skin_segmentation", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      model, train_loader, test_loader, loss_function, optimizer = make(config)
      print(f"Model: {model}")

      # and use them to train the model
      train(config, model, train_loader, loss_function, optimizer)

      # and test its final performance
    test(config, model, test_loader, loss_function)

    return model

if __name__ == '__main__':
    start_time = time.time()
    model = model_pipeline(config)
    run_time = time.time() - start_time
    print("Running time: ", round(run_time,3))