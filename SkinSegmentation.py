#MSc Thesis Marieke Kopmels
import random
import time
from os import listdir
import numpy as np

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch import flatten

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax

from torch.utils.data import DataLoader

# from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
BATCH_SIZE      = 256
NUM_EPOCHS      = 5
LR              = 0.01
MOMENTUM        = 0.9
TRAIN_SIZE      = 5000
TEST_SIZE       = 500
NO_PIXELS       = 224

"""Returns images in a given directory
        Format of return: list of NumPy arrays.
        NumPy arrays are of shape (224,224,3) for images and (224,224) for ground truths 
"""
def load_images(images, gts, image_dir_path, gt_dir_path, test=False):
    # Load list of files and directories
    image_list = listdir(image_dir_path)
    gt_list = listdir(gt_dir_path)
    
    # Not all images have a ground truth, select those that do
    dir_list = [file for file in image_list if file in gt_list]
    dir_list = sorted(dir_list, key=str.casefold)
    
    if test:
        dir_list = dir_list[:TEST_SIZE]
    else:
        dir_list = dir_list[:TRAIN_SIZE]
    
    for file_name in dir_list:
        # Hidden files, irrelevant for this usecase
        if file_name.startswith('.'):
            continue
        # Read the images
        img_path = image_dir_path + "/" + file_name
        gt_path = gt_dir_path + "/" + file_name
        
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path)
        
        # Convert Ground Truth from RGB to Black or White
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        ret,gt = cv2.threshold(gt,70,255,0)
        
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
def load_data(train, test):
    base_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/visuAAL"
    device = torch.device("mps")

    train_images, train_gts, test_images, test_gts = [], [], [], []
    print("Loading training data...")
    train_images, train_gts = load_images(train_images, train_gts, base_path + "/TrainImages", base_path + "/TrainGroundTruth")
    # train_gt = load_images(train_gt, base_path + "/TrainGroundTruth", gt=True)
    print("Loading testing data...")
    test_images, test_gts = load_images(test_images, test_gts, base_path + "/TestImages", base_path + "/TestGroundTruth", test = True)
    # test_gt = load_images(test_gt, base_path + "/TestGroundTruth", gt=True)
    
    # print(f"train_images.shape: {len(train_images)}, train_gts.shape: {len(train_gts)}")
    # print(f"train_images.shape: {len(test_images)}, train_gts.shape: {len(test_gts)}")
    
    # TODO: fix de eerst naar numpy en daarna pas naar tensor (sneller dan vanaf een list direct naar tensor maar nu heel lelijk)
    train = torch.utils.data.TensorDataset(torch.as_tensor(np.array(train_images)).permute(0,3,1,2), torch.as_tensor(np.array(train_gts)).permute(0,1,2))
    test = torch.utils.data.TensorDataset(torch.as_tensor(np.array(test_images)).permute(0,3,1,2), torch.as_tensor(np.array(test_gts)).permute(0,1,2))

    return train, test

class LeNet(Module):
    def __init__(self, numChannels, num_pixels):
		# call the parent constructor
        super(LeNet, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=10, kernel_size=(3, 3))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=20*54*54, out_features=1000)
        self.relu3 = ReLU()
        
        # initialize our softmax classifier
        #TODO: Wil ik jpeg (dus RBG) voorspellen of skin/non-skin voorspellen en dan JPEG ground truths naar 2D formatten
        self.fc2 = Linear(in_features=1000, out_features=NO_PIXELS*NO_PIXELS)
        self.logSoftmax = LogSoftmax(dim=1)
        
    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        # print("Performing forward pass")
        # print(f"Input type: {type(x)}")
        # print(f"Input shape: {x.shape}")
        # print(f"Input[0][0][0][0] type: {type(x[0][0][0][0])}")
        # print(f"Input[0] dtype: {x.dtype}")
        
        # x = x.float()
        
        # print(f"Input type: {type(x)}")
        # print(f"Input shape: {x.shape}")
        # print(f"Input[0][0][0][0] type: {type(x[0][0][0][0])}")
        # print(f"Input[0] dtype: {x.dtype}")
        
        # For the last few items, the batch size may be smaller than BATCH_SIZE
        batch_size = len(x)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        x = self.logSoftmax(x)
        
        # TODO: Checken of deze manier van batch_size werkt ipv hardcoded
        output = x.reshape(batch_size,NO_PIXELS,NO_PIXELS)
        
        # return the output predictions
        return output

"""Goal is to create a model, load traning and test data and evaltually train and evaluate the model.
"""
if __name__ == '__main__':
    start_time = time.time()
    # Create model
    # model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT, num_channels=2)
    model = LeNet(numChannels=3)
    
    # This should allow for it to run on M1 GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    
    # Store the losses
    train_losses = []
    test_losses = []

    # Load data, both input images and ground truths. Put them into a DataLoader
    train, test = load_data([], [])
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)
    
    demo_check = False
        
    #Train the model
    for epoch in range(NUM_EPOCHS):
        print(f"-------------------------Starting Epoch {epoch+1}/{NUM_EPOCHS} epochs-------------------------")
        model.train()
        epoch_train_loss = 0.0
        epoch_test_loss = 0.0
        for images, targets in train_loader:            
            images = images.to(device)
            images = images.float()
            targets = targets.to(device)
            targets = targets.float()
            
            # Train the model
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            if epoch == NUM_EPOCHS-1 and demo_check == False:
                demo_image = images[0].permute(1,2,0)
                demo_gt = targets[0]
                demo_output = model(images[0:5])
                demo_check = True
            
            epoch_train_loss += loss.item()
            
        for images, targets in test_loader:
            
            images = images.to(device)
            images = images.float()
            targets = targets.to(device)
            targets = targets.float()
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            epoch_test_loss += loss.item()
        
        # Print losses TODO: is the /X factor correct? 
        train_losses.append(epoch_train_loss/TRAIN_SIZE)
        test_losses.append(epoch_test_loss/TEST_SIZE)
        print(f"[{epoch + 1}] \ntrain loss: {epoch_train_loss/TRAIN_SIZE:.6f} \ntest  loss: {epoch_test_loss/TEST_SIZE:.6f} ")
    
    
    run_time = time.time() - start_time
    print("Running time: ", round(run_time,3))
    
    # Print demo. Image, model output and ground truth.
    img =  demo_image.cpu().numpy().astype(np.uint8)
    out = demo_output[0].cpu().detach().numpy()
    gt = demo_gt.cpu().numpy()
    
    # print(f"out.shape: {np.shape(out)}")
    rgb_out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    # print(f"rgb_out.shape: {np.shape(rgb_out)}")
    
    # print("Model output")
    # print(out)
    # print("Ground truth")
    # print(gt)
    
    # cv2.namedWindow("demo_image")
    # cv2.imshow('demo_image', img)
    # cv2.waitKey(0)
    # cv2.namedWindow("demo_output")
    # cv2.imshow('demo_output',rgb_out)
    # cv2.waitKey(0)
    # cv2.namedWindow("demo_gt")
    # cv2.imshow('demo_gt',gt)
    # cv2.waitKey(0)
    
    # Plot losses
    x = range(1, NUM_EPOCHS + 1)
    plt.plot(x, train_losses, label='Train Loss', color='blue')
    plt.plot(x, test_losses, label='Test Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(50, 65)
    plt.legend()
    plt.show()
    
    
            
