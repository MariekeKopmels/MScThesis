import time
import os
import numpy as np
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.utils.data import DataLoader
import torch.utils.data
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from PIL import Image
from torch import optim
from torch import flatten
from os import listdir
import cv2
import torch.nn as nn
import torch
import torch.utils.data

# LET OP: Batch size moet devidable by 60 zijn, anders gaat output = x.reshape(6,2,224,224) niet goed bij de laatste batch.
BATCH_SIZE = 20
NUM_EPOCHS = 20
LR=0.05
MOMENTUM = 0.99

"""Returns images in a given directory
        Format of return: list of NumPy arrays.
        NumPy arrays are of shape (224,224,3) 
"""
def load_images(data, dir_path):
    # Load list of files and directories
    data_list = listdir(dir_path)
    data_list = sorted(data_list, key=str.casefold)
    
    for img_name in data_list:
        # Hidden files, irrelevant for this usecase
        if img_name.startswith('.'):
            continue
        # Files
        img_path = dir_path + "/" + img_name
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224,224), interpolation=cv2.INTER_CUBIC)
        data.append(img)
    return data

"""Returns train and test, both being tensors with tuples of input images and corresponding ground truths.
        Format of return: Tensor of tuples containging an input tensor and a ground truth tensor
        Image tensors are of shape (batch_size, channels, height, width)
"""
def load_data(train, test):
    base_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/Pratheepan_Dataset"
    device = torch.device("mps")

    train_images, train_gt, test_images, test_gt = [], [], [], []
    train_images = load_images(train_images, base_path + "/TrainData")
    train_gt = load_images(train_gt, base_path + "/TrainGroundTruth")
    test_images = load_images(test_images, base_path + "/TestData")
    test_gt = load_images(test_gt, base_path + "/TestGroundTruth")
    # test_images = load_images(test_images, base_path + "/MockTrainData")
    # test_gt = load_images(test_gt, base_path + "/MockTrainGroundTruth")
        
    # TODO: fix de eerst naar numpy en daarna pas naar tensor (sneller dan vanaf een list direct naar tensor maar nu heel lelijk)
    train = torch.utils.data.TensorDataset(torch.as_tensor(np.array(train_images)).permute(0,3,1,2), torch.as_tensor(np.array(train_gt)).permute(0,3,1,2))
    test = torch.utils.data.TensorDataset(torch.as_tensor(np.array(test_images)).permute(0,3,1,2), torch.as_tensor(np.array(test_gt)).permute(0,3,1,2))

    return train, test

class LeNet(Module):
    def __init__(self, numChannels, classes):
		# call the parent constructor
        super(LeNet, self).__init__()

        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=10,kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv2d(in_channels=10, out_channels=20,kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=20*53*53, out_features=1000)
        self.relu3 = ReLU()
        
        # initialize our softmax classifier
        #TODO: Wil ik jpeg (dus RBG) voorspellen of skin/non-skin voorspellen en dan JPEG ground truths naar 2D formatten
        self.fc2 = Linear(in_features=1000, out_features=224*224*3)
        self.logSoftmax = LogSoftmax(dim=1)
        
    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
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
        
        # TODO: Fixen dat dit niet hardcoded is
        output = x.reshape(BATCH_SIZE,3,224,224)
        
        # return the output predictions
        return output

"""Goal is to create a model, load traning and test data and evaltually train and evaluate the model.
"""
if __name__ == '__main__':
    start_time = time.time()
    # Create model
    # model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT, num_channels=2)
    model = LeNet(numChannels=3, classes=2)
    
    # This should allow for it to run on M1 GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")
    model.to(device)
    
    # Initialize optimizer and learining rate scheduler
    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    # Load data, both input images and ground truths. Put them into a DataLoader
    train, test = load_data([], [])
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    #Train the model
    for epoch in range(NUM_EPOCHS):
        print("-------------------------New Epoch-------------------------")
        model.train()
        epoch_train_loss = 0.0
        epoch_test_loss = 0.0
        for images, targets in train_loader:
            # print("-------------------------New Batch-------------------------")
            list_images = list(image for image in images)
            list_targets = list(target for target in targets)
            
            images = images.to(device)
            images = images.float()
            targets = targets.to(device)
            targets = targets.float()
                    
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
        for images, targets in test_loader:
            list_images = list(image for image in images)
            list_targets = list(target for target in targets)
            images = images.to(device)
            images = images.float()
            targets = targets.to(device)
            targets = targets.float()
            
            loss = criterion(outputs, targets)
            epoch_test_loss += loss.item()
        
        
        # Print losses TODO: is the /X factor correct? 
        print(f"[{epoch + 1}] \ntrain loss: {epoch_train_loss/60:.6f} \ntest  loss: {epoch_test_loss/20:.6f} ")
        
            
    run_time = time.time() - start_time
    print("Running time: ", round(run_time,3))
            
    






