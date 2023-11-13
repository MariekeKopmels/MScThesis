
from types import SimpleNamespace
import numpy as np

import wandb
import MyModels
import DataFunctions
import LossFunctions
import torch
import torch.nn as nn
from torch import optim
from sklearn.ensemble import RandomForestClassifier
import torchvision

# Options for loss function
loss_dictionary = {
    "IoU": LossFunctions.IoULoss(),
    "Focal": LossFunctions.FocalLoss(),
    "CE": nn.CrossEntropyLoss(),
    "BCE": nn.BCELoss(),
    "L1": nn.L1Loss(),
}

config = SimpleNamespace(
    num_epochs = 50,
    batch_size = 64, 
    train_size = 16384,
    test_size = 1024,
    lr = 0.01, 
    momentum = 0.999, 
    colour_space = "RGB",
    loss_function = "BCE", 
    device = torch.device("cpu"),
    # dataset = "VisuAAL", 
    # architecture = "UNet"
)

""" Trains the passed model. 
"""
def train(config, model, train_loader, loss_function, optimizer):
    wandb.watch(model, log="all", log_freq=1)
    model.train()
    
    for epoch in range(config.num_epochs):
        print(f"-------------------------Starting Epoch {epoch+1}/{config.num_epochs} epochs-------------------------")
        epoch_loss = 0.0
        batch = 0
        epoch_tn, epoch_fn, epoch_fp, epoch_tp = 0, 0, 0, 0
        for pixels, labels in train_loader:  
            batch += 1
            # print(f"-------------------------Starting new Batch -------------------------")
            batch_loss, batch_tn, batch_fn, batch_fp, batch_tp = train_batch(config, pixels, labels, model, optimizer, loss_function)
            epoch_loss += batch_loss.item()
            epoch_tn += batch_tn
            epoch_fn += batch_fn
            epoch_fp += batch_fp
            epoch_tp += batch_tp
        mean_loss = epoch_loss/config.train_size
        accuracy, fn_rate, fp_rate, sensitivity, f1_score = DataFunctions.metrics(batch_tn, batch_fn, batch_fp, batch_tp, pixels=True)
        train_epoch_log(config, mean_loss, accuracy, fn_rate, fp_rate, sensitivity, f1_score, epoch)
        
""" Performs training for one batch of datapoints. Returns the true/false positive/negative metrics. 
"""
def train_batch(config, pixels, labels, model, optimizer, loss_function):
    pixels, labels = pixels.to(config.device), labels.to(config.device)
    pixels = pixels.float()
    labels = labels.float()
    
    outputs = model(pixels)
    
    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()
    
    tn, fn, fp, tp = DataFunctions.pixel_confusion_matrix(config, outputs, labels)
    
    return loss, tn, fn, fp, tp
    
""" Prints and logs the intermediate training results to WandB.
"""
def train_epoch_log(config, mean_loss, accuracy, fn_rate, fp_rate, sensitivity, f1_score, epoch):
    wandb.log({"epoch": epoch, "mean_loss": mean_loss, "accuracy": accuracy, "fn_rate": fn_rate, "fp_rate": fp_rate, "sensitivity": sensitivity, "f1_score": f1_score})
    print(f"Mean loss: {mean_loss:.6f}, accuracy: {accuracy:.3f}, fn_rate: {fn_rate:.3f}, fp_rate:: {fp_rate:.3f}, sensitivity: {sensitivity:.3f}, f1-score: {f1_score:.3f}") 
    
        
def test(config, model, test_loader, loss_function):
    print(f"-------------------------Start testing-------------------------")
    model.eval()
     
    with torch.no_grad():   
        total_loss = 0.0
        total_tn, total_fn, total_fp, total_tp = 0, 0, 0, 0
        for pixels, labels in test_loader:  
                    
            pixels, labels = pixels.to(config.device), labels.to(config.device)
            pixels = pixels.float()
            labels = labels.float()
            
            outputs = model(pixels)
                    
            batch_loss = loss_function(outputs, labels)   
            batch_tn, batch_fn, batch_fp, batch_tp = DataFunctions.pixel_confusion_matrix(config, outputs, labels, test=True)
            
            total_loss += batch_loss.item()
            total_tn += batch_tn
            total_fn += batch_fn
            total_fp += batch_fp
            total_tp += batch_tp
            
        mean_test_loss = total_loss/config.test_size
        accuracy, fn_rate, fp_rate, sensitivity, f1_score, IoU = DataFunctions.metrics(total_tn, total_fn, total_fp, total_tp)
        test_log(config, mean_test_loss, accuracy, fn_rate, fp_rate, sensitivity, f1_score, IoU)

""" Prints and logs the test results to WandB.
"""
def test_log(config, mean_test_loss, test_accuracy, test_fn_rate, test_fp_rate, test_sensitivity, test_f1_score, test_IoU):
    # wandb.log({"mean_test_loss": mean_test_loss, "test_accuracy": test_accuracy, "test_fn_rate": test_fn_rate, "test_fp_rate": test_fp_rate, "test_sensitivity": test_sensitivity, "test_f1_score": test_f1_score, "test_IoU": test_IoU})
    print(f"Mean loss: {mean_test_loss:.6f}, accuracy: {test_accuracy:.3f}, fn_rate: {test_fn_rate:.3f}, fp_rate:: {test_fp_rate:.3f}, sensitivity: {test_sensitivity:.3f}, f1-score: {test_f1_score:.3f}, IoU: {test_IoU:.3f}")
    
def model_pipeline(hyperparameters):
    # Start WandB
    with wandb.init(mode="disabled", project="skin_segmentation", config=hyperparameters): #mode="disabled", 
        # set hyperparameters
        config = wandb.config
        
        # model = torchvision.models.resnet50().to(config.device)
        model = MyModels.SkinClassifier()
        train_loader, test_loader = DataFunctions.load_pixel_data(config, [], [])
        loss_function = loss_dictionary[config.loss_function]
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
        
        train(config, model, train_loader, loss_function, optimizer)
        test(config, model, test_loader, loss_function)

if __name__ == '__main__':
    model = model_pipeline(config)