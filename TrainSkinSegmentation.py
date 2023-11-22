#MSc Thesis Marieke Kopmels
import argparse
from types import SimpleNamespace

import cv2
import MyModels
import LossFunctions
import LogFunctions
import DataFunctions
import torch
import wandb
import torch.nn as nn
from torch import optim
import numpy as np
import warnings

# Options for loss function
loss_dictionary = {
    "IoU": LossFunctions.IoULoss(),
    "Focal": LossFunctions.FocalLoss(),
    # "CE": nn.CrossEntropyLoss(),
    "WBCE": nn.BCEWithLogitsLoss(),
    "WBCE_9": nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9])),
    "BCE": nn.BCELoss(),
}


# Default parameters
# Size of dataset: Train=44783 , Test=1157
default_config = SimpleNamespace(
    machine = "TS2",
    device = torch.device("cuda"),
    num_workers = 1,
    dims = 224,
    num_epochs = 25,
    batch_size = 32, 
    train_size = 32768, 
    validation_size = 128,
    test_size = 1024,
    lr = 0.0001, 
    momentum = 0.99, 
    colour_space = "RGB",
    loss_function = "WBCE_9",
    optimizer = "Adam", 
    dataset = "VisuAAL", 
    data_path = "/home/oddity/marieke/Datasets/VisuAAL",
    model_path = "/home/oddity/marieke/Output/Models/",
    architecture = "UNet"

    # machine = "Mac",
    # device = torch.device("mps"),
    # num_workers = 1,
    # dims = 224,
    # num_epochs = 1,
    # batch_size = 8, 
    # train_size = 8, 
    # validation_size = 8,
    # test_size = 8,
    # lr = 0.0001, 
    # momentum = 0.99, 
    # colour_space = "RGB",
    # loss_function = "WBCE_9",
    # optimizer = "Adam", 
    # dataset = "VisuAAL", 
    # data_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/visuAAL",
    # model_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Thesis/Models/",
    # architecture = "UNet"
)

def parse_args():
    "Overriding default arguments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--num_workers', type=int, default=default_config.num_workers, help='number of workers in DataLoader')
    argparser.add_argument('--num_epochs', type=int, default=default_config.num_epochs, help='number of epochs')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--train_size', type=int, default=default_config.train_size, help='trains size')
    argparser.add_argument('--validation_size', type=int, default=default_config.validation_size, help='validation size')
    argparser.add_argument('--test_size', type=int, default=default_config.test_size, help='test size')
    argparser.add_argument('--lr', type=float, default=default_config.lr, help='learning rate')
    argparser.add_argument('--momentum', type=float, default=default_config.momentum, help='momentum')
    argparser.add_argument('--colour_space', type=str, default=default_config.colour_space, help='colour space')
    argparser.add_argument('--loss_function', type=str, default=default_config.loss_function, help='loss function')
    argparser.add_argument('--optimizer', type=str, default=default_config.optimizer, help='optimizer')
    argparser.add_argument('--device', type=torch.device, default=default_config.device, help='device')
    argparser.add_argument('--dataset', type=str, default=default_config.dataset, help='dataset')
    argparser.add_argument('--architecture', type=str, default=default_config.architecture, help='architecture')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

""" Returns the optimizer based on the configurations
"""
def get_optimizer(config, model):
    if config.optimizer == "SGD":
        return optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    elif config.optimizer == "Adam":
        return optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == "RMSprop":
        return optim.RMSprop(model.parameters(),lr=config.lr)
    else:
        warnings.warn("No matching optimizer found! Used default SGD")
        print(f"Current config.optimizer = {config.optimizer}")
        return optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    
""" Returns dataloaders, model, loss function and optimizer.
"""
def make(config):
    # Fetch data
    train_loader, validation_loader, test_loader = DataFunctions.load_data(config)
    
    # Make the model
    model = MyModels.UNET(config.dims).to(config.device)

    # Define loss function and optimizer
    loss_function = loss_dictionary[config.loss_function].to(config.device)
    optimizer = get_optimizer(config, model)
    
    return model, train_loader, validation_loader, test_loader, loss_function, optimizer


""" Trains the passed model, tests it performance after each epoch on the validation set. Prints and logs the results to WandB.
"""
def train(config, model, train_loader, validation_loader, loss_function, optimizer):
    print(f"-------------------------Start Training-------------------------")
    for epoch in range(config.num_epochs):
        print(f"-------------------------Starting Training Epoch {epoch+1}/{config.num_epochs} epochs-------------------------")
        model.train()
        epoch_loss = 0.0
        batch = 0
        epoch_tn, epoch_fn, epoch_fp, epoch_tp = 0, 0, 0, 0
        for images, targets in train_loader:  
            batch += 1
            print(f"-------------------------Starting Batch {batch}/{int(config.train_size/config.batch_size)} batches-------------------------")
            batch_loss, batch_tn, batch_fn, batch_fp, batch_tp = train_batch(config, images, targets, model, optimizer, loss_function)
            epoch_loss += batch_loss.item()
            epoch_tn += batch_tn
            epoch_fn += batch_fn
            epoch_fp += batch_fp
            epoch_tp += batch_tp
            LogFunctions.log_metrics(batch_loss/config.batch_size, batch_tn, batch_fn, batch_fp, batch_tp, "batch")
            
        mean_loss = epoch_loss/config.train_size
        LogFunctions.log_metrics(mean_loss, epoch_tn, epoch_fn, epoch_fp, epoch_tp, "train")
        LogFunctions.save_model(config, model, epoch+1)
        test_performance(config, epoch, model, validation_loader, loss_function, "validation")
        

""" Performs training for one batch of datapoints. Returns the true/false positive/negative metrics. 
"""
def train_batch(config, images, targets, model, optimizer, loss_function):
    
    images, targets = images.to(config.device), targets.to(config.device)
    images = images.float()
    targets = targets.float()
    outputs = model(images)
    
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
    
    tn, fn, fp, tp = DataFunctions.confusion_matrix(outputs, targets, "train")
    return loss, tn, fn, fp, tp

def test_performance(config, epoch, model, data_loader, loss_function, stage):
    print(f"-------------------------Start {stage}-------------------------")
    model.eval()
        
    # Test the performance of the model on the data in the passed data loader, either test or validation data
    with torch.no_grad():
        total_loss = 0.0
        total_tn, total_fn, total_fp, total_tp = 0, 0, 0, 0
        batch = 0
        for images, targets in data_loader:
            batch += 1
            print(f"-------------------------Starting Batch {batch}/{int(len(data_loader.dataset)/config.batch_size)} batches-------------------------")

            # Store example for printing while on CPU
            example_image = np.array(images[0].permute(1,2,0))
            
            images, targets = images.to(config.device), targets.to(config.device)
            images = images.float()
            targets = targets.float()
            outputs = model(images)
            
            # Log example to WandB
            LogFunctions.log_example(config, example_image, targets[0], outputs[0], stage)
            
            # Update metrics
            batch_loss = loss_function(outputs, targets).item()
            total_loss += batch_loss
            batch_tn, batch_fn, batch_fp, batch_tp = DataFunctions.confusion_matrix(outputs, targets, stage)
            total_tn += batch_tn
            total_fn += batch_fn
            total_fp += batch_fp
            total_tp += batch_tp
            LogFunctions.log_metrics(batch_loss/config.batch_size, batch_tn, batch_fn, batch_fp, batch_tp, "batch")

        mean_loss = total_loss/len(data_loader.dataset)
        LogFunctions.log_metrics(mean_loss, total_tn, total_fn, total_fp, total_tp, stage)
        
        # if stage == "validation":
        #     # Save the model
        #     # TODO: fix, it now raises errors
        #     x = torch.ones(config.batch_size, 3, config.dims, config.dims).to(dtype=torch.float32).to(config.device)
        #     torch.onnx.export(model, x, f"Models/model_epoch_{epoch}.onnx")
        #     wandb.save(f"model_epoch_{epoch}.onnx")

""" Runs the whole pipeline of creating, training and testing a model
"""
def model_pipeline(hyperparameters):
    # Start wandb
    with wandb.init(project="skin_segmentation", config=hyperparameters): #mode="disabled", 
        # Set hyperparameters
        config = wandb.config
        run_name = f"{config.machine}_batch_size:{config.batch_size}_num_workers:{config.num_workers}"
        wandb.run.name = run_name

        # Create model, data loaders, loss function and optimizer
        model, train_loader, validation_loader, test_loader, loss_function, optimizer = make(config)
        wandb.watch(model, log="all", log_freq=1)

        # Train the model, incl. validation
        train(config, model, train_loader, validation_loader, loss_function, optimizer)

        # Test the models performance
        test_performance(config, config.num_epochs, model, test_loader, loss_function, "test")
        
        # Store the final model
        LogFunctions.save_model(config, model, 0, final=True)

    return model

if __name__ == '__main__':
    parse_args()
    model = model_pipeline(default_config)