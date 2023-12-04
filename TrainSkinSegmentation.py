#MSc Thesis Marieke Kopmels
import argparse
from types import SimpleNamespace

import time
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
    "WBCE": nn.BCEWithLogitsLoss(),
    "WBCE_9": nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9])),
    "BCE": nn.BCELoss(),
}

# Default parameters
# Size of dataset: Train=44783, Test=1157
default_config = SimpleNamespace(
    # machine = "TS2",
    # device = torch.device("cuda"),
    # log = True,
    # num_workers = 2,
    # dims = 224,
    # num_epochs = 10,
    # batch_size = 16, 
    # train_size = 2048, 
    # validation_size = 256,
    # test_size = 60,
    # cm_train = False,
    # cm_parts = 16,
    # lr = 0.001, 
    # momentum = 0.999, 
    # colour_space = "RGB",
    # loss_function = "WBCE_9",
    # optimizer = "RMSprop", 
    # dataset = "VisuAAL", 
    # testset = "Combined",
    # data_path = "/home/oddity/marieke/Datasets/VisuAAL",
    # testdata_path = "/home/oddity/marieke/Datasets/CombinedTestset",
    # model_path = "/home/oddity/marieke/Output/Models/",
    # architecture = "UNet"

    machine = "Mac",
    device = torch.device("mps"),
    log = True,
    num_workers = 4,
    dims = 224,
    num_epochs = 5,
    batch_size = 8, 
    train_size = 64, 
    validation_size = 16,
    test_size = 32,
    cm_train = True,
    cm_parts = 1,
    lr = 0.0001, 
    momentum = 0.99, 
    colour_space = "RGB",
    loss_function = "WBCE_9",
    optimizer = "Adam", 
    dataset = "AugmentationPratheepan", 
    testset = "Combined",
    data_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/VisuAAL",
    testdata_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/CombinedTestset",
    model_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Thesis/Models/",
    architecture = "UNet"
)

def parse_args():
    "Overriding default arguments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--log', type=str, default=default_config.log, help='turns logging on or off')
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
    train_loader, validation_loader, test_loader = DataFunctions.load_image_data(config)
    
    # Make the model
    model = MyModels.UNET(config).to(config.device)

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
        epoch_outputs = torch.empty((0, config.dims, config.dims)).to(config.device)
        epoch_targets = torch.empty((0, config.dims, config.dims)).to(config.device)
        for images, targets in train_loader:  
            batch += 1
            print(f"-------------------------Starting Batch {batch}/{int(config.train_size/config.batch_size)} batches-------------------------", end="\r")
            batch_loss, batch_outputs = train_batch(config, images, targets, model, optimizer, loss_function)
            epoch_loss += batch_loss.item()
            
            if config.cm_train:
                epoch_outputs = torch.cat((epoch_outputs, batch_outputs), dim=0)
                epoch_targets = torch.cat((epoch_targets, targets.to(config.device)), dim=0)
            
        print(f"-------------------------Finished training batches-------------------------") 
        
        # Keep track of training epoch stats, or skip for sake of efficiency
        if config.cm_train:
            epoch_tn, epoch_fn, epoch_fp, epoch_tp = DataFunctions.confusion_matrix(config, epoch_outputs, epoch_targets, "train")
            # drop_last=True in the dataloader, so we compute the amount of batches first
            num_batches = train_loader.dataset // config.batch_size
            mean_loss = epoch_loss / (num_batches * config.batch_size)
            LogFunctions.log_metrics(config, mean_loss, epoch_tn, epoch_fn, epoch_fp, epoch_tp, "train")
        
        # Save the model
        LogFunctions.save_model(config, model, epoch+1)
        
        # Test the performance with validation data
        test_performance(config, model, validation_loader, loss_function, "validation")
        

""" Performs training for one batch of datapoints. Returns the true/false positive/negative metrics. 
"""
def train_batch(config, images, targets, model, optimizer, loss_function):
    
    # Model inference
    images, targets = images.to(config.device), targets.to(config.device)
    outputs = model(images)
    
    # Compute loss, update model
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
    
    return loss, outputs

def test_performance(config, model, data_loader, loss_function, stage):
    print(f"-------------------------Start {stage}-------------------------")
    model.eval()
        
    # Test the performance of the model on the data in the passed data loader, either test or validation data
    with torch.no_grad():
        total_loss = 0.0
        batch = 0
        test_outputs = torch.empty((0, config.dims, config.dims)).to(config.device)
        test_targets = torch.empty((0, config.dims, config.dims)).to(config.device)
        for images, targets in data_loader:
            batch += 1
            print(f"-------------------------Starting Batch {batch}/{int(len(data_loader.dataset)/config.batch_size)} batches-------------------------", end="\r")

            # Store example for printing while on CPU
            example_image = np.array(images[0].permute(1,2,0))
            
            # Model inference 
            images, targets = images.to(config.device), targets.to(config.device)
            batch_outputs = model(images)
            
            # Log example to WandB
            LogFunctions.log_example(config, example_image, targets[0], batch_outputs[0], stage)
            
            # Compute batch loss
            batch_loss = loss_function(batch_outputs, targets).item()
            
            test_outputs = torch.cat((test_outputs, batch_outputs), dim=0)
            test_targets = torch.cat((test_targets, targets.to(config.device)), dim=0)
            
            total_loss += batch_loss
            
        print(f"-------------------------Finished {stage} batches-------------------------")
        # Compute and log metrics
        epoch_tn, epoch_fn, epoch_fp, epoch_tp = DataFunctions.confusion_matrix(config, test_outputs, test_targets, stage)
        # drop_last=True in the dataloader, so we compute the amount of batches first
        num_batches = data_loader.dataset // config.batch_size
        mean_loss = total_loss / (num_batches * config.batch_size)
        LogFunctions.log_metrics(config, mean_loss, epoch_tn, epoch_fn, epoch_fp, epoch_tp, stage)
        

""" Runs the whole pipeline of creating, training and testing a model
"""
def model_pipeline(hyperparameters):
    # Start wandb
    with wandb.init(mode="disabled", project="skin_segmentation", config=hyperparameters): #mode="disabled", 
        # Set hyperparameters
        config = wandb.config
        run_name = f"Machine:{config.machine}_Dataset:{config.dataset}_train_size:{config.train_size}_num_epochs:{config.num_epochs}"
        wandb.run.name = run_name

        # TODO: Aanzetten en testen
        # LogFunctions.init_device(config)

        # Create model, data loaders, loss function and optimizer
        model, train_loader, validation_loader, test_loader, loss_function, optimizer = make(config)
        if config.log:
            wandb.watch(model, log="all", log_freq=1)
        else: 
            wandb.watch(model, log=None, log_freq=1)

        # Train the model, incl. validation
        train(config, model, train_loader, validation_loader, loss_function, optimizer)

        # Test the models performance
        test_performance(config, model, test_loader, loss_function, "test")
        
        # Store the final model
        LogFunctions.save_model(config, model, 0, final=True)

    return model

if __name__ == '__main__':
    start_time = time.time()
    parse_args()
    model = model_pipeline(default_config)
    print(f"Runtime = {time.time() - start_time}")