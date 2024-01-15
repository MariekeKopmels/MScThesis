# Main for training the multi-head violence and skin colour prediction mode

import Config.ConfigFunctions as ConfigFunctions
import Data.DataFunctions as DataFunctions
import Models.MyModels as MyModels
import Models.LossFunctions as LossFunctions
import Logging.LogFunctions as LogFunctions
import wandb
import torch
import numpy as np
from torch import optim
from torch.cuda import amp
from sklearn.metrics import f1_score


"""Returns the grad scaler in case of automatic mixed precision.
"""
def getGradScaler(config):
    if config.automatic_mixed_precision:
        if config.machine == "TS2" or config.machine == "OS4":
            return amp.GradScaler(enabled=config.automatic_mixed_precision)
        else: 
            Warning("Machine not approved to use for Automatic Mixed Precision, so AMP turned off.")
    return None
    
""" Trains the passed model"""
def train(config, model, scaler, loss_function, optimizer, data_loader):
    val_f1_scores = np.zeros((config.num_epochs, 2))
    
    for epoch in range(config.num_epochs):
        print(f"-------------------------Starting Training Epoch {epoch+1}/{config.num_epochs} epochs-------------------------")
        model.train()
        epoch_loss = 0.0
        batch = 0
        
        train_loader, validation_loader = DataFunctions.split_dataset(config, data_loader)
        for videos, targets in train_loader:  
            violence_targets, skincolour_targets = targets[:, 0], targets[:, 1:]
            batch += 1
            print(f"-------------------------Starting Batch {batch}/{int(config.train_size/config.batch_size)} batches-------------------------", end="\r")
            batch_loss, batch_violence_outputs, batch_skincolour_outputs = train_batch(config, scaler, videos, violence_targets, skincolour_targets, model, optimizer, loss_function)
            epoch_loss += batch_loss.item()
                                    
        print(f"-------------------------Finished training batches-------------------------") 
        
        # Validate the performance of the model
        val_f1_scores[epoch] = test_performance(config, model, validation_loader, loss_function, "validation")
    return

def train_batch(config, scaler, videos, violence_targets, skincolour_targets, model, optimizer, loss_function):
    # TODO: create helper function and reduce code duplication in this if/else statement (not possible yet when running on mac as well)
    if config.automatic_mixed_precision:
        with amp.autocast(enabled=config.automatic_mixed_precision):
            # Model inference
            videos, violence_targets, skincolour_targets = videos.to(config.device), violence_targets.to(config.device), skincolour_targets.to(config.device)
            
            # TODO: normalize videos? 
            # normalized_images = DataFunctions.normalize_images(config, images)
            raw_violence_outputs, raw_skincolour_outputs = model(videos)
            
            # Compute loss, update model
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(raw_violence_outputs, raw_skincolour_outputs, violence_targets, skincolour_targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    else:
        # Model inference
        videos, violence_targets, skincolour_targets = videos.to(config.device), violence_targets.to(config.device), skincolour_targets.to(config.device)
        # TODO: normalize videos? 
        # normalized_images = DataFunctions.normalize_images(config, images)
                
        raw_violence_outputs, raw_skincolour_outputs = model(videos)
        
        # Compute loss, update model
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(raw_violence_outputs, raw_skincolour_outputs, violence_targets, skincolour_targets)
        loss.backward()
        optimizer.step()
        
    # Return the outputs in the correct format where violence is either 0 or 1
    # and skin colour is one-hot encoded. 
    violence_outputs = raw_violence_outputs > 0.5
    skincolour_outputs = torch.argmax(raw_skincolour_outputs, dim=1)
    
    return loss, violence_outputs, skincolour_outputs

def test_performance(config, model, data_loader, loss_function, stage):
    print(f"-------------------------Start {stage}-------------------------")
    model.eval()
    
    with torch.no_grad():
        total_loss = 0.0
        batch = 0
        test_violence_outputs = torch.empty((0)).to(config.device)
        test_violence_targets = torch.empty((0)).to(config.device)
        test_skincolour_outputs = torch.empty((0)).to(config.device)
        test_skincolour_targets = torch.empty((0)).to(config.device)
        
        for videos, targets in data_loader:
            violence_targets, skincolour_targets = targets[:, 0], targets[:, 1:]
            batch += 1
            print(f"-------------------------Starting Batch {batch}/{int(len(data_loader.dataset)/config.batch_size)} batches-------------------------", end="\r")

            # Model inference 
            videos, violence_targets, skincolour_targets = videos.to(config.device), violence_targets.to(config.device), skincolour_targets.to(config.device)
            # TODO: normalize videos?
            # normalized_images = DataFunctions.normalize_images(config, images)
            raw_violence_outputs, raw_skincolour_outputs = model(videos)
            
            # Transform the outputs into the correct format where violence is either 0 or 1
            # and skin colour is one-hot encoded. 
            violence_outputs = raw_violence_outputs > 0.5
            skincolour_outputs = torch.argmax(raw_skincolour_outputs, dim=1)
            
            # Compute batch loss and add to total loss
            batch_loss = loss_function(raw_violence_outputs, raw_skincolour_outputs, violence_targets, skincolour_targets).item()
            total_loss += batch_loss
            
            # Store all outputs and targets
            test_violence_outputs = torch.cat((test_violence_outputs, violence_outputs), dim=0)
            test_violence_targets = torch.cat((test_violence_targets, violence_targets.to(config.device)), dim=0)
            test_skincolour_outputs = torch.cat((test_skincolour_outputs, skincolour_outputs), dim=0)
            test_skincolour_targets = torch.cat((test_skincolour_targets, skincolour_targets.to(config.device)), dim=0)
            
        print(f"-------------------------Finished {stage} batches-------------------------")
        
        # Compute and log metrics
        argmax_test_skincolour_targets = torch.argmax(test_skincolour_targets, dim=1).float()
        num_batches = len(data_loader.dataset) // config.batch_size
        mean_loss = total_loss / (num_batches * config.batch_size)
        violence_f1_score = f1_score(test_violence_targets.to("cpu").numpy(), test_violence_outputs.to("cpu").numpy(), )  
        skincolour_f1_score = f1_score(argmax_test_skincolour_targets.to("cpu").numpy(), test_skincolour_outputs.to("cpu").numpy(), average='micro')
        LogFunctions.log_multitask_metrics(config, mean_loss, violence_f1_score, skincolour_f1_score, stage)

    return violence_f1_score, skincolour_f1_score
    
def make(config):
    print("Creating data loaders")
    train_loader, test_loader = DataFunctions.load_video_data(config)
    
    model = MyModels.I3DMultiTaskModel(config).to(config.device)
    loss_function = LossFunctions.MultiTaskLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scaler = getGradScaler(config)
    
    return model, scaler, loss_function, optimizer, train_loader, test_loader

def multitask_learning_pipeline(hyperparameters):
    with wandb.init(mode="online", project="multi-task-model", config=hyperparameters):
        config = wandb.config
        
        # Create model, data loaders, loss function and optimizer
        model, scaler, loss_function, optimizer, train_loader, test_loader = make(config)
        
        # Train the model, incl. validation
        train(config, model, scaler, loss_function, optimizer, train_loader)
        
        # Test the final model's performance
        test_performance(config, model, test_loader, loss_function, "test")
        
    return        

if __name__ == '__main__':
    args = ConfigFunctions.parse_args()
    config = ConfigFunctions.load_config(args.config)
    multitask_learning_pipeline(config)
