# Main for training the I3D violence model
import Config.ConfigFunctions as ConfigFunctions
import Data.DataFunctions as DataFunctions
import Models.MyModels as MyModels
import Logging.LogFunctions as LogFunctions
import wandb
import torch
import random
import torch.nn as nn
import numpy as np
from torch import optim
from torch.cuda import amp

"""Returns the grad scaler in case of automatic mixed precision.
"""
def getGradScaler(config):
    if config.automatic_mixed_precision:
        if config.machine == "OTS5":
            return amp.GradScaler(enabled=config.automatic_mixed_precision)
        else: 
            Warning("Machine not approved to use for Automatic Mixed Precision, so AMP is turned off.")
    return None
    
""" Trains the passed model"""
def train(config, model, scaler, loss_function, optimizer, data_loader):
    # TODO Implement early stopping, or remove this f1 score tracker
    val_f1_scores = np.zeros((config.num_epochs, 2))
    
    for epoch in range(config.num_epochs):
        print(f"-------------------------Starting Training Epoch {epoch+1}/{config.num_epochs} epochs-------------------------")
        model.train()
        epoch_loss = 0.0
        batch = 0
        
        train_loader, validation_loader = DataFunctions.split_dataset(config, data_loader)
        for videos, targets in train_loader:  
            batch += 1
            print(f"-------------------------Starting Batch {batch}/{int(len(train_loader.dataset)/config.batch_size)} batches-------------------------", end="\r")
            # TODO: Wil ik nog iets met deze outputs doen? Anders hoef ik ze niet terug te krijgen
            batch_loss, _ = train_batch(config, scaler, videos, targets, model, optimizer, loss_function)
            epoch_loss += batch_loss.item()
                                    
        print(f"-------------------------Finished training batches-------------------------") 
        
        # Validate the performance of the model
        val_f1_scores[epoch] = test_performance(config, model, validation_loader, loss_function, "validation")
        
        # TODO: Save the (best) model
        
    return

def train_batch(config, scaler, videos, targets, model, optimizer, loss_function):
    with amp.autocast(enabled=config.automatic_mixed_precision):
        # Model inference
        videos, targets = videos.to(config.device), targets.to(config.device)
        
        # Normalize videos
        normalized_videos = DataFunctions.normalize_videos(config, videos)
        raw_outputs, outputs = model(normalized_videos)
        
        # Compute loss, update model
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(raw_outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    return loss, outputs

def test_performance(config, model, data_loader, loss_function, stage):
    print(f"-------------------------Start {stage}-------------------------")
    model.eval()
    
    with torch.no_grad():
        total_loss = 0.0
        total_violence = 0
        total_neutral = 0
        batch = 0
        test_outputs = torch.empty((0)).to(config.device)
        test_targets = torch.empty((0)).to(config.device)
        
        for videos, targets in data_loader:
            batch += 1
            print(f"-------------------------Starting Batch {batch}/{int(len(data_loader.dataset)/config.batch_size)} batches-------------------------", end="\r")

            # Store example for printing while on CPU and non-normalized
            example_video = videos[0]
            
            # Model inference 
            videos, targets = videos.to(config.device), targets.to(config.device)
            
            # Normalize videos
            normalized_videos = DataFunctions.normalize_videos(config, videos)
            raw_outputs, outputs = model(normalized_videos)
            
            # Transform the outputs into the binary format where violence is either 0 or 1. Solely for logging purposes
            binary_outputs = outputs > 0.5
            batch_violence = (binary_outputs==1).sum()
            batch_neutral = outputs.shape[0] - batch_violence
            total_violence += batch_violence.item()
            total_neutral += batch_neutral.item()
            
            # Log example to WandB
            LogFunctions.log_video_example(config, example_video, targets[0], outputs[0], stage)
            
            # Compute batch loss and add to total loss
            batch_loss = loss_function(raw_outputs, targets).item()
            total_loss += batch_loss
            
            # Store all outputs and targets
            test_outputs = torch.cat((test_outputs, outputs), dim=0)
            test_targets = torch.cat((test_targets, targets.to(config.device)), dim=0)
            
        print(f"-------------------------Finished {stage} batches-------------------------")
        
        print(f"{total_violence = }")
        print(f"{total_neutral = }")
        
        # Compute and log metrics
        num_batches = len(data_loader.dataset) // config.batch_size
        mean_loss = total_loss / (num_batches * config.batch_size)
        tn, fn, fp, tp = DataFunctions.confusion_matrix(config, test_outputs, test_targets, stage)
        accuracy, fn_rate, fp_rate, _, f1_score, f2_score, _ = DataFunctions.metrics(tn, fn, fp, tp)
        # f1_score = metrics.f1_score(test_targets.to("cpu").numpy(), test_outputs.to("cpu").numpy()) 
        LogFunctions.log_violence_metrics(config, mean_loss, accuracy, fn_rate, fp_rate, f1_score, f2_score, stage)

    return f1_score
    
    
def make(config):
    print(f"{config.sampletype = }")
    
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    print("Creating data loaders")
    train_loader, test_loader = DataFunctions.load_violence_data(config)
    
    model = MyModels.I3DViolenceModel(config).to(config.device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.WBCEweight])).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scaler = getGradScaler(config)
    
    return model, scaler, loss_function, optimizer, train_loader, test_loader


def violence_pipeline(hyperparameters):
    # Give the run a name
    hyperparameters.run_name = f"num_epochs:{hyperparameters.num_epochs}_LR:{hyperparameters.lr}_WBCEweight:{hyperparameters.WBCEweight}"

    with wandb.init(mode="online", project="violence-model", config=hyperparameters):
        config = wandb.config
        wandb.run.name = config.run_name
        
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
    violence_pipeline(config)
