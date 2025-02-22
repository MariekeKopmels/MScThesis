# MSc Thesis Marieke Kopmels
# Violence detection model, based on I3D

import numpy as np
import math
import random
import wandb
import torch
import torch.nn as nn
from torch import optim
from torch.cuda import amp

import Config.ConfigFunctions as ConfigFunctions
import Data.DataFunctions as DataFunctions
import Models.MyModels as MyModels
import Models.ModelFunctions as ModelFunctions
import Logging.LogFunctions as LogFunctions


"""Returns the grad scaler in case of automatic mixed precision.
"""
def getGradScaler(config):
    if config.automatic_mixed_precision:
        if config.machine == "OTS5":
            return amp.GradScaler(enabled=config.automatic_mixed_precision)
        else: 
            Warning("Machine not approved to use for Automatic Mixed Precision, so AMP is turned off.")
    return None
    
""" Trains the passed model. Returns the model with the highest validation f1-score.
"""
def train(config, model, scaler, loss_function, optimizer, data_list):
    # Keep track of performance throughout the training process
    val_f1_scores = np.zeros((config.num_epochs))

    # Split the data into a train and validation set
    train_list, validation_list = DataFunctions.split_video_list(config, data_list)
    
    for epoch in range(config.num_epochs):
        print(f"-------------------------Starting Training Epoch {epoch+1}/{config.num_epochs} epochs-------------------------")
        model.train()
        batch = 0
        
        # Shuffle the list to ensure every training epoch is different
        random.shuffle(train_list)
        num_batches = math.ceil(len(train_list) / config.batch_size)
        
        # Train the model with the training data
        for batch in range(num_batches):
            videos, targets = DataFunctions.load_video_data(config, train_list, batch)            
            print(f"-------------------------Starting Batch {batch}/{num_batches} batches-------------------------", end="\r")
            train_batch(config, scaler, videos, targets, model, optimizer, loss_function)
                                    
        print(f"-------------------------Finished training batches-------------------------") 
        
        # Validate the performance of the model
        val_f1_scores[epoch] = test_performance(config, model, validation_list, loss_function, "validation")

        # Save the model of this epoch, overwrite the best model if it has a better f1-score than all previous model
        if epoch == 0 or val_f1_scores[epoch] > max(val_f1_scores[:epoch]):
            ModelFunctions.save_model(config, model, epoch+1, best=True)
        else:
            ModelFunctions.save_model(config, model, epoch+1)

    # After training, return the model with the highest validation f1-score.
    model = ModelFunctions.load_model(config, model_from_current_run=True)
            
    return model

""" Performs a training step given the passed data.
"""
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

        return
     
""" Tests the performance of the passed model, using test data. Logs the results to Weights and Biases.
"""
def test_performance(config, model, data_list, loss_function, stage):
    print(f"-------------------------Start {stage}-------------------------")
    model.eval()
    
    with torch.no_grad():
        total_loss = 0.0
        total_violence = 0
        total_neutral = 0
        batch = 0
        test_outputs = torch.empty((0)).to(config.device)
        test_targets = torch.empty((0)).to(config.device)
        
        num_batches = math.ceil(len(data_list) / config.batch_size)
                
        for batch in range(num_batches):
            videos, targets = DataFunctions.load_video_data(config, data_list, batch)            
        
            print(f"-------------------------Starting Batch {batch}/{num_batches} batches-------------------------", end="\r")

            # Store example for printing while on CPU and non-normalized
            example_video = videos[0]
            
            # Model inference  
            videos, targets = videos.to(config.device), targets.to(config.device)
            
            # Normalize videos
            normalized_videos = DataFunctions.normalize_videos(config, videos)
            raw_outputs, outputs = model(normalized_videos)
            
            # Transform the outputs into the binary format where violence is either 0 or 1. Solely for logging purposes.
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
        
        # Compute and log metrics
        mean_loss = total_loss / (num_batches * config.batch_size)
        tn, fn, fp, tp = DataFunctions.confusion_matrix(config, test_outputs, test_targets, stage)
        accuracy, fn_rate, fp_rate, _, f1_score, f2_score, _ = DataFunctions.metrics(tn, fn, fp, tp)
        LogFunctions.log_violence_metrics(config, mean_loss, accuracy, fn_rate, fp_rate, f1_score, f2_score, stage, test_targets, test_outputs)

    return f1_score
    
""" Sets the seeds to ensure reproducability, initializes the model, its loss function, 
    optimizer and scaler and loads the train and test lists. 
"""
def make(config):    
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    print("Creating video lists")
    train_list, test_list = DataFunctions.load_video_list(config)
    model = MyModels.I3DViolenceModel(config).to(config.device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.WBCEweight])).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scaler = getGradScaler(config)
    
    return model, scaler, loss_function, optimizer, train_list, test_list

""" Runs the complete pipeline of training and testing the Violence detection model.
"""
def violence_pipeline(hyperparameters):
    with wandb.init(mode="disabled", project="violence-model", config=hyperparameters):
        config = wandb.config
        
        # Give the run a name
        config.run_name = f"{config.dataset_size}_{config.sampletype}_LR:{config.lr}_WBCE:{config.WBCEweight}_num_epochs:{config.num_epochs}_batch_size:{config.batch_size}"
        wandb.run.name = config.run_name
        
        # Create model, data loaders, loss function and optimizer
        model, scaler, loss_function, optimizer, train_list, test_list = make(config)
        
        # Train the model, incl. validation
        train(config, model, scaler, loss_function, optimizer, train_list)
        
        # Test the final model's performance
        test_performance(config, model, test_list, loss_function, "test")
        
    return

""" Parses the arguments, loads the configuration file and runs the Violence detection pipeline.
"""
if __name__ == '__main__':
    args = ConfigFunctions.parse_args()
    config = ConfigFunctions.load_config(args.config)
    violence_pipeline(config)
