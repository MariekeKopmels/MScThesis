# Main for training the I3D violence model
import Config.ConfigFunctions as ConfigFunctions
import Data.DataFunctions as DataFunctions
import Models.MyModels as MyModels
import Models.LossFunctions as LossFunctions
import Models.ModelFunctions as ModelFunctions
import Logging.LogFunctions as LogFunctions
import wandb
import torch
import random
import math
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


""" Decides whether or not the validation score of the model improves enough. 
    If not, it will return early_stop=True which will stop the training process.
"""
def early_stopping(config, epoch, patience_counter, val_IoU_scores):
    early_stop = False
    if epoch > 0:
        if val_IoU_scores[epoch] <= val_IoU_scores[epoch-1]*(1+config.min_improvement):
            patience_counter += 1
            if patience_counter > config.patience:
                print(f"Not enough improvement, should be at least {config.min_improvement*100}% better than the last epoch, so training is stopped early.")
                early_stop = True
        else: 
            patience_counter = 0

    return early_stop, patience_counter

    
""" Trains the passed model"""
def train(config, model, scaler, loss_function, optimizer, data_list):
    # Keep track of score and patience for early stopping
    val_mse_scores = np.zeros((config.num_epochs, 2))
    patience_counter = 0
    
    train_list, validation_list = DataFunctions.split_video_list(config, data_list)
    
    for epoch in range(config.num_epochs):
        print(f"-------------------------Starting Training Epoch {epoch+1}/{config.num_epochs} epochs-------------------------")
        model.train()
        epoch_loss = 0.0
        batch = 0
        
        # Shuffle the list to ensure every training epoch is different
        random.shuffle(train_list)
        num_batches = math.ceil(len(train_list) / config.batch_size)
                
        for batch in range(num_batches):
            videos, targets = DataFunctions.load_video_data(config, train_list, batch)            
            print(f"-------------------------Starting Batch {batch}/{num_batches} batches-------------------------", end="\r")
            # TODO: Wil ik nog iets met deze outputs doen? Anders hoef ik ze niet terug te krijgen
            batch_loss, _ = train_batch(config, scaler, videos, targets, model, optimizer, loss_function)
            epoch_loss += batch_loss.item()
                                    
        print(f"-------------------------Finished training batches-------------------------") 
        
        # Validate the performance of the model
        val_mse_scores[epoch] = test_performance(config, model, validation_list, loss_function, "validation")
        
        # TODO: Save the (best) model        
        # Save the model of this epoch, overwrite the best model if it has a better IoU score than all previous model
        if epoch == 0 or val_mse_scores[epoch] < min(val_mse_scores[:epoch]):
            ModelFunctions.save_model(config, model, epoch+1, best=True)
        else:
            ModelFunctions.save_model(config, model, epoch+1)
        
        # Early stopping
        if config.early_stopping:
            early_stop, patience_counter = early_stopping(config, epoch, patience_counter, val_mse_scores)
            if early_stop:
                # If stopped early, retrieve the best model for further computations
                model = ModelFunctions.load_model(config, model_from_current_run=True)
                return model
        
    return

def train_batch(config, scaler, videos, targets, model, optimizer, loss_function):
    with amp.autocast(enabled=config.automatic_mixed_precision):
        # Model inference
        videos, targets = videos.to(config.device), targets.to(config.device)
        
        # Normalize videos
        normalized_videos = DataFunctions.normalize_videos(config, videos)
        outputs = model(normalized_videos)
        
        # Compute loss, update model
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    return loss, outputs

def test_performance(config, model, data_list, loss_function, stage):
    print(f"-------------------------Start {stage}-------------------------")
    model.eval()
    
    with torch.no_grad():
        total_loss = 0.0
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
            outputs = model(normalized_videos)
            
            # Log example to WandB
            LogFunctions.log_video_example(config, example_video, targets[0], outputs[0], stage)
            
            # Compute batch loss and add to total loss
            batch_loss = loss_function(outputs, targets).item()
            total_loss += batch_loss
            
            # Store all outputs and targets
            test_outputs = torch.cat((test_outputs, outputs), dim=0)
            test_targets = torch.cat((test_targets, targets.to(config.device)), dim=0)
            
        print(f"-------------------------Finished {stage} batches-------------------------")

        # Compute and log metrics
        mean_loss = total_loss / (num_batches * config.batch_size)
        mae, mse = DataFunctions.regression_metrics(test_outputs, test_targets)
        LogFunctions.log_skin_tone_metrics(config, mean_loss, mae, mse, stage)

    return mse
    
    
def make(config):    
    print("Initializing...")
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    train_list, test_list = DataFunctions.load_skin_tone_video_list(config)
    
    model = MyModels.SkinToneModel(config).to(config.device)    
    loss_weights = ConfigFunctions.toArray(config, config.WMSE_weights)
    loss_function = LossFunctions.WeightedMSELoss(config, loss_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scaler = getGradScaler(config)
    
    return model, scaler, loss_function, optimizer, train_list, test_list


def skin_tone_pipeline(hyperparameters):
    with wandb.init(mode="online", project="skin-tone-model", config=hyperparameters):
        config = wandb.config
        
        # Give the run a name
        config.run_name = f"LR:{config.lr}_num_epochs:{config.num_epochs}_batch_size:{config.batch_size}"
        wandb.run.name = config.run_name
        
        # Create model, data loaders, loss function and optimizer
        model, scaler, loss_function, optimizer, train_list, test_list = make(config)
        
        # Train the model, incl. validation
        train(config, model, scaler, loss_function, optimizer, train_list)
        
        # Test the final model's performance
        test_performance(config, model, test_list, loss_function, "test")
        
    return

if __name__ == '__main__':
    args = ConfigFunctions.parse_args()
    config = ConfigFunctions.load_config(args.config)
    skin_tone_pipeline(config)
