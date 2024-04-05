# Main for training the I3D violence model
import Config.ConfigFunctions as ConfigFunctions
import Data.DataFunctions as DataFunctions
import Data.AugmentationFunctions as AugmentationFunctions
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


""" Print information about the class distribution of the data, for both the training and test data.
"""
def print_data_stats(config, train_list, test_list):
    # Print some stats regarding the train and test data
    train_gt_list = [video + ".json" for video in train_list]
    test_gt_list = [video + ".json" for video in test_list]
    train_gts = DataFunctions.load_video_gts(config, train_gt_list, config.data_path + "/skin_tone_labels")
    test_gts = DataFunctions.load_video_gts(config, test_gt_list, config.data_path + "/skin_tone_labels")
    unique_values, counts = torch.unique(train_gts, return_counts=True)
    print("Train")
    for value, count in zip(unique_values, counts):
        print(f"Value: {value}, Count: {count}")
    unique_values, counts = torch.unique(test_gts, return_counts=True)
    print("Test")
    for value, count in zip(unique_values, counts):
        print(f"Value: {value}, Count: {count}")
    return

    
""" Trains the passed model
"""
def train(config, model, scaler, loss_function, optimizer, data_list):
    # Keep track of score and patience for early stopping
    val_mse_scores = np.zeros((config.num_epochs))
    patience_counter = 0
    
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
        val_mse_scores[epoch] = test_performance(config, model, validation_list, loss_function, "validation")
        
        # Save the model of this epoch, overwrite the best model if it has a lower mse score than all previous models
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
            
    # After training, return the best model.
    model = ModelFunctions.load_model(config, model_from_current_run=True)
        
    return

""" Performs a training step given the passed data.
"""
def train_batch(config, scaler, videos, targets, model, optimizer, loss_function):
    with amp.autocast(enabled=config.automatic_mixed_precision):
        # Model inference
        videos, targets = videos.to(config.device), targets.to(config.device)
        
        # Normalize videos
        normalized_videos = DataFunctions.normalize_videos(config, videos)
        # TODO: finish video augmentation
        # augmented_videos = AugmentationFunctions.augment_videos(config, normalized_videos)
        outputs = model(normalized_videos)
        
        # Compute loss, update model
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(outputs, targets)
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
    
""" Initializes the model, its loss function, optimizer and scaler and loads the train and test lists. 
"""
def make(config):    
    print("Initializing...")
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    
    train_list, test_list = DataFunctions.load_skin_tone_video_list(config)
    
    print_data_stats(config, train_list, test_list)
    if config.architecture == "I3D_SkinTone":
        model = MyModels.I3DSkintoneModel(config).to(config.device)
    elif config.architecture == "ResNet_SkinTone":
        model = MyModels.ResNetSkinToneModel(config).to(config.device)
    loss_weights = ConfigFunctions.toArray(config, config.WMSE_weights)
    loss_function = LossFunctions.WeightedMSELoss(config, loss_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scaler = getGradScaler(config)
    
    return model, scaler, loss_function, optimizer, train_list, test_list

""" Runs the complete pipeline of training and testing the Skin tone prediction model.
"""
def skin_tone_pipeline(hyperparameters):
    with wandb.init(mode="disabled", project="skin-tone-model", config=hyperparameters):
        config = wandb.config
        
        # Give the run a name
        config.run_name = f"{config.sampletype}_LR:{config.lr}_{config.colour_space}"
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
    skin_tone_pipeline(config)
