#MSc Thesis Marieke Kopmels
import argparse
from types import SimpleNamespace

import time
import Models.MyModels as MyModels
import Logging.LogFunctions as LogFunctions
import Models.ModelFunctions as ModelFunctions
import Data.DataFunctions as DataFunctions
import torch
import wandb
import torch.nn as nn
from torch import optim
import numpy as np
import warnings

from torch.cuda import amp

# Default parameters
# Size of VisuAAL dataset: Train=44783, Test=1157
# Size of Augmented Pratheepan dataset: Train=300
# Size of LargeCombined Train=6528, Validation=384, Test=768
# Size of LargeCombinedAugmented Train=32640

default_config = SimpleNamespace(
    machine = "OTS5",
    device = torch.device("cuda"),
    dims = 224,
    num_channels = 3,
    
    pretrained = True,
    lr = 0.00001, 
    colour_space = "BGR",
    optimizer = "AdamW",
    weight_decay = 0.01,
    
    num_workers = 4,
    num_epochs = 5,
    batch_size = 32, 
    split = 0.95,
    
    # train_size = 44783,       #VisuAAL
    # test_size = 768,          #LargeCombinedTest
    
    # train_size = 6912,        #LargeCombined
    # test_size = 768,          #LargeCombinedTest
    
    train_size = 34560,         #LargeCombinedAugmented
    test_size = 768,            #LargeCombinedTest
    
    # train_size = 128,         #Smaller part
    # test_size = 64,           #Smaller part

    automatic_mixed_precision = True,
    
    early_stopping = False,
    patience = 2,               #The number of epochs the model is allowed not to improve
    min_improvement = 0.05,     #Minimal improvement needed for early stopping 
    
    data_path = "/home/oddity/marieke/Datasets",
    # trainset = "VisuAAL", 
    trainset = "LargeCombinedAugmented", 
    testset = "LargeCombinedTest",
    
    model_path = "/home/oddity/marieke/Output/Models",
    model_name = "test_pretrained.pt",
    run_name = "",
    architecture = "UNet"
)


def parse_args():
    "Overriding default arguments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--num_epochs', type=int, default=default_config.num_epochs, help='number of epochs')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--train_size', type=int, default=default_config.train_size, help='train size')
    argparser.add_argument('--test_size', type=int, default=default_config.test_size, help='test size')
    argparser.add_argument('--split', type=float, default=default_config.split, help='split used for train/validation, defines the size of the train set')
    argparser.add_argument('--lr', type=float, default=default_config.lr, help='learning rate')
    argparser.add_argument('--weight_decay', type=float, default=default_config.weight_decay, help='weight decay for Weighted Adam optimizer')
    argparser.add_argument('--colour_space', type=str, default=default_config.colour_space, help='colour space')
    argparser.add_argument('--model_name', type=str, default=default_config.model_name, help='name of the model to be loaded')    
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

""" Returns the optimizer based on the configurations
"""
def get_optimizer(config, model):
    if config.optimizer == "SGD":
        return optim.SGD(model.parameters(), lr=config.lr, momentum=0.999)
    elif config.optimizer == "Adam":
        return optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == "AdamW":
        return optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == "RMSprop":
        return optim.RMSprop(model.parameters(),lr=config.lr, momentum=config.momentum) 
    else:
        warnings.warn("No matching optimizer found! Used default Adam")
        print(f"Current {config.optimizer=}")
        return optim.Adam(model.parameters(), lr=config.lr)
    
""" Returns dataloaders, model, loss function and optimizer.
"""
def make(config):
    # Fetch data
    start_time = time.time()
    train_loader, test_loader = DataFunctions.load_image_data(config)
    end_time = time.time() - start_time
    print(f"Loading of data done in %.2d seconds" % end_time)
    
    # Create or load the model
    if config.pretrained:
        model = ModelFunctions.load_model(config, config.model_name)
    else: 
        model = MyModels.UNET(config).to(config.device)

    # Define loss function and optimizer.
    # Set weight for positive elements (skin pixels) to be 9 times larger than negative elements (background pixels)
    # This is done to compensate for the imbalance in the skin:background pixel ratio
    weight = 9
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight])).to(config.device)
    optimizer = get_optimizer(config, model)
    
    return model, train_loader, test_loader, loss_function, optimizer

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

""" Trains the passed model, tests it performance after each epoch on the validation set. Prints and logs the results to WandB.
"""
def train(config, model, data_loader, loss_function, optimizer):
    if config.automatic_mixed_precision:
        if config.machine == "TS2" or config.machine == "OS4" or config.machine == "OTS5":
            scaler = amp.GradScaler(enabled=config.automatic_mixed_precision)
        else: 
            Warning("Machine not approved to use for Automatic Mixed Precision, AMP should be turned off.")
    else:
        scaler = None
        
    val_IoU_scores = np.zeros(config.num_epochs)
    patience_counter = 0
    
    # Split the data into a train and validation part
    train_loader, validation_loader = DataFunctions.split_dataset(config, data_loader)
        
    print("-------------------------Start Training-------------------------")
    for epoch in range(config.num_epochs):
                
        print(f"-------------------------Starting Training Epoch {epoch+1}/{config.num_epochs} epochs-------------------------")
        model.train()
        epoch_loss = 0.0
        batch = 0
        for images, targets in train_loader:
            
            batch += 1
            print(f"-------------------------Starting Batch {batch}/{int(config.train_size/config.batch_size)} batches-------------------------", end="\r")
            batch_loss, batch_outputs = train_batch(config, scaler, images, targets, model, optimizer, loss_function)
            epoch_loss += batch_loss.item()
            
        print(f"-------------------------Finished training batches-------------------------") 
        
         # Test the performance with validation data
        val_IoU_scores[epoch] = test_performance(config, model, validation_loader, loss_function, "validation")        
        
        # Save the model of this epoch, overwrite the best model if it has a better IoU score than all previous model
        if epoch == 0 or val_IoU_scores[epoch] > max(val_IoU_scores[:epoch]):
            ModelFunctions.save_model(config, model, epoch+1, best=True)
        else:
            ModelFunctions.save_model(config, model, epoch+1)
        
        # Early stopping
        if config.early_stopping:
            early_stop, patience_counter = early_stopping(config, epoch, patience_counter, val_IoU_scores)
            if early_stop:
                # If stopped early, retrieve the best model for further computations
                model = ModelFunctions.load_model(config, model_from_current_run=True)
                return model
            
    return model
            
""" Performs training for one batch of datapoints. Returns the true/false positive/negative metrics. 
"""
def train_batch(config, scaler, images, targets, model, optimizer, loss_function):
    with amp.autocast(enabled=config.automatic_mixed_precision):
        # Model inference
        images, targets = images.to(config.device), targets.to(config.device)
        normalized_images = DataFunctions.normalize_images(config, images)
        raw_outputs, outputs = model(normalized_images)
        
        # Compute loss, update model
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(raw_outputs, targets)
        if config.automatic_mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
    return loss, outputs

""" Tests the performance of the passed model on the data that is passed, either in validation or in test stage.
    Returns the IoU, the metric that is used for early stopping.
"""
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
            example_image = np.array(images[0].permute(1,2,0), dtype=np.uint8)

            # Model inference 
            images, targets = images.to(config.device), targets.to(config.device)
            normalized_images = DataFunctions.normalize_images(config, images)
            raw_batch_outputs, batch_outputs = model(normalized_images)
            
            # Log example to WandB
            LogFunctions.log_example(config, example_image, targets[0], batch_outputs[0], stage)
            
            # Compute batch loss
            batch_loss = loss_function(raw_batch_outputs, targets).item()
            
            # Store all outputs and targets
            test_outputs = torch.cat((test_outputs, batch_outputs), dim=0)
            test_targets = torch.cat((test_targets, targets.to(config.device)), dim=0)
            
            total_loss += batch_loss
            
        print(f"-------------------------Finished {stage} batches-------------------------")
        
        # Compute and log metrics
        epoch_tn, epoch_fn, epoch_fp, epoch_tp = DataFunctions.confusion_matrix(config, test_outputs, test_targets, stage)
        # Drop_last=False for both the validation and test dataloader, so the size of the dataset is used to compute the mean
        mean_loss = total_loss / len(data_loader.dataset)
        IoU = LogFunctions.log_metrics(config, mean_loss, epoch_tn, epoch_fn, epoch_fp, epoch_tp, stage)
        
    return IoU
        

""" Runs the whole pipeline of creating, training and testing a model
"""
def model_pipeline(hyperparameters):
    # Give the run a name
    hyperparameters.run_name = f"{hyperparameters.machine}_Colourspace:{hyperparameters.colour_space}_num_epochs:{hyperparameters.num_epochs}_batch_size:{hyperparameters.batch_size}_LR:{hyperparameters.lr}_Pretrained:{hyperparameters.pretrained}_Trainset:{hyperparameters.trainset}"
    # hyperparameters.run_name = f"{hyperparameters.machine}_{hyperparameters.colour_space}"
    
    # Start wandb
    with wandb.init(mode="online", project="skin_segmentation", config=hyperparameters): 
        # Set hyperparameters
        config = wandb.config
        wandb.run.name = config.run_name

        # Create model, data loaders, loss function and optimizer
        # Note that the data in the train loader is split into a train and validation dataset during training
        model, train_loader, test_loader, loss_function, optimizer = make(config)
        wandb.watch(model, log="all", log_freq=1)
            
        # In case of a pretrained model, test the performance on the test set before training to provide baseline information 
        if config.pretrained: 
            print("-------------------------Loaded a pretrained model, producing a test baseline-------------------------")
            test_performance(config, model, test_loader, loss_function, "test")
            
        # Train the model, incl. validation
        best_model = train(config, model, train_loader, loss_function, optimizer)

        # Test the best model's performance
        test_performance(config, best_model, test_loader, loss_function, "test")

    return

if __name__ == '__main__':
    start_time = time.time()
    parse_args()
    model_pipeline(default_config)
    print(f"Runtime = {time.time() - start_time}")