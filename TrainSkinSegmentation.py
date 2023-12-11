#MSc Thesis Marieke Kopmels
import argparse
from types import SimpleNamespace

import time
import MyModels
import LogFunctions
import DataFunctions
import torch
import wandb
import torch.nn as nn
from torch import optim
import numpy as np
import warnings

from torch.cuda import amp

# Options for loss function
# TODO: eruit halen, gewoon eentje kiezen 
loss_dictionary = {
    # "IoU": LossFunctions.IoULoss(),
    # "Focal": LossFunctions.FocalLoss(),
    # "WBCE": nn.BCEWithLogitsLoss(),
    "WBCE_9": nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9])),
    # "BCE": nn.BCELoss(),
}

# Default parameters
# Size of VisuAAL dataset: Train=44783, Test=1157
# Size of Augmented Pratheepan dataset: Train=300
# Size of LargeCombined Train=6528, Validation=384, Test=768
# Size of LargeCombinedAugmented Train=32640
default_config = SimpleNamespace(
    machine = "TS2",
    device = torch.device("cuda"),
    log = True,
    num_workers = 4,
    dims = 224,
    num_epochs = 20,
    batch_size = 16, 
    train_size = 44783,       #VisuAAL
    # train_size = 6528,        #LargeCombined
    # train_size = 32640,       #LargeCombinedAugmented
    # train_size = 128,          #Smaller part
    validation_size = 1157,    #VisuAAL
    # validation_size = 384,    #LargeCombined
    # validation_size = 32,     #Smaller part
    test_size = 768,          #LargeCombined
    cm_train = False,
    cm_parts = 16,
    lr = 0.00001, 
    pretrained = False,
    automatic_mixed_precision = False,
    patience = 2,               #The number of epochs the model is allowed not to improve
    min_improvement = 0.05,     #Minimal improvement needed for early stopping
    colour_space = "BGR",
    loss_function = "WBCE_9",
    optimizer = "AdamW", 
    dataset = "VisuAAL", 
    # dataset = "LargeCombinedAugmented",
    testset = "LargeCombined",
    data_path = "/home/oddity/marieke/Datasets/VisuAAL",
    # data_path = "/home/oddity/marieke/Datasets/LargeCombinedAugmentedDataset",
    testdata_path = "/home/oddity/marieke/Datasets/LargeCombinedDataset",
    model_path = "/home/oddity/marieke/Output/Models",
    architecture = "UNet"

    # machine = "Mac",
    # device = torch.device("mps"),
    # log = True,
    # num_workers = 1,
    # dims = 224,
    # num_epochs = 5,
    # batch_size = 16, 
    # # train_size = 44783,               #VisuAAL
    # # train_size = 6528,                #LargeCombined
    # # train_size = 32640,               #LargeCombinedAugmented
    # train_size = 100,                 #Small test
    # # validation_size = 128,            #VisuAAL
    # # validation_size = 384,            #LargeCombined
    # validation_size = 32,             #Small test
    # # test_size = 1024,                 #VisuAAL
    # test_size = 768,                  #LargeCombined
    # cm_train = False,
    # cm_parts = 1,
    # lr = 0.00001, 
    # # momentum = 0.99, 
    # pretrained = False,
    # automatic_mixed_precision = False, #Not possible on mps or cpu
    # patience = 3,               #The number of epochs the model is allowed not to improve
    # min_improvement = 0.05,     #Minimal improvement needed for early stopping
    # colour_space = "BGR",
    # loss_function = "WBCE_9",
    # optimizer = "AdamW", 
    # dataset = "LargeCombinedAugmented", 
    # testset = "LargeCombined",
    # data_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/LargeCombinedAugmentedDataset",
    # testdata_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/LargeCombinedDataset",
    # model_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Thesis/Models",
    # architecture = "UNet"
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
        return optim.SGD(model.parameters(), lr=config.lr, momentum=0.999)
    elif config.optimizer == "Adam":
        return optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == "AdamW":
        return optim.AdamW(model.parameters(), lr=config.lr)
    # elif config.optimizer == "RMSprop":
    #     return optim.RMSprop(model.parameters(),lr=config.lr, momentum=config.momentum) 
    else:
        warnings.warn("No matching optimizer found! Used default Adam")
        print(f"Current config.optimizer = {config.optimizer}")
        return optim.Adam(model.parameters(), lr=config.lr)
    
""" Returns dataloaders, model, loss function and optimizer.
"""
def make(config):
    # Fetch data
    start_time = time.time()
    train_loader, validation_loader, test_loader = DataFunctions.load_image_data(config)
    end_time = time.time() - start_time
    print(f"Loading of data done in %.2d seconds" % end_time)
    
    # Make the model
    if config.pretrained:
        # path = config.model_path + "/Dataset:VisuAAL_Val_Testset:LargeCombined/final.pt"
        path = config.model_path + "/pretrained.pt"
        model = torch.load(path).to(config.device)
    else: 
        model = MyModels.UNET(config).to(config.device)

    # Define loss function and optimizer
    loss_function = loss_dictionary[config.loss_function].to(config.device)
    optimizer = get_optimizer(config, model)
    
    return model, train_loader, validation_loader, test_loader, loss_function, optimizer

""" Decides wether or not the validation score of the model improves enough. 
    If not, it will retrun early_stop=True which will stop the training process.
"""
def early_stopping(config, epoch, patience_counter, val_IoU_scores):
    early_stop = False
    # print(f"at epoch {epoch}, val_IoU_scores: {val_IoU_scores}, patience_counter: {patience_counter}")
    if epoch > 0:
        if val_IoU_scores[epoch] <= val_IoU_scores[epoch-1]*(1+config.min_improvement):
            patience_counter += 1
            if patience_counter > config.patience:
                print(f"Not enough improvement, should be at least {config.min_improvement*100}% better than the last epoch, training stopped early.")
                early_stop = True
        else: 
            patience_counter = 0

    return early_stop, patience_counter

""" Trains the passed model, tests it performance after each epoch on the validation set. Prints and logs the results to WandB.
"""
def train(config, model, train_loader, validation_loader, loss_function, optimizer):
    if config.machine == "TS2" and config.automatic_mixed_precision:
        scaler = amp.GradScaler(enabled=config.automatic_mixed_precision)
    else:
        scaler = None
        
    val_IoU_scores = np.zeros(config.num_epochs)
    patience_counter = 0
        
    if config.pretrained: 
        print("-------------------------Loaded a pretrained model, producing validation baseline-------------------------")
        test_performance(config, model, validation_loader, loss_function, "validation")
        
    print("-------------------------Start Training-------------------------")
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
            # batch_loss, batch_outputs = train_batch(config, images, targets, model, optimizer, loss_function)
            batch_loss, batch_outputs = train_batch(config, scaler, images, targets, model, optimizer, loss_function)
            epoch_loss += batch_loss.item()
            
            if config.cm_train:
                epoch_outputs = torch.cat((epoch_outputs, batch_outputs), dim=0)
                epoch_targets = torch.cat((epoch_targets, targets.to(config.device)), dim=0)

        print(f"-------------------------Finished training batches-------------------------") 
        
        # Keep track of training epoch stats, or skip for sake of efficiency
        if config.cm_train:
            epoch_tn, epoch_fn, epoch_fp, epoch_tp = DataFunctions.confusion_matrix(config, epoch_outputs, epoch_targets, "train")
            # drop_last=True in the dataloader, so we compute the amount of batches first
            num_batches = len(train_loader.dataset) // config.batch_size
            mean_loss = epoch_loss / (num_batches * config.batch_size)
            _ = LogFunctions.log_metrics(config, mean_loss, epoch_tn, epoch_fn, epoch_fp, epoch_tp, "train")
        
        # Save the model
        LogFunctions.save_model(config, model, epoch+1)
        
        # Test the performance with validation data
        val_IoU_scores[epoch] = test_performance(config, model, validation_loader, loss_function, "validation")
        
        # Early stopping
        early_stop, patience_counter = early_stopping(config, epoch, patience_counter, val_IoU_scores)
        if early_stop:
            break
        
""" Performs training for one batch of datapoints. Returns the true/false positive/negative metrics. 
"""
def train_batch(config, scaler, images, targets, model, optimizer, loss_function):
# def train_batch(config, images, targets, model, optimizer, loss_function):
    if config.automatic_mixed_precision:
        with amp.autocast(enabled=config.automatic_mixed_precision):
            # Model inference
            images, targets = images.to(config.device), targets.to(config.device)
            normalized_images = DataFunctions.normalize_images(config, images)
            outputs = model(normalized_images)
            
            # Compute loss, update model
            loss = loss_function(outputs, targets)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        # print("Should not enable automatic mixed precision, not implemented.")
    else:
        # Model inference
        images, targets = images.to(config.device), targets.to(config.device)
        normalized_images = DataFunctions.normalize_images(config, images)
        outputs = model(normalized_images)
        
        # Compute loss, update model
        optimizer.zero_grad(set_to_none=True)
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
            normalized_images = DataFunctions.normalize_images(config, images)
            batch_outputs = model(normalized_images)
            
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
        num_batches = len(data_loader.dataset) // config.batch_size
        mean_loss = total_loss / (num_batches * config.batch_size)
        IoU = LogFunctions.log_metrics(config, mean_loss, epoch_tn, epoch_fn, epoch_fp, epoch_tp, stage)
        
    return IoU
        

""" Runs the whole pipeline of creating, training and testing a model
"""
def model_pipeline(hyperparameters):
    # Start wandb
    with wandb.init(project="skin_segmentation", config=hyperparameters): #mode="disabled", 
        # Set hyperparameters
        config = wandb.config
        run_name = f"{config.machine}_Pretrained:{config.pretrained}_Dataset:{config.dataset}_Testset:{config.testset}"
        wandb.run.name = run_name

        # TODO: Aanzetten en testen
        # LogFunctions.init_device(config)

        # Create model, data loaders, loss function and optimizer
        model, train_loader, validation_loader, test_loader, loss_function, optimizer = make(config)
        if config.log:
            wandb.watch(model, log="all", log_freq=1)
        else: 
            wandb.watch(model, log=None, log_freq=1)
            
        # Test the performance on the test set before training (in case of a pretrained model)
        if config.pretrained: 
            print("-------------------------Loaded a pretrained model, producing test baseline-------------------------")
            test_performance(config, model, test_loader, loss_function, "test")

        # Train the model, incl. validation
        train(config, model, train_loader, validation_loader, loss_function, optimizer)

        # Test the models performance
        test_performance(config, model, test_loader, loss_function, "test")
        
        # Store the final model
        LogFunctions.save_model(config, model, 0, final=True)

    return

if __name__ == '__main__':
    start_time = time.time()
    parse_args()
    model_pipeline(default_config)
    print(f"Runtime = {time.time() - start_time}")