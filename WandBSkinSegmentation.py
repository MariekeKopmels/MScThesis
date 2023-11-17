#MSc Thesis Marieke Kopmels
import argparse
from types import SimpleNamespace

import cv2
import MyModels
import LossFunctions
import DataFunctions
import torch
import wandb
import torch.nn as nn
from torch import optim
import numpy as np
import warnings

# TODO Voor morgen: bij de IoU gaat hij goed, maar bij de WBCE gaan de confusion matrices op hol, en komen er 
# Rare (X*224*224) getallen uit, en dan zou niet moeten kunnen vgm

# Options for loss function
loss_dictionary = {
    "IoU": LossFunctions.IoULoss(),
    "Focal": LossFunctions.FocalLoss(),
    # "CE": nn.CrossEntropyLoss(),
    "WBCE": nn.BCEWithLogitsLoss(),
    "WBCE_10": nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10])),
    "WBCE_20": nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20])),
    "BCE": nn.BCELoss(),
}

# Default parameters
# Size of dataset: Train=44783 , Test=1157
default_config = SimpleNamespace(
    num_epochs = 5,
    batch_size = 4, 
    train_size = 32, 
    test_size = 16,
    validation_size = 8,
    lr = 0.0001, 
    momentum = 0.99, 
    colour_space = "RGB",
    loss_function = "WBCE_10",
    optimizer = "Adam", 
    device = torch.device("mps"),
    dataset = "VisuAAL", 
    architecture = "UNet"
)

def parse_args():
    "Overriding default arguments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--num_epochs', type=int, default=default_config.num_epochs, help='number of epochs')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--train_size', type=int, default=default_config.train_size, help='trains size')
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
    train_loader, validation_loader, test_loader = DataFunctions.load_data(config, [], [])
    
    # Make the model
    # TODO: pass config to remove hardcoded NO_PIXELS = 224
    model = MyModels.UNET().to(config.device)

    # Define loss function and optimizer
    loss_function = loss_dictionary[config.loss_function].to(config.device)
    optimizer = get_optimizer(config, model)
    
    return model, train_loader, validation_loader, test_loader, loss_function, optimizer

""" Trains the passed model. 
"""
def train(config, model, train_loader, validation_loader, loss_function, optimizer):
    # Tell WandB to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, log="all", log_freq=1)
    
    for epoch in range(config.num_epochs):
        print(f"-------------------------Starting Epoch {epoch+1}/{config.num_epochs} epochs-------------------------")
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
        mean_loss = epoch_loss/config.train_size
        accuracy, fn_rate, fp_rate, sensitivity, f1_score, IoU = DataFunctions.metrics(epoch_tn, epoch_fn, epoch_fp, epoch_tp)
        train_epoch_log(config, mean_loss, accuracy, fn_rate, fp_rate, sensitivity, f1_score, IoU, epoch)
        
        validate(config, model, validation_loader, loss_function)

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
    
    tn, fn, fp, tp = DataFunctions.confusion_matrix(outputs, targets)
    return loss, tn, fn, fp, tp

""" Prints and logs intermediate results to WandB.
"""
def train_epoch_log(config, mean_loss, accuracy, fn_rate, fp_rate, sensitivity, f1_score, IoU, epoch):
    wandb.log({"epoch": epoch, "mean_loss": mean_loss, "accuracy": accuracy, "fn_rate": fn_rate, "fp_rate": fp_rate, "sensitivity": sensitivity, "f1_score": f1_score, "IoU": IoU})
    print(f"Train results\nMean loss: {mean_loss:.6f}, accuracy: {accuracy:.3f}, fn_rate: {fn_rate:.3f}, fp_rate:: {fp_rate:.3f}, sensitivity: {sensitivity:.3f}, f1-score: {f1_score:.3f}, IoU: {IoU:.3f}") 
    
def validate(config, model, validation_loader, loss_function):
    print(f"-------------------------Start validation-------------------------")
    model.eval()

# Run the model on validation examples
    with torch.no_grad():
        total_loss = 0.0
        total_tn, total_fn, total_fp, total_tp = 0, 0, 0, 0
        batch = 0
        for images, targets in validation_loader:
            batch += 1
            print(f"-------------------------Starting Batch {batch}/{int(config.validation_size/config.batch_size)} batches-------------------------")

            # Store example for printing while on CPU
            example_image = np.array(images[0].permute(1,2,0))
            
            images, targets = images.to(config.device), targets.to(config.device)
            images = images.float()
            targets = targets.float()
            outputs = model(images)
            
            # Log example to WandB
            log_example(config, example_image, targets[0], outputs[0], type="validation")
            
            batch_loss = loss_function(outputs, targets)
            batch_tn, batch_fn, batch_fp, batch_tp = DataFunctions.confusion_matrix(outputs, targets, test=False)
            
            total_loss += batch_loss.item()
            total_tn += batch_tn
            total_fn += batch_fn
            total_fp += batch_fp
            total_tp += batch_tp

        mean_validation_loss = total_loss/config.validation_size
        
        # TODO: Aangepast, dubbel checken of dit nu wel correct is
        # OUD:
        # tn, fn, fp, tp = DataFunctions.confusion_matrix(outputs, targets)
        # accuracy, fn_rate, fp_rate, sensitivity, f1_score, IoU = DataFunctions.metrics(tn, fn, fp, tp)
        # NIEUW:
        accuracy, fn_rate, fp_rate, sensitivity, f1_score, IoU = DataFunctions.metrics(total_tn, total_fn, total_fp, total_tp)

        validation_log(config, mean_validation_loss, accuracy, fn_rate, fp_rate, sensitivity, f1_score, IoU)


""" Prints and logs the validation results to WandB.
"""
def validation_log(config, mean_val_loss, val_accuracy, val_fn_rate, val_fp_rate, val_sensitivity, val_f1_score, val_IoU):
    wandb.log({"mean_val_loss": mean_val_loss, "_accuracy": val_accuracy, "val_fn_rate": val_fn_rate, "val_fp_rate": val_fp_rate, "val_sensitivity": val_sensitivity, "val_f1_score": val_f1_score, "val_IoU": val_IoU})
    print(f"Validation results\nMean loss: {mean_val_loss:.6f}, accuracy: {val_accuracy:.3f}, fn_rate: {val_fn_rate:.3f}, fp_rate:: {val_fp_rate:.3f}, sensitivity: {val_sensitivity:.3f}, f1-score: {val_f1_score:.3f}, IoU: {val_IoU:.3f}") 
    
    
""" Tests the performance of a model on the test set. Prints and logs the results to WandB.
"""
def test(config, model, test_loader, loss_function):
    print(f"-------------------------Start testing-------------------------")
    
    model.eval()
    
    # Run the model on test examples
    with torch.no_grad():
        total_loss = 0.0
        total_tn, total_fn, total_fp, total_tp = 0, 0, 0, 0
        for images, targets in test_loader:
            # Store example for printing while on CPU
            example_image = np.array(images[0].permute(1,2,0))
            
            images, targets = images.to(config.device), targets.to(config.device)
            images = images.float()
            targets = targets.float()
            outputs = model(images)
            print(f"torch.min(output): {torch.min(outputs[0])}, torch.max(output) {torch.max(outputs[0])}")
            
            # Log example to WandB
            log_example(config, example_image, targets[0], outputs[0], type="test")
            
            batch_loss = loss_function(outputs, targets)
            batch_tn, batch_fn, batch_fp, batch_tp = DataFunctions.confusion_matrix(outputs, targets, test=True)
            
            total_loss += batch_loss.item()
            total_tn += batch_tn
            total_fn += batch_fn
            total_fp += batch_fp
            total_tp += batch_tp

        mean_test_loss = total_loss/config.test_size
        
        # TODO: Aangepast, dubbel checken of dit nu wel correct is
        # OUD:
        # tn, fn, fp, tp = DataFunctions.confusion_matrix(outputs, targets)
        # accuracy, fn_rate, fp_rate, sensitivity, f1_score, IoU = DataFunctions.metrics(tn, fn, fp, tp)
        # NIEUW:
        accuracy, fn_rate, fp_rate, sensitivity, f1_score, IoU = DataFunctions.metrics(total_tn, total_fn, total_fp, total_tp)

        test_log(config, mean_test_loss, accuracy, fn_rate, fp_rate, sensitivity, f1_score, IoU)

""" Prints and logs the test results to WandB.
"""
def test_log(config, mean_test_loss, test_accuracy, test_fn_rate, test_fp_rate, test_sensitivity, test_f1_score, test_IoU):
    wandb.log({"mean_test_loss": mean_test_loss, "test_accuracy": test_accuracy, "test_fn_rate": test_fn_rate, "test_fp_rate": test_fp_rate, "test_sensitivity": test_sensitivity, "test_f1_score": test_f1_score, "test_IoU": test_IoU})
    print(f"Mean loss: {mean_test_loss:.6f}, accuracy: {test_accuracy:.3f}, fn_rate: {test_fn_rate:.3f}, fp_rate:: {test_fp_rate:.3f}, sensitivity: {test_sensitivity:.3f}, f1-score: {test_f1_score:.3f}, IoU: {test_IoU:.3f}") 
    
    # Save the model
    # TODO: Save the model, fix: raises errors
    # torch.onnx.export(model, "model.onnx")
    # wandb.save("model.onnx")
    
""" Logs test examples of input image, ground truth and model output.
"""
def log_example(config, example, target, output, type="validation"):
    # Change cannels from YCrCb or BGR to RGB
    if config.colour_space == "YCrCb": 
        # print("Converted YCrCb to RGB")
        example = cv2.cvtColor(example, cv2.COLOR_YCR_CB2RGB)
    elif config.colour_space == "RGB":
        # print("Converted BGR to RGB")
        example = cv2.cvtColor(example, cv2.COLOR_BGR2RGB)
    else:
        warnings.warn("No colour space found!")
        print(f"Current config.colour_space = {config.colour_space}")
      
    wandb.log({f"Input {type} image":[wandb.Image(example)]})
    wandb.log({f"Target {type} output": [wandb.Image(target)]})
    wandb.log({f"Model {type} greyscale output": [wandb.Image(output)]})
    bw_image = (output >= 0.5).float()
    wandb.log({f"Model {type} bw output": [wandb.Image(bw_image)]})

""" Runs the whole pipeline of creating, training and testing a model
"""
def model_pipeline(hyperparameters):
    # start wandb
    with wandb.init(mode="disabled", project="skin_segmentation", config=hyperparameters): #mode="disabled", 
        # set hyperparameters
        config = wandb.config
        run_name = f"{config.loss_function}_train_size:{config.train_size}"
        wandb.run.name = run_name

        # create model, data loaders, loss function and optimizer
        model, train_loader, validation_loader, test_loader, loss_function, optimizer = make(config)
        # print(f"Model: {model}")

        # train the model
        train(config, model, train_loader, validation_loader, loss_function, optimizer)

        # test the models performance
        test(config, model, test_loader, loss_function)

    return model

if __name__ == '__main__':
    parse_args()
    model = model_pipeline(default_config)