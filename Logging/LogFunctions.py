import cv2
import wandb
import warnings
import Data.DataFunctions as DataFunctions
import torch
import os
import datetime
import numpy as np


""" Prints intermediate results to WandB, also logs them to WandB if not in batch stage.
"""
def log_metrics(config, mean_loss, tn, fn, fp, tp, stage):
    if config.log:
        accuracy, fn_rate, fp_rate, sensitivity, f1_score, IoU = DataFunctions.metrics(tn, fn, fp, tp)
        print(f"{stage} results\nMean loss: {mean_loss:.6f}, accuracy: {accuracy:.3f}, fn_rate: {fn_rate:.3f}, fp_rate:: {fp_rate:.3f}, sensitivity: {sensitivity:.3f}, f1-score: {f1_score:.3f}, IoU: {IoU:.3f}") 
        
        if stage != "batch":
            wandb.log({f"{stage}_accuracy": accuracy, f"{stage}_fn_rate": fn_rate, f"{stage}_fp_rate": fp_rate, f"{stage}_sensitivity": sensitivity, f"{stage}_f1_score": f1_score, f"{stage}_IoU": IoU})
    
    return IoU


""" Logs intermediate results from the multitask model to WandB.
"""    
def log_multitask_metrics(config, mean_loss, violence_f1_score, skincolour_f1_score, stage):
    print(f"{stage}_mean_loss = {mean_loss}, {stage}_violence_f1 = {violence_f1_score}, {stage}_skincolour_f1 = {skincolour_f1_score}")
    wandb.log({f"{stage}_mean_loss": mean_loss, f"{stage}_violence_f1": violence_f1_score, f"{stage}_skincolour_f1": skincolour_f1_score})
    
    
""" Stores the model to the disk.
"""
def save_model(config, model, epoch, final=False):
    print("Saving model")
    if not config.log:
        print("Warning: Saving the model even though config.log == False")
    
    os.chdir(config.model_path)
    folder = f"LR:{config.lr}_Colour_space:{config.colour_space}_Pretrained:{config.pretrained}_Trainset:{config.trainset}_Validationset:{config.validationset}"
    os.makedirs(folder, exist_ok=True)
    path = config.model_path + f"/{folder}/epoch_{epoch}.pt"
    if final:
        path = config.model_path + f"/{folder}/final.pt"
    wandb.unwatch()
    
    # Store the model on CPU, since my laptop can't open cuda and the server can't open mps
    model = model.to("cpu")
    torch.save(model, path)
    model = model.to(config.device)
    wandb.watch(model, log="all", log_freq=1)
        

""" Logs test examples of input image, ground truth and model output to WandB.
"""
def log_example(config, example, target, output, stage="UNKNOWN"):
    if config.log:
        bw_output = (output >= 0.5).float()
        grinch = DataFunctions.make_grinch(config, example, bw_output)
        
        # Change channels from YCrCb or BGR to RGB
        if config.colour_space == "YCrCb": 
            example = cv2.cvtColor(example, cv2.COLOR_YCR_CB2RGB)
            grinch = cv2.cvtColor(grinch, cv2.COLOR_YCR_CB2RGB)
        elif config.colour_space == "HSV":
            example = cv2.cvtColor(example, cv2.COLOR_HSV2RGB)
            grinch = cv2.cvtColor(grinch, cv2.COLOR_HSV2RGB)
        elif config.colour_space == "BGR":
            example = cv2.cvtColor(example, cv2.COLOR_BGR2RGB)
            grinch = cv2.cvtColor(grinch, cv2.COLOR_BGR2RGB)
        else:
            warnings.warn("No colour space found!")
            print(f"Current config.colour_space = {config.colour_space}")
            
        wandb.log({f"Input {stage} image":[wandb.Image(example)]})
        wandb.log({f"Target {stage} output": [wandb.Image(target)]})
        wandb.log({f"Model {stage} greyscale output": [wandb.Image(output)]})
        wandb.log({f"Model {stage} bw output": [wandb.Image(bw_output)]})
        wandb.log({f"Model {stage} grinch output": [wandb.Image(grinch)]})
