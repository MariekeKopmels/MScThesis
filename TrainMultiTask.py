import argparse
from types import SimpleNamespace

import DataFunctions
import torch
import numpy as np
import os
import wandb

default_config = SimpleNamespace(
    # machine = "TS2",
    # device = torch.device("cuda"),
    # num_workers = 1,
    # dims = 224,
    # max_video_length = 100,
    # batch_size = 32, 
    # dataset = "test",
    # architecture = "I3D_Multitask", 
    # model_path = "/home/oddity/marieke/Output/Models/LargeModel/final.pt",
    # input_video_path = "/home/oddity/marieke/Datasets/Demo/demovideos/",
    
    machine = "Mac",
    device = torch.device("mps"),
    num_workers = 1,
    log = True,
    dims = 224,
    max_video_length = 300,
    batch_size = 32, 
    dataset = "test",
    architecture = "I3D_Multitask",
    # model_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Thesis/Models/GoodModelTest.pt",    
    input_video_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/Demos/Grinch/DemoInputVideos", 
)

def parse_args():
    "Overriding default arguments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--machine', type=str, default=default_config.machine, help='type of machine')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

def make(config):
    return
    

def multitask_learning_pipeline(hyperparameters):
    with wandb.init(mode="disabled", project="Violence_SkinColour", config=hyperparameters):
        config = wandb.config
        
        # Create model, data loaders, loss function and optimizer
        # model, train_loader, validation_loader, test_loader, loss_function, optimizer = make(config)
        
        # Do things
        

if __name__ == '__main__':
    parse_args()
    multitask_learning_pipeline(default_config)
    