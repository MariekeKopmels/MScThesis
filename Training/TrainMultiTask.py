# Main for training the multi-head violence and skin colour prediction mode

import argparse
from types import SimpleNamespace

import Data.DataFunctions as DataFunctions
import Models.MyModels as MyModels
import torch
import wandb

default_config = SimpleNamespace(
    # machine = "OS4",
    # device = torch.device("cuda"),
    # num_workers = 1,
    # num_channels = 3,
    # dims = 224,
    # max_video_length = 100,
    # batch_size = 32, 
    # dataset = "test",
    # architecture = "I3D_Multitask", 
    # model_path = "/home/oddity/marieke/Output/Models/LargeModel/final.pt",
    # input_video_path = "/home/oddity/marieke/Datasets/Demo/demovideos/",
    
    machine = "mac",
    device = torch.device("mps"),
    num_workers = 1,
    num_channels = 3,
    log = True,
    dims = 224,
    num_violence_classes = 2,           #Violence and Neutral
    num_skincolour_classes = 6,         #White, Asian, Latino, Black, Unknown, NonSuitableVideo 

    max_video_length = 16,
    batch_size = 32, 
    
    train_size = 5, 
    validation_size = 2,
    test_size = 3,
    
    trainset = "DemoGrinchVideos",
    validationset = "DemoGrinchVideos",
    testset = "DemoGrinchVideos",
    data_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/Demos/Grinch",
    
    architecture = "I3D_Multitask",
    model_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/SkinDetection/Models/pretrained.pt",    
)

def parse_args():
    "Overriding default arguments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--config', type=str, default='Config/mac.config', help='path to the configuration file.')
    args = argparser.parse_args()
    return args


def load_config(config_path):
    config = {}
    with open(config_path, 'r') as config_file:
        for line in config_file:
            key, value = line.strip().split('=')
            if value.isdigit():
                config[key.strip()] = int(value)
            else:
                config[key.strip()] = value
    return SimpleNamespace(**config)
    
    
def make(config):
    print("Creating data loaders")
    train_loader, test_loader = DataFunctions.load_video_data(config)
    
    model = MyModels.MultiTaskModel(config).to(config.device)
    
    return model, train_loader, test_loader

def multitask_learning_pipeline(hyperparameters):
    with wandb.init(mode="disabled", project="multi-task-model", config=hyperparameters):
        config = wandb.config
        
        # Create model, data loaders, loss function and optimizer
        model, train_loader, test_loader = make(config)
        
        # Do things
    return        

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    multitask_learning_pipeline(config)
