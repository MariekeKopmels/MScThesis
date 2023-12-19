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
    num_violence_classes = 2,           #Violence and Neutral
    num_skincolour_classes = 6,         #White, Asian, Latino, Black, Unknown, NonSuitableVideo 

    max_video_length = 16,
    batch_size = 32, 
    
    train_size = 5, 
    validation_size = 2,
    test_size = 3,
    
    trainset = "DemoGrinchVideos",
    data_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/Demos/Grinch",
    
    architecture = "I3D_Multitask",
    model_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Thesis/Models/pretrained.pt",    
)

def parse_args():
    "Overriding default arguments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--machine', type=str, default=default_config.machine, help='type of machine')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

def make(config):
    print("Creating data loaders")
    train_loader, validation_loader, test_loader = DataFunctions.load_video_data(config)
    
    model = MyModels.MultiTaskModel(config).to(config.device)
    
    return model, train_loader, validation_loader, test_loader
    

def multitask_learning_pipeline(hyperparameters):
    with wandb.init(mode="disabled", project="Violence_SkinColour", config=hyperparameters):
        config = wandb.config
        
        # Create model, data loaders, loss function and optimizer
        train_loader, validation_loader, test_loader = make(config)
        
        # Do things
    return        

if __name__ == '__main__':
    parse_args()
    multitask_learning_pipeline(default_config)
