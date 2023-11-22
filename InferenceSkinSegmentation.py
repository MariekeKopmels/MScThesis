import argparse
from types import SimpleNamespace

import DataFunctions
import torch
import numpy as np
import os
import cv2

default_config = SimpleNamespace(
    machine = "TS2",
    device = torch.device("cpu"),
    num_workers = 1,
    dims = 224,
    batch_size = 32, 
    dataset = "Demo",
    colour_space = "RGB",
    architecture = "UNet", 
    model_path = "/home/oddity/marieke/Output/Models/SmallModel/final.pt",
    data_path = "/home/oddity/marieke/Datasets/Demo/demoimages/",
    grinch_path = "/home/oddity/marieke/Datasets/Demo/demogrinches/",
    
    # machine = "Mac",
    # device = torch.device("cpu"),
    # num_workers = 1,
    # dims = 224,
    # batch_size = 32, 
    # dataset = "VisuAAL",
    # colour_space = "RGB",
    # architecture = "UNet",
    # model_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Thesis/Models/final.pt",
    # grinch_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/visuAAL/Grinch",
    # data_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/visuAAL/TestImages"
)


def parse_args():
    "Overriding default arguments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--machine', type=str, default=default_config.machine, help='type of machine')
    argparser.add_argument('--num_workers', type=int, default=default_config.num_workers, help='number of workers in DataLoader')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--dataset', type=str, default=default_config.dataset, help='dataset')
    argparser.add_argument('--colour_space', type=str, default=default_config.colour_space, help='colour space')
    argparser.add_argument('--architecture', type=str, default=default_config.architecture, help='architecture')
    argparser.add_argument('--model_path', type=str, default=default_config.model_path, help='path to the model')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return


def inference(config):
    model = torch.load(config.model_path).to(config.device)
    model.eval()
    dir_list = os.listdir(config.data_path)
    dir_list = [dir for dir in dir_list if not dir.startswith(".")]
    dir_list.sort()
    dir_list = dir_list[:200]
    
    print("len of Dir_list: ", len(dir_list))
    
    print("Loading images...")
    images = DataFunctions.load_images(config, dir_list, config.data_path)
    images.to(config.device)
    
    # print(f"Device of images: {images.get_device()}")
    # print(f"Device of model: {model.get_device()}")
    
    print("Calculating masks")
    with torch.no_grad():
        masks = model(images)
        masks = (masks >= 0.5).float()
        
    print("Masks shape: ", np.shape(masks))
    print("Start making grinches")    
    grinches = DataFunctions.to_grinches(config, images, masks)
    
    print("Grinches shape: ", np.shape(grinches))
    
    return 


if __name__ == '__main__':
    parse_args()
    inference(default_config)