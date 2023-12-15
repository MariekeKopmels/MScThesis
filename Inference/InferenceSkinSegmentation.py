import argparse
from types import SimpleNamespace

import Data.DataFunctions as DataFunctions
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
    # dataset = "Demo",
    # colour_space = "RGB",
    # architecture = "UNet", 
    # model_path = "/home/oddity/marieke/Output/Models/LargeModel/final.pt",
    # video_path = "/home/oddity/marieke/Datasets/Demo/demovideos/",
    # grinch_path = "/home/oddity/marieke/Datasets/Demo/demogrinches/"

    
    machine = "Mac",
    device = torch.device("mps"),
    num_workers = 1,
    log = True,
    dims = 224,
    max_video_length = 300,
    batch_size = 32, 
    dataset = "VisuAAL",
    colour_space = "RGB",
    architecture = "UNet",
    model_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Thesis/Models/GoodModelTest.pt",    
    video_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/Demos/Grinch/DemoInputVideos", 
    grinch_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/Demos/Grinch/DemoGrinchVideos"
)

def parse_args():
    "Overriding default arguments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--machine', type=str, default=default_config.machine, help='type of machine')
    argparser.add_argument('--num_workers', type=int, default=default_config.num_workers, help='number of workers in DataLoader')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--trainset', type=str, default=default_config.trainset, help='trainset')
    argparser.add_argument('--colour_space', type=str, default=default_config.colour_space, help='colour space')
    argparser.add_argument('--architecture', type=str, default=default_config.architecture, help='architecture')
    argparser.add_argument('--model_path', type=str, default=default_config.model_path, help='path to the model')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return


def inference(config):
    model = torch.load(config.model_path).to(config.device)
    model.eval()
    
    video_dir = os.listdir(config.video_path)
    video_dir = [folder for folder in video_dir if not folder.startswith(".") and not folder.endswith(".mp4")]
    video_dir.sort()
    print("directory contents: ", video_dir)
        
    for video in video_dir:
        image_list = os.listdir(f"{config.video_path}/{video}")
        image_list = [image for image in image_list if not image.startswith(".")]
        image_list.sort()
        image_list = image_list[:config.max_video_length]
        
        print("len of file_list: ", len(image_list))
        
        print("Loading images...")
        images = DataFunctions.load_images(config, image_list, f"{config.video_path}/{video}")
        images = images.to(device=config.device)
        
        # print(f"Device of images: {images.get_device()}")
        # print(f"Device of model: {model.get_device()}")
        
        with torch.no_grad():
            print(f"Calculating masks of {video}")
            images.to(config.device)
            masks = model(images)
            print("Making masks black&white")
            masks = (masks >= 0.5).float()
            
        print("Masks shape: ", np.shape(masks))
        print("Start making grinches")    
        grinches = DataFunctions.to_grinches(config, images, masks, video)
        
        print("Grinches shape: ", np.shape(grinches))
        
    return 
    

def inference_pipeline(hyperparameters):
    with wandb.init(mode="disabled", project="skin_inference", config=hyperparameters):
        config = wandb.config
        
        # Splits videos into images
        DataFunctions.split_video_to_images(config)
        
        # put images through model
        inference(config)
        
        # merge grinch images into grinch videos
        DataFunctions.merge_images_to_video(config)
        

if __name__ == '__main__':
    parse_args()
    inference_pipeline(default_config)
    