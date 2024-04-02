import argparse
from types import SimpleNamespace

import Data.DataFunctions as DataFunctions
import Models.ModelFunctions as ModelFunctions
import torch
import numpy as np
import os
import torch.nn as nn

default_config = SimpleNamespace(
    machine = "OTS5",
    device = torch.device("cuda"),
    num_workers = 1,
    dims = 224,
    num_channels = 3,
    max_video_length = 16,
    batch_size = 32, 
    dataset = "refined_nms",
    colour_space = "YCrCb",
    architecture = "UNet", 
    model_path = "/home/oddity/marieke/Output/Models",
    model_name = "",
    video_path = "/home/oddity/marieke/Datasets/05_firsthalf_testset/balanced_samples/samples",
    grinch_path = "/home/oddity/marieke/Datasets/05_firsthalf_testset/balanced_samples/YCrCb_grinchsamples"
    
    # machine = "Mac",
    # device = torch.device("mps"),
    # num_workers = 1,
    # dims = 224,ls
    # num_channels = 3,
    # max_video_length = 50,
    # batch_size = 32, 
    # dataset = "VisuAAL",
    # colour_space = "BGR",
    # architecture = "UNet",
    # model_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Output/SkinDetectionModels",
    # model_name = "test_pretrained.pt",
    # video_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/Demos/Grinch/DemoInputVideos", 
    # grinch_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/Demos/Grinch/DemoGrinchVideosNEW"
)

def parse_args():
    "Overriding default arguments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--machine', type=str, default=default_config.machine, help='type of machine')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--colour_space', type=str, default=default_config.colour_space, help='colour space')
    argparser.add_argument('--model_path', type=str, default=default_config.model_path, help='path to the model')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

def inference(config):
    model = ModelFunctions.load_model(config, config.model_name)
    
    if model.colour_space != config.colour_space:
        print("WARNING! Colour space is not the same as the model's colour space!")
    
    video_dir = os.listdir(config.video_path)
    video_dir = [folder for folder in video_dir if not folder.startswith(".") and not folder.endswith(".mp4")]
    video_dir.sort()
        
    i = 0
    for video in video_dir:
        print(f"Processing video {i}/{len(video_dir)}", end="\r")
        i += 1
        
        image_list = os.listdir(f"{config.video_path}/{video}")
        image_list = [image for image in image_list if not image.startswith(".")]
        image_list.sort()
        image_list = image_list[:config.max_video_length]
                
        # print("Loading images...")
        images = DataFunctions.load_images(config, image_list, f"{config.video_path}/{video}")
        images = images.to(device=config.device)
        
        # print("Calculating masks...")
        with torch.no_grad():
            images.to(config.device)
            normalized_images = DataFunctions.normalize_images(config, images)
            _, masks = model(normalized_images)
            binary_masks = (masks >= 0.5).float()
            
        # print("Start making grinches")    
        _ = DataFunctions.to_grinches(config, images, binary_masks, video)
        
    return 


def inference_pipeline(hyperparameters):
    config = hyperparameters
    config.model_name = config.colour_space + "_DefinitiveBestShortFinetune.pt"
    
    # Splits videos into images
    DataFunctions.split_video_to_images(config)
    
    # put images through model
    inference(config)
    
    # merge grinch images into grinch videos
    DataFunctions.merge_images_to_video(config)
        

if __name__ == '__main__':
    parse_args()
    inference_pipeline(default_config)
    
