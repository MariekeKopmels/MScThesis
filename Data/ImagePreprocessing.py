import argparse
from types import SimpleNamespace
import torch 
import os
import shutil
import Data.DataFunctions as DataFunctions
import Data.AugmentationFunctions as AugmentationFunctions

default_config = SimpleNamespace(
    # machine = "OS4",
    # device = torch.device("cuda"),
    # num_workers = 1,
    # dims = 224,
    # max_video_length = 100,
    # batch_size = 32, 
    # dataset = "Demo",
    # colour_space = "BGR",
    # architecture = "UNet", 
    # model_path = "/home/oddity/marieke/Output/Models/LargeModel/final.pt",
    # data_path = "/home/oddity/marieke/Datasets/Demo/demoimages/",
    # grinch_path = "/home/oddity/marieke/Datasets/Demo/demogrinches/",
    # video_path = "/home/oddity/marieke/Datasets/Demo/demovideos/"
    
    machine = "Mac",
    device = torch.device("mps"),
    num_workers = 1,
    dims = 224,
    batch_size = 32, 
    dataset = "LargeCombinedDataset",
    colour_space = "BGR",
    num_channels = 3,
    
    mirror = True,
    sheerX = True, 
    sheerY = True,
    brightness = True,
    
    image_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/LargeCombinedDataset/TrainImages",
    gt_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/LargeCombinedDataset/TrainGroundTruths",
    
    augmented_folder = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets",
    augmented_foldername = "LargeCombinedAugmentedDataset",
    
    augmented_image_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/LargeCombinedAugmentedDataset/TrainImages",
    augmented_gt_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/LargeCombinedAugmentedDataset/TrainGroundTruths",
)

def parse_args():
    "Overriding default arguments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--machine', type=str, default=default_config.machine, help='type of machine')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return


""" Stores the original images and ground truths, compute and store the augmentations
"""
def create_augmentations(config, images, gts):
    for i, (image, gt) in enumerate(zip(images, gts)):
        DataFunctions.save_augmentation(config, i, image, gt, "original")
        
        # Compute the augmentations
        if config.mirror:
            AugmentationFunctions.mirror(config, i, image, gt)
        if config.sheerX:
            AugmentationFunctions.sheerX(config, i, image, gt)
        if config.sheerY:
            AugmentationFunctions.sheerY(config, i, image, gt)
        if config.brightness:
            AugmentationFunctions.brightness(config, i, image, gt)
    return
    

""" Complete pipeline of loading images, creating augmentations and storing those augmentations.
"""
def preprocessing_pipeline(config):
    # Clear out previous images
    os.chdir(config.augmented_folder)
    shutil.rmtree(config.augmented_foldername)

    # Load input data
    image_list = os.listdir(config.image_path)
    image_list = [image for image in image_list if not image.startswith(".")]
    image_list.sort()
    images, gts = DataFunctions.load_input_images(config, config.image_path, config.gt_path, "augmentation")

    # Process images and gts as numpy arrays of shape (batch_size, height, width, channels) 
    # and (batch_size, height, width) respectively.
    images = images.numpy()
    images = images.transpose(0,2,3,1)
    gts = gts.numpy()
    
    # Create and store augmentations
    create_augmentations(config, images, gts)
    
    return


if __name__ == '__main__':
    parse_args()
    preprocessing_pipeline(default_config)