import argparse
from types import SimpleNamespace

import torch 
import os
import DataFunctions
import cv2

import torchvision.transforms as transforms


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
    # data_path = "/home/oddity/marieke/Datasets/Demo/demoimages/",
    # grinch_path = "/home/oddity/marieke/Datasets/Demo/demogrinches/",
    # video_path = "/home/oddity/marieke/Datasets/Demo/demovideos/"
    
    machine = "Mac",
    device = torch.device("mps"),
    num_workers = 1,
    dims = 224,
    
    batch_size = 32, 
    dataset = "VisuAAL",
    colour_space = "RGB",
    
    flip = True, 
    rotate = True,
    
    image_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/Pratheepan/TrainImages",
    gt_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/Pratheepan/TrainGroundTruth",
    augmented_image_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/AugmentedPratheepan/TrainImages",
    augmented_gt_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/AugmentedPratheepan/TrainGroundTruth",
)

def parse_args():
    "Overriding default arguments"
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--machine', type=str, default=default_config.machine, help='type of machine')
    argparser.add_argument('--num_workers', type=int, default=default_config.num_workers, help='number of workers in DataLoader')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--dataset', type=str, default=default_config.dataset, help='dataset')
    argparser.add_argument('--colour_space', type=str, default=default_config.colour_space, help='colour space')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return

def flip(config, i, image, gt):
    augmented_image = torch.flip(image, [-2])
    augmented_gt = torch.flip(gt, [0])
    
    DataFunctions.save_augmentation(config, i, augmented_image, augmented_gt, "upsidedown")
    return
    
def rotate(config, i, image, gt):
    # TODO: aanpassen, zodat flip en rotate van dezelfde package komen en ook meer dan 180* en mirrored kunnen.
    # rotater = transforms.tworandom(transforms.RandomRotation(degrees=(0, 180)))
    # augmented_image = rotater(image)
    augmented_image = torch.flip(image, [2])
    augmented_gt = torch.flip(gt, [1])
    
    # augmented_image = cv2.flip(image, 0)
    # augmented_gt = cv2.flip(gt, 0)
    # augmented_gt = rotater(gt)
    
    DataFunctions.save_augmentation(config, i, augmented_image, augmented_gt, "mirrored")
    
    return

""" Stores the original images and ground truths, compute and store the augmentations
"""
def create_augmentations(config, images, gts):
    # TODO: Kijken of dit efficienter kan (per batch of images)
    for i, (image, gt) in enumerate(zip(images, gts)):
        
        DataFunctions.save_augmentation(config, i, image, gt, "original")
        
        # Compute the augmentations
        if config.flip:
            flip(config, i, image, gt)
        if config.rotate:
            rotate(config, i, image, gt)
    return
    

""" Complete pipeline of loading images, creating augmentations and storing those augmentations.
"""
def preprocessing_pipeline(config):

    # Load input data
    image_list = os.listdir(config.image_path)
    image_list = [image for image in image_list if not image.startswith(".")]
    image_list.sort()
    images, gts = DataFunctions.load_input_images(config, config.image_path, config.gt_path, "augmentation")
    print(f"Dims of images: {images.shape} and gts: {gts.shape}")    
    images = images.to(config.device)
    gts = gts.to(config.device)
    
    # Create and store augmentations
    create_augmentations(config, images, gts)
    
    return


if __name__ == '__main__':
    parse_args()
    preprocessing_pipeline(default_config)