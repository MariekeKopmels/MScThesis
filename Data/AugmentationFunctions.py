import numpy as np
import cv2
import random
import torch
import Data.DataFunctions as DataFunctions

def augment_images(config, images, gts):
    images = images.to("cpu")
    gts = gts.to("cpu")
    
    no_samples = len(images)
    augmented_images = np.zeros((no_samples, *images[0].shape), dtype=np.float32)
    augmented_gts = np.zeros((no_samples, *gts[0].shape), dtype=np.float32)
    
    for i in range(len(images)):
        image = images[i]
        gt = gts[i]
        
        image = np.array(image)
        gt = np.array(gt)
        
        image = image.transpose(1,2,0)
            
        augmented_image, augmented_gt = choose_and_perform_augmentation(image, gt)
        augmented_images[i] = augmented_image.transpose(2,0,1)
        augmented_gts[i] = augmented_gt

    
    augmented_images = torch.from_numpy(augmented_images).to(config.device)
    augmented_gts = torch.from_numpy(augmented_gts).to(config.device)
    
    return augmented_images, augmented_gts

def choose_and_perform_augmentation(image, gt):    
    augmented_image = np.copy(image)
    augmented_gt = np.copy(gt)
    augmentations = [random.random() for _ in range(0,4)]
    
    if augmentations[0] < 0.33:
        augmented_image, augmented_gt = mirror(augmented_image, augmented_gt)
    if augmentations[1] < 0.33:
        augmented_image, augmented_gt = sheerX(augmented_image, augmented_gt)
    if augmentations[2] < 0.33:
        augmented_image, augmented_gt = sheerY(augmented_image, augmented_gt)
    if augmentations[3] < 0.33:
        augmented_image, augmented_gt = brightness(augmented_image, augmented_gt)

    return augmented_image, augmented_gt
    
def mirror(image, gt):
    augmented_image = cv2.flip(image, 1)
    augmented_gt = cv2.flip(gt, 1)
    return augmented_image, augmented_gt

def sheerX(image, gt):
    x_shear = random.uniform(0, 0.3) * random.choice([-1, 1])
    rows, cols, _ = np.shape(image)
    transform_mat = np.float32([[1, x_shear, 0], [0, 1, 0]])
    augmented_image = cv2.warpAffine(image, transform_mat, (rows, cols), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    augmented_gt = cv2.warpAffine(gt, transform_mat, (rows, cols), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0))
    return augmented_image, augmented_gt

def sheerY(image, gt):
    y_shear = random.uniform(0, 0.3) * random.choice([-1, 1])
    rows, cols, _ = np.shape(image)
    transform_mat = np.float32([[1, 0, 0], [y_shear, 1, 0]])
    augmented_image = cv2.warpAffine(image, transform_mat, (rows, cols), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    augmented_gt = cv2.warpAffine(gt, transform_mat, (rows, cols), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0))
    return augmented_image, augmented_gt

def brightness(image, gt):
    brightness_ratio = 1 + random.uniform(0, 0.5) * random.choice([-1, 1])
    augmented_image = blend(image, np.zeros_like(image), brightness_ratio)
    augmented_gt = gt
    return augmented_image, augmented_gt

def blend(img1, img2, ratio):
    upper_pixel_bound = 255.0
    return (ratio * img1 + (1.0 - ratio) * img2).clip(0, upper_pixel_bound).astype(img1.dtype)