import numpy as np
import cv2
import random

import DataFunctions
from ultralytics.data import mosaic as ultralytics_mosaic

def mirror(config, i, image, gt):
    # Double transpose necessary. Else, image is upside down instead of mirrored. No other flip code works.
    augmented_image = cv2.flip(image, 1)
    augmented_gt = cv2.flip(gt, 1)
    DataFunctions.save_augmentation(config, i, augmented_image, augmented_gt, "mirrored")
    return

def sheerX(config, i, image, gt):
    x_shear = random.uniform(0, 0.3) * random.choice([-1, 1])
    rows, cols, _ = np.shape(image)
    transform_mat = np.float32([[1, x_shear, 0], [0, 1, 0]])
    augmented_image = cv2.warpAffine(image, transform_mat, (rows, cols), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    augmented_gt = cv2.warpAffine(gt, transform_mat, (rows, cols), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0))

    DataFunctions.save_augmentation(config, i, augmented_image, augmented_gt, "sheerX")
    return

def sheerY(config, i, image, gt):
    y_shear = random.uniform(0, 0.3) * random.choice([-1, 1])
    rows, cols, _ = np.shape(image)
    transform_mat = np.float32([[1, 0, 0], [y_shear, 1, 0]])
    augmented_image = cv2.warpAffine(image, transform_mat, (rows, cols), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    augmented_gt = cv2.warpAffine(gt, transform_mat, (rows, cols), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0))

    DataFunctions.save_augmentation(config, i, augmented_image, augmented_gt, "sheerY")
    return

def brightness(config, i, image, gt):
    brightness_ratio = 1 + random.uniform(0, 0.5) * random.choice([-1, 1])
    
    augmented_image = blend(image, np.zeros_like(image), brightness_ratio)
    augmented_gt = gt
    
    DataFunctions.save_augmentation(config, i, augmented_image, augmented_gt, "brightness")
    return

def blend(img1, img2, ratio):
    bound = 225.0
    return (ratio * img1 + (1.0 - ratio) * img2).clip(0, bound).astype(img1.dtype)

# def mosaic(config, images):
#     augmented_images = ultralytics_mosaic(images, imgsz=config.dims)
#     for image in augmented_images
    