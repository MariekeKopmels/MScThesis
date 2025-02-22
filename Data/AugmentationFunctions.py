import numpy as np
import cv2
import random
import torch


""" Augments a set of videos.
"""
def augment_videos(config, videos):
    augmented_videos = videos
    for i, video in enumerate(videos):
        augmented_videos[i] = augment_images(config, video, video=True)

    return augmented_videos

''' Augments the passed images, and corresponding ground truths if applicable. 
    If the set of images is a video, all images are augmented in the same fashion.
'''
def augment_images(config, images, gts=None, video=False):
    no_samples = len(images)

    images = images.to("cpu")
    augmented_images = np.zeros((no_samples, *images[0].shape), dtype=np.float32)
    if gts != None:
        gts = gts.to("cpu")
        augmented_gts = np.zeros((no_samples, *gts[0].shape), dtype=np.float32)

    augmentations = [random.random() for _ in range(0,4)]

    for i in range(len(images)):
        image = images[i]
        image = np.array(image)
        image = image.transpose(1,2,0)
        augmented_image = perform_augmentation(config, image, augmentations)
        augmented_images[i] = augmented_image.transpose(2,0,1)

        if gts != None:
            gt = gts[i]
            gt = np.array(gt)
            augmented_gt = perform_augmentation(config, gt, augmentations, True)
            augmented_gts[i] = augmented_gt

        # New type of augmentation for each image, if the set is not a video
        if not video:
            augmentations = [random.random() for _ in range(0,4)]

    augmented_images = torch.from_numpy(augmented_images).to(config.device)

    if gts != None:
        augmented_gts = torch.from_numpy(augmented_gts).to(config.device)
        return augmented_images, augmented_gts
    
    return augmented_images

""" Performs possibly several augmentations to the image.
"""
def perform_augmentation(config, image, augmentations, gt=False):    
    augmented_image = image
    
    if augmentations[0] < config.augmentation_rate:
        augmented_image = mirror(augmented_image)
    if augmentations[1] < config.augmentation_rate:
        augmented_image = sheerX(augmented_image, gt)
    if augmentations[2] < config.augmentation_rate:
        augmented_image = sheerY(augmented_image, gt)
    if augmentations[3] < config.augmentation_rate:
        augmented_image = brightness(augmented_image, gt)

    return augmented_image
    
def mirror(image):
    augmented_image = cv2.flip(image, 1)
    return augmented_image

def sheerX(image, gt=False):
    x_shear = random.uniform(0, 0.3) * random.choice([-1, 1])
    transform_mat = np.float32([[1, x_shear, 0], [0, 1, 0]])
    if not gt:
        rows, cols, _ = np.shape(image)
        augmented_image = cv2.warpAffine(image, transform_mat, (rows, cols), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    else: 
        rows, cols = np.shape(image)
        augmented_image = cv2.warpAffine(image, transform_mat, (rows, cols), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0))
    return augmented_image

def sheerY(image, gt=False):
    y_shear = random.uniform(0, 0.3) * random.choice([-1, 1])
    transform_mat = np.float32([[1, 0, 0], [y_shear, 1, 0]])
    if not gt:
        rows, cols, _ = np.shape(image)
        augmented_image = cv2.warpAffine(image, transform_mat, (rows, cols), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    else:
        rows, cols = np.shape(image)
        augmented_image = cv2.warpAffine(image, transform_mat, (rows, cols), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0))
    return augmented_image

def brightness(image, gt=False):
    if gt:
        return image
    brightness_ratio = 1 + random.uniform(0, 0.5) * random.choice([-1, 1])
    augmented_image = blend(image, np.zeros_like(image), brightness_ratio)
    return augmented_image

def blend(img1, img2, ratio):
    upper_pixel_bound = 255.0
    return (ratio * img1 + (1.0 - ratio) * img2).clip(0, upper_pixel_bound).astype(img1.dtype)