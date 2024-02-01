import os
import cv2
import sklearn.metrics
import torch 
import numpy as np 
from torch.utils.data import DataLoader, random_split
import time
import Logging.LogFunctions as LogFunctions

""" Loads the requested images, returns them in a Torch tensor.
"""
def load_images(config, dir_list, dir_path, gts=False):
    # Initialize the images tensor
    if gts: 
        images = torch.empty(len(dir_list), config.dims, config.dims, dtype=torch.float32)
    else:
        images = torch.empty(len(dir_list), config.num_channels, config.dims, config.dims, dtype=torch.float32)

    # Read images one by one
    for i, file_name in enumerate(dir_list):
        # Load image
        path = dir_path + "/" + file_name
        image = cv2.imread(path)

        # Convert image from BGR to YCrCb or HSV if needed
        if config.colour_space == "YCrCb" and not gts:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
        elif config.colour_space == "HSV" and not gts:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Convert Ground Truth from BGR to 1 channel (Black or White)
        if gts: 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _,image = cv2.threshold(image,127,1,0)
        
        # Resize images and ground truths to size dims*dims (usually 224*224)
        image = cv2.resize(image, (config.dims,config.dims), interpolation=cv2.INTER_CUBIC)
        
        # Reformat the image, add to the images tensor
        if gts:
            images[i] = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        else:
            images[i] = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            
    return images

"""Returns input images in a given directory
        Format of return: torch tensors containin images and corresponding ground truths.
        Torch tensors are of shape batch_size,num_channels,dims,dims for images and batch_size,dims,dims for ground truths 
"""
def load_input_images(config, image_dir_path, gt_dir_path, stage):
    # Load list of files in directories
    image_list = os.listdir(image_dir_path)
    gt_list = os.listdir(gt_dir_path)
    
    # Not all images have a ground truth, select those that do. Also skip the hidden files.
    dir_list = [file for file in image_list if file in gt_list and not file.startswith(".")]
    # TODO: shuffle? 
        
    # Include as many items as requested
    if stage == "train":
        dir_list = dir_list[:config.train_size]
    elif stage == "test":
        dir_list = dir_list[:config.test_size]
    
    # Load the images
    images = load_images(config, dir_list, image_dir_path)
    gts = load_images(config, dir_list, gt_dir_path, gts=True)
    
    return images, gts

""" Returns train and test, all being data loaders with tuples of input images and corresponding ground truths.
        Format of return: Data loaders of tuples containging an input image tensor and a ground truth image tensor
        Image tensors are of shape (batch_size, channels, height, width)
"""
def load_image_data(config):
    # Load train and test data
    print("Loading training data...")
    base_path = config.data_path + "/" + config.trainset
    train_images, train_gts = load_input_images(config, base_path + "/TrainImages", base_path + "/TrainGroundTruths", stage = "train")
    print("Loading testing data...")
    base_path = config.data_path + "/" + config.testset
    test_images, test_gts = load_input_images(config, base_path + "/TestImages", base_path + "/TestGroundTruths", stage = "test")

    # Combine images and ground truths in TensorDataset format
    train = torch.utils.data.TensorDataset(train_images, train_gts)
    test = torch.utils.data.TensorDataset(test_images, test_gts)

    # Put data into dataloaders
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True, drop_last=False)
    
    return train_loader, test_loader


""" Splits the data in the passed data loader into a train and validation loader.
"""
def split_dataset(config, data_loader):
    generator = torch.Generator().manual_seed(42)
    train_data, validation_data = random_split(data_loader.dataset, [config.split, 1-config.split], generator)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, drop_last=False) 
    validation_loader = DataLoader(validation_data, batch_size=config.batch_size, shuffle=True, drop_last=False) 
    
    return train_loader, validation_loader


""" Splits the videos in config.video_path directory into images, stores them on disk
"""
def split_video_to_images(config):    
    video_list = os.listdir(config.video_path)
    video_list = [video for video in video_list if not video.startswith(".") and video.endswith(".mp4")]
    
    print("Video_list: ", video_list)
        
    video_no = 0
    for video in video_list:
        video_path = config.video_path + "/" + video
        print("video_path: ", video_path)
        os.chdir(config.video_path)
        os.makedirs(f"video_{video_no}", exist_ok=True)
        os.chdir(f"{config.video_path}/video_{video_no}")
        
        video_capture = cv2.VideoCapture(video_path)
        frame_no = 0
        while video_capture.isOpened():
            frame_is_read, frame = video_capture.read()
            if frame_is_read:
                cv2.imwrite(f"frame_{str(frame_no).zfill(5)}.jpg", frame)
                frame_no += 1
            else: 
                print(f"Reached the end of video {video_no}!")
                video_capture.release()
            if frame_no == config.max_video_length: 
                print(f"Reached the max video length!")
                video_capture.release()

        video_no += 1
    return

""" Reads images from config.grinch_path directory, merges them into a video and stores the video on disk
"""
def merge_images_to_video(config):
    video_list = os.listdir(config.grinch_path)
    video_list = [video for video in video_list if not video.startswith(".") and not video.endswith(".mp4")]
    video_list.sort()
    for video in video_list:
        os.chdir(f"{config.grinch_path}")
        
        image_list = os.listdir(f"{config.grinch_path}/{video}")
        image_list = [image for image in image_list if not image.startswith(".")]
        image_list.sort()
        
        video_name = "grinch_" + video + ".mp4"
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')        
        video_writer = cv2.VideoWriter(filename=video_name, fourcc=fourcc, fps=25, frameSize=(config.dims, config.dims))
        
        os.chdir(f"{config.grinch_path}/{video}")
        while video_writer.isOpened():
            for i in image_list:
                image = cv2.imread(i)
                video_writer.write(image)
            video_writer.release()
    return

""" Normalizes values in the passed images by dividing by the max value of 255, irrespective of the colour space.
    Therefore, the original images have values in the range [0,255] whereas the normalized
    Images are in range [0,255].
"""
def normalize_images(config, images):
    channel1, channel2, channel3 = images[:, 0, :, :], images[:, 1, :, :], images[:, 2, :, :]
    if channel1.max() > 255.0 or channel2.max() > 255.0 or channel3.max() > 255.0:
        print(f"WARNING: there is a value larger than 255! Should not happen. Colour_space:{config.colour_space}")
    
    normalized_images = images/255.0
    
    return normalized_images

''' Returns the grinch version of the passed image, based on the passed output of the model
'''
def make_grinch(config, image, output):
    if type(output) == torch.Tensor:
        output = output.to("cpu").numpy()
    grinch = np.copy(image)
    mask = output == 1
    green_bgr = [0, 255, 0]
    if config.colour_space == "BGR":
        grinch[mask] = green_bgr
    elif config.colour_space == "YCrCb":
        green_ycrcb = cv2.cvtColor(np.uint8([[green_bgr]]), cv2.COLOR_BGR2YCrCb)[0][0]
        grinch[mask] = green_ycrcb
    elif config.colour_space == "HSV":
        green_hsv = cv2.cvtColor(np.uint8([[green_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        grinch[mask] = green_hsv
    
    return grinch

""" Takes baches of images in ndarrays, stores as well as returns the grinch versions
"""
def to_grinches(config, images, outputs, video):
    outputs = outputs.cpu().numpy()
    grinches = images.cpu().numpy()
    mask = outputs == 1
    
    os.makedirs(config.grinch_path, exist_ok=True)
    os.chdir(config.grinch_path)
    os.makedirs(video, exist_ok=True)
    
    for i in range(len(mask)):
        grinches[i] = make_grinch(config, grinches[i].transpose(1,2,0), outputs[i]).transpose(2,0,1)
        save_path = config.grinch_path + "/" + video
        save_name = "grinchframe_" + str(i).zfill(5) + ".jpg"
        save_image(config, grinches[i].transpose(1,2,0), save_path, save_name)
            
    return torch.from_numpy(grinches)
    
    
"""Returns the values of the confusion matrix of true negative, false negative, true positive and false positive values
"""
def confusion_matrix(config, outputs, targets, stage):
    # Flatten output and target
    output = torch.flatten(outputs)
    target = torch.flatten(targets)
    
    # Binarize output of the model, convert to floats
    output = output > 0.5
    output.float()
    
    # Compute the confusion matrix
    matrix = sklearn.metrics.confusion_matrix(target.to("cpu"), output.to("cpu"))
    
    tn = matrix[0][0]
    fn = matrix[1][0]
    tp = matrix[1][1]
    fp = matrix[0][1]
    
    return tn, fn, fp, tp

""" Computes metrics based on true/false positive/negative values.
        Returns accuracy, fn_rate, fp_rate and sensitivity.
"""
def metrics(tn, fn, fp, tp):
    if tn+fn+fp+tp == 0:
        print("Error occured in DataFunctions.metrics function: tn+fn+fp+tp=0. So, there are no metrics to be computed.")
    accuracy = (tp+tn)/(tn+fn+fp+tp)
    # If there are no pixels that should be marked as skin, the fn_rate should be 0
    fn_rate = fn/(fn+tp) if fn+tp != 0 else 0
    # None of the images are all skin, but added a safety measure
    fp_rate = fp/(tn+fp) if tn+fp != 0 else 0
    # If there are no pixels that should be marked as skin, the sensitivity should be 1
    sensitivity = tp/(fn+tp) if fn+tp!= 0 else 1
    
    f1_score = tp/(tp+((fp+fn)*0.5))
    IoU = tp/(tp+fp+fn)
    
    return accuracy, fn_rate, fp_rate, sensitivity, f1_score, IoU


""" Stores an image (in form of ndarray or tensor) to the disk.
"""
def save_image(config, image, path, filename, bw=False, gt=False):
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    # cv2.imwrite takes input in form height, width, channels
    if type(image) == torch.Tensor:
        image = image.to("cpu")
        if gt:
            image = cv2.cvtColor(image.numpy(), cv2.COLOR_GRAY2BGR)
            cv2.imwrite(filename, image*255)
        else:
            cv2.imwrite(filename, image.numpy().transpose(1,2,0))
    else:   
        if gt: 
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(filename, image*255)
        else:
            cv2.imwrite(filename, image)
    return
    

""" Stores the image and ground truth with the same filename, but in a different folder.
"""
def save_augmentation(config, i, image, gt, augmentation):
    # Determine the filename
    filename = f"image_{i}_{augmentation}.jpg"
    
    # Store the image
    path = config.augmented_image_path
    save_image(config, image, path, filename)
    
    # Store the ground truth
    path = config.augmented_gt_path
    save_image(config, gt, path, filename, gt=True)
    
    return


""" Copies images from a folder into a new folder.
"""
def move_images(config, start_index, origin_path, destination_path, image_list=[], gts=False):  
    if image_list == []:
        os.makedirs(origin_path, exist_ok=True)
        image_list = os.listdir(origin_path)
        image_list = [image for image in image_list if not image.startswith(".")]
        image_list.sort()
    
    images = load_images(config, image_list, origin_path, gts=gts)

    index = start_index
    for image in images:
    
        save_name = "image_" + str(index).zfill(6) + ".jpg"
        save_image(config, image, destination_path, save_name, gt=gts)
        index += 1
            
    return index