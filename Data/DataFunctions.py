import os
import cv2
import sklearn.metrics
import json
import torch 
import numpy as np 
import random
from torch.utils.data import DataLoader, random_split


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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        
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


""" Loads the frames of the requested videos, returns them in a Torch tensor.
"""
def load_video_frames(config, dir_list, dir_path):
    # dir_path = /Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/COPY-oddity-copy-refined-nms-data-0103/samples
    # dir_list = list of folders samples folder (so video names)
    
    # Initialize the videos tensor
    videos = torch.empty(len(dir_list), config.max_video_length, config.num_channels, config.dims, config.dims, dtype=torch.float32)
        
    # read the videos
    for i, video_name in enumerate(dir_list):
        video_path = dir_path + "/" + video_name
        frame_list = os.listdir(video_path)
        frame_list = [frame for frame in frame_list if not frame.startswith(".")]
        frame_list.sort()
        
        frames = load_images(config, frame_list, video_path)
        videos[i][:] = frames
        
    return videos

""" Loads the ground truths of videos in torch tensor format. In the returned ground truths, 
    the first element of each row is the violence target and the rest of the columns represent 
    the one-hot encoded skin colour class.
"""
def load_video_gts(config, dir_list, dir_path):
    gts = []
    for gt_name in dir_list:
        path = dir_path + "/" + gt_name
        with open(path, "r") as json_file:
            gts_json = json.load(json_file)
        label = gts_json['class_label'][0]
        gts.append(map_to_numeric(config, label))

    # Reshape to correct format of [dataset_size,1]
    gts = torch.tensor(gts).unsqueeze(dim=1)
    
    return gts.float()

""" Function to map the violence and skin tone labels to a numeric value
    For the violence labels it is defined as: 
    Violence = 1, Neutral = 0
    and for skin tone it is defined as:
    White = 1.0, Light brown = 2.0, Medium brown = 3.0, Dark brown = 4.0, Black = 5.0
"""
def map_to_numeric(config, label):
    # Define mappings from label strings to numeric values
    violence_label_mapping = {
        "neutral": 0.0,
        "violence": 1.0,
    }
    skin_tone_label_mapping = {
        "Questionable": 0.0,
        "White": 1.0,
        "LightBrown": 2.0,
        "MediumBrown": 3.0,
        "DarkBrown": 4.0,
        "Black": 5.0
    }
    # Return label or -1 if label is not found in mapping
    if config.architecture == "I3D_Violence":
        mapped_label = violence_label_mapping.get(label, -1)
        if mapped_label == -1:
            print(f"Label {label} not found, mapped to {mapped_label}.")
        return mapped_label
    elif config.architecture == "ResNet_SkinTone":
        mapped_label =  skin_tone_label_mapping.get(label, -1)
        if mapped_label == -1:
            print(f"Label {label} not found, mapped to {mapped_label}.")
        return mapped_label
    else:
        return -1


""" Returns the videos and gts of a certain batch if one is given,
    returns all violence or skin tone data otherwise.
"""
def load_video_data(config, video_list, batch=-1):       
    if batch != -1:
        start_index = batch*config.batch_size
        end_index = min((batch+1)*config.batch_size, len(video_list))
    else:
        start_index = 0
        end_index = len(video_list)
    
    # Get the correct part of the videos, dependent on what batch is retrieved
    video_list = video_list[start_index:end_index]    
    gt_list = [video + ".json" for video in video_list]
    
    # load the video frames
    dir_path = config.data_path + "/" + config.sampletype 
    videos = load_video_frames(config, video_list, dir_path)
    
    # load the ground truths, depending on the task at hand
    if config.architecture == "I3D_Violence":
        dir_path = config.data_path + "/labels" 
    elif config.architecture == "ResNet_SkinTone":
        dir_path = config.data_path + "/skin_tone_labels"
    else:
        print(f"Error! Architecture ({config.architecture}) not found. Error trown in DataFunctions.load_video_data()")
        exit()
     
    gts = load_video_gts(config, gt_list, dir_path)
    
    return videos, gts

""" Splits the data in the passed data loader into a train and validation loader.
"""
def split_dataset(config, data_loader, split_type="train/validation"):
    generator = torch.Generator().manual_seed(config.seed)
    if split_type == "train/validation":
        split_value = config.trainvalidation_split
    else:
        split_value = config.traintest_split
    train_data, test_validation_data = random_split(data_loader.dataset, [split_value, 1-split_value], generator)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, drop_last=False) 
    test_validation_loader = DataLoader(test_validation_data, batch_size=config.batch_size, shuffle=True, drop_last=False) 
    
    return train_loader, test_validation_loader


""" Returns a list of videos, which will be used to load data during each training batch.
"""
def load_video_list(config):
    videos_dir_path = config.data_path + "/" + config.sampletype 
    
    # Load list of files in directories
    video_list = os.listdir(videos_dir_path)
    
    # Keep only the folders in video_list to get all video names
    video_list = [entry for entry in video_list if os.path.isdir(os.path.join(videos_dir_path, entry))]

    # Shuffle list to randomize
    random.shuffle(video_list)
    
    # Use as much videos as predefined
    video_list = video_list[:config.dataset_size]
    
    # Split the video list in a train and test list
    train_list, test_list = split_video_list(config, video_list, split_type="train/test")
    
    return train_list, test_list

def load_skin_tone_video_list(config):
    videos_dir_path = config.data_path + "/skin_tone_labels"
        
    # Load list of files in directory
    video_list = os.listdir(videos_dir_path)
    video_list = [video[:-5] for video in video_list if video.endswith(".json")]
    
    # Remove all samples with annotation "Questionable"
    # TODO: Remove all violent samples (use neutral samples only)?
    video_dict = {}
    for video in video_list:
        path = config.data_path + "/skin_tone_labels/" + video + ".json"
        # Consider the name of the source video rather than the video itself. We need to
        # split into train/test based on the source rather than the video.
        source_video = video.split('.mp4')[0]
        with open(path, 'r') as annotation_file:
            annotation = json.load(annotation_file)
            if annotation['class_label'] != 'Questionable':
                if source_video in video_dict:
                    video_dict[source_video].append(video)
                else:
                    video_dict[source_video] = [video]

    # Split videos into train and test sets
    # Keeps into account that videos originating from the same source (in the keys) 
    # should all be in either train or test, not both.
    source_videos = list(video_dict.keys())
    random.shuffle(source_videos)  
    # Partition the source videos into train/test split
    train_size = int(config.traintest_split * len(source_videos))
    train_videos = source_videos[:train_size]
    test_videos = source_videos[train_size:]
    
    # Create the final train and test lists, with the data stored in the video_dict
    train_list = []
    test_list = []

    for video in train_videos:
        train_list.extend(video_dict[video])

    for video in test_videos:
        test_list.extend(video_dict[video])
        
    return train_list, test_list
        

""" Splits the already shuffled video list into a train/test or train/validation split, dependent on the type.
"""
def split_video_list(config, video_list, split_type="train/validation"):
    # Decide what the split value is, determined by either the train/validation or the train/test split.
    if split_type == "train/validation":
        split_value = config.trainvalidation_split
    else:
        split_value = config.traintest_split
        
    # Find the split index
    split = int(len(video_list) * split_value)
    
    # Split the list into train/test or train/validation lists
    train_list = video_list[:split]
    test_validation_list = video_list[split:]
    
    return train_list, test_validation_list

""" Splits the videos in config.video_path directory into images, stores them on disk
"""
def split_video_to_images(config):    
    video_list = os.listdir(config.video_path)
    video_list = [video for video in video_list if not video.startswith(".") and video.endswith(".mp4")]
            
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
    Images are in range [0,1].
"""
def normalize_images(config, images):
    channel1, channel2, channel3 = images[:, 0, :, :], images[:, 1, :, :], images[:, 2, :, :]
    if channel1.max().item() > 255.0 or channel2.max().item() > 255.0 or channel3.max().item() > 255.0:
        print(f"{channel1.min().item() = }{channel1.max().item() = }\n{channel2.min().item() = }{channel2.max().item() = }\n{channel3.min().item() = }{channel3.max().item() = }")
        print(f"WARNING: there is a value larger than 255! Should not happen. Colour_space:{config.colour_space}")
    
    normalized_images = images/255.0
    
    return normalized_images

""" Normalizes values in the passed videos.
"""
def normalize_videos(config, videos):
    for i in range(videos.shape[0]):
        frames = videos[i]
        videos[i] = normalize_images(config, frames)
    return videos
        
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
        green_hsv = cv2.cvtColor(np.uint8([[green_bgr]]), cv2.COLOR_BGR2HSV_FULL)[0][0]
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
    
    
""" Returns the values of the confusion matrix of true negative, false negative, true positive and false positive values.
    Receives the outputs as values in the range [0,1] and targets of either 0 or 1.
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
    
    f1_score = f_beta_score(tp, fp, fn, 1)
    f2_score = f_beta_score(tp, fp, fn, 2)
    
    IoU = tp/(tp+fp+fn)
    
    return accuracy, fn_rate, fp_rate, sensitivity, f1_score, f2_score, IoU

def regression_metrics(outputs, targets):
    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    # Calculate the mean absolute error and mean squared error 
    abs_diff = np.abs(targets - outputs)
    squared_diff = np.square(targets - outputs)
    mae = np.mean(abs_diff)
    mse = np.mean(squared_diff)
    
    return mae, mse
    

# New function
# squared_diff = np.square(target - predictions)

""" Calculates and returns the F-beta score given the passed tp, fp and fn rates.
"""
def f_beta_score(tp, fp, fn, beta):
    numerator = (1 + np.square(beta)) * tp
    denominator = (1 + np.square(beta)) * tp + fp + (np.square(beta) * fn)
    return numerator/denominator


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


""" Copies images from a folder into a new folder. Note that the images are renamed before
    storing.
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