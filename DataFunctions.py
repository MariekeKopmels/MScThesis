import os
import cv2
import sklearn.metrics
import torch 
import numpy as np 
from torch.utils.data import DataLoader
import time

""" Loads the requeste images, returns them in a Torch tensor.
"""
def load_images(config, dir_list, dir_path, gts=False):
    if gts: 
        images = torch.empty(len(dir_list), config.dims, config.dims, dtype=torch.float32)
    else:
        images = torch.empty(len(dir_list), 3, config.dims, config.dims, dtype=torch.float32)

    for i, file_name in enumerate(dir_list):
        # Read the images
        path = dir_path + "/" + file_name
        
        img = cv2.imread(path)
        
        # Convert image from RGB to YCrCb if needed
        if config.colour_space == "YCrCb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    
        # Convert Ground Truth from RGB to 1 channel (Black or White)
        if gts: 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _,img = cv2.threshold(img,127,1,0)
        
        # Resize images and ground truths to size 224*224
        img = cv2.resize(img, (config.dims,config.dims), interpolation=cv2.INTER_CUBIC)

        if gts:
            images[i] = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        else:
            images[i] = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            
    return images

"""Returns input in a given directory
        Format of return: torch tensors containin images and corresponding ground truths.
        Torch tensors are of shape batch_size,3,224,224 for images and batch_size,224,224 for ground truths 
"""
def load_input_images(config, image_dir_path, gt_dir_path, stage):
    # Load list of files and directories
    image_list = os.listdir(image_dir_path)
    gt_list = os.listdir(gt_dir_path)
    
    # Not all images have a ground truth, select those that do. Also skip the hidden files.
    dir_list = [file for file in image_list if file in gt_list and not file.startswith(".")]
    
    # print(f"Number of examples in the dataset: {len(dir_list)}\nGiven that train_size:{config.train_size}, validation_size:{config.validation_size} and test_size:{config.test_size}")
    
    # Include as many items as requested. test and validation are both from the test set and should not overlap.
    if stage == "train":
        dir_list = dir_list[:config.train_size]
    elif stage == "validation":
        #TODO: terugzetten als ik weer 1 dataset voor train en validation gebruik (K-fold)
        # end = config.test_size + config.validation_size
        # end = config.validation_size
        # if end > len(dir_list):
        #     print(f"in stage validation, should be of size {config.validation_size} but is of size {len(dir_list)}")
            # raise Exception(f"Test ({config.test_size}) and validation ({config.validation_size}) are larger than the test set of size 1157 (for VisuAAL).")
        #TODO: terugzetten als ik weer 1 dataset gebruik
        # dir_list = dir_list[config.test_size:end]
        dir_list = dir_list[:config.validation_size]
    elif stage == "test":
        dir_list = dir_list[:config.test_size]
    
    images = load_images(config, dir_list, image_dir_path)
    gts = load_images(config, dir_list, gt_dir_path, gts=True)
    
    return images, gts

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
                # TODO: temp, eruit halen
                # frame = cv2.resize(frame, (config.dims,config.dims), interpolation=cv2.INTER_CUBIC)
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


def merge_images_to_video(config):
    video_list = os.listdir(config.grinch_path)
    video_list = [video for video in video_list if not video.startswith(".") and not video.endswith(".mp4")]
    video_list.sort()
    print("videolist: ", video_list)
    for video in video_list:
        os.chdir(f"{config.grinch_path}")
        
        image_list = os.listdir(f"{config.grinch_path}/{video}")
        image_list = [image for image in image_list if not image.startswith(".")]
        image_list.sort()
        # print("image_list: ", image_list)
        
        if config.log: 
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


"""Returns train and test, both being data loaders with tuples of input images and corresponding ground truths.
        Format of return: Data loaders of tuples containging an input tensor and a ground truth tensor
        Image tensors are of shape (batch_size, channels, height, width)
"""
def load_image_data(config):
    base_path = config.data_path
    print("Loading training data...")
    train_images, train_gts = load_input_images(config, base_path + "/TrainImages", base_path + "/TrainGroundTruths", stage = "train")
    print("Loading validation data...")
    base_path = config.testdata_path
    # validation_images, validation_gts = load_input_images(config, base_path + "/TestImages", base_path + "/TestGroundTruth", stage = "validation")
    validation_images, validation_gts = load_input_images(config, base_path + "/ValidationImages", base_path + "/ValidationGroundTruths", stage = "validation")
    print("Loading testing data...")
    test_images, test_gts = load_input_images(config, base_path + "/TestImages", base_path + "/TestGroundTruths", stage = "test")

    # Combine images and ground truths in TensorDataset format
    train = torch.utils.data.TensorDataset(train_images, train_gts)
    validation = torch.utils.data.TensorDataset(validation_images, validation_gts)
    test = torch.utils.data.TensorDataset(test_images, test_gts)

    # Put data into dataloaders
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True, drop_last=True)
    validation_loader = DataLoader(validation, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True, drop_last=True)
    
    return train_loader, validation_loader, test_loader


def load_pixel_data(config, train, test, YCrCb = False):
    path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/Skin_NonSkin_Pixel.txt"
    print("Loading data...") 
    with open(path) as file:
       lines = [[int(num) for num in line.split()] for line in file]
       
    lines = np.array(lines)
    np.random.shuffle(lines)    
    
    # Balance the dataset
    lines_1 = lines[lines[:, 3] == 1]
    lines_2 = lines[(lines[:, 3] == 2)]
    lines = np.concatenate([lines_1, lines_2[:lines_1.shape[0]]], axis=0)
    np.random.shuffle(lines)
    print(f"Lines_1: {len(lines_1)} and Lines_2: {len(lines_2)}")
        
    pixels = lines[:,:3] 
    labels = lines[:,3]
    
    if YCrCb:
        print("YCrCb colour space, so converting.")
        B = pixels[:, 0]
        G = pixels[:, 1]
        R = pixels[:, 2]
        
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cr = (R - Y) * 0.713 + 128
        Cb = (B - Y) * 0.564 + 128
        
        pixels_YCrCb = np.column_stack((Y, Cr, Cb))
        print(f"Pixels: \n{pixels}\n\n and YCrCb pixels: \n{pixels_YCrCb}")
        pixels = pixels_YCrCb
    
    # Labels are 1 and 2, so transform to 0 and 1
    labels = labels - 1
    
    split = config.train_size
    end = config.train_size + config.test_size
    train_pixels = pixels[:split,:]
    test_pixels =  pixels[split:end,:]
    train_labels = labels[:split].reshape(-1, 1)
    test_labels =  labels[split:end].reshape(-1, 1)
    
    # TODO: Figure out waarom train wel maar test niet exact balanced is
    print(f"Balanced data? train labels info: {np.unique(train_labels, return_counts=True)}")
    print(f"Balanced data? test labels info: {np.unique(test_labels, return_counts=True)}")
    
    train = torch.utils.data.TensorDataset(torch.tensor(train_pixels), torch.tensor(train_labels))
    test = torch.utils.data.TensorDataset(torch.tensor(test_pixels), torch.tensor(test_labels))
    
    # Put data into dataloader
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, test_loader

def normalize_images(config, images):
    # Imagenet normilization 
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(config.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(config.device)
    images = ((images/255) - mean) / std
    
    return images

''' Returns the grinch version of the image, based on the given output of the model
'''
def make_grinch(image, output):
    # print(f"Input types: {type(image)}, and {type(output)}")
    if type(output) == torch.Tensor:
        output = output.to("cpu").numpy()
    # print(f"New input types: {type(image)}, and {type(output)}")
    grinch = np.copy(image)
    mask = output == 1
    grinch[mask] = [0,255,0]
    
    # print(f"Grinch type: {type(grinch)}")
    # print(f"Mask type: {type(mask)}")
    
    return grinch

# Takes baches of images in ndarrays, stores and returns the grinch versions
def to_grinches(config, images, outputs, video):
    outputs = outputs.cpu().numpy()
    grinches = np.copy(images.cpu().numpy())
    mask = outputs == 1
    
    os.chdir(config.grinch_path)
    os.makedirs(video, exist_ok=True)
    
    # print(f"Grinches dims: {np.shape(grinches)}, mask dims: {np.shape(mask)}, outputs dims: {np.shape(outputs)}")
    for i in range(len(mask)):
        # print(f"Grinches[{i}] dims: {np.shape(grinches[i])}, outputs[{i}] dims: {np.shape(outputs[i])}")
        # print(f"ptp: {np.ptp(outputs[i])}")
        grinches[i] = make_grinch(grinches[i].transpose(1,2,0), outputs[i]).transpose(2,0,1)
        save_path = config.grinch_path + "/" + video
        save_name = "grinchframe_" + str(i).zfill(5) + ".jpg"
        if config.log:
            save_image(config, grinches[i].transpose(1,2,0), save_path, save_name)
    return torch.from_numpy(grinches)
    
    
"""Returns the values of the confusion matrix of true negative, false negative, true positive and false positive values
"""
def confusion_matrix(config, outputs, targets, stage):
    start_time = time.time()
    
    # Flatten output and target
    output = torch.flatten(outputs)
    target = torch.flatten(targets)
    
    # Binarize output of the model
    output = output > 0.5
    
    # Make output floats, split up in parts if confusion matrix parts is larger than 1
    if stage == "train" and config.cm_parts > 1:
        total_elements = output.numel()
        part_size = total_elements/config.cm_parts
        parts = torch.Tensor((0)).to(config.device)
        for i in range(config.cm_parts):
            start_idx = int(i * part_size)
            end_idx = int(min((i + 1) * part_size, total_elements))
            chunk = output[start_idx:end_idx].float()
            parts = torch.cat((parts, chunk))
        output = parts
    else:
        output.float()
    
    # Compute the confusion matrix
    matrix = sklearn.metrics.confusion_matrix(target.to("cpu"), output.to("cpu"))
    duration = time.time() - start_time
    # print("Matrix calculation finished in %.2f seconds." % duration)
    
    tn = matrix[0][0]
    fn = matrix[1][0]
    tp = matrix[1][1]
    fp = matrix[0][1]
    
    return tn, fn, fp, tp

"""Returns the values of the confusion matrix of true negative, false negative, true positive and false positive values
"""
def pixel_confusion_matrix(config, outputs, labels, test=False):    
    outputs = outputs.to("cpu")
    outputs = outputs.detach().numpy()
    labels = labels.to("cpu")
    labels = labels.detach().numpy().flatten()
        
    outputs = np.array([1.0 if x > 0.5 else 0.0 for x in outputs])
    # labels = np.array([0 if x == 1 else 2 for x in labels])
    # print(f"Values in outputs: {list(set(outputs))} and in labels: {labels.unique()}")
    # print(f"labels: {labels}, outputs: {outputs}")
    matrix = sklearn.metrics.confusion_matrix(labels, outputs)
    
    if test:
        print("Confusion matrix:\n", matrix)
        
    # Check if all entries are either 0 or all are 1 (resulting in one value as matrix)
    if matrix.shape == (1, 1):
        output_matrix = [[0.0, 0.0],[0.0, 0.0]]
        if outputs[0] == 1.0:
            output_matrix[1][1] = matrix[0][0]
        elif outputs[0] == 0.0:
            output_matrix[0][0] = matrix[0][0]
        else:
            print(f"QUEeeee? outputs[0]:{outputs[0]} so outputs[0] == 1.0:{outputs[0] == 1.0} and matrix[0][0]:{matrix[0][0]}")
        matrix = output_matrix

    tn = matrix[0][0]
    fn = matrix[1][0]
    tp = matrix[1][1]
    fp = matrix[0][1]
    
    return tn, fn, fp, tp

""" Computes metrics based on true/false positive/negative values.
        Returns accuracy, fn_rate, fp_rate and sensitivity.
"""
def metrics(tn, fn, fp, tp, pixels=False):
    # print("Calculating metrics")
    accuracy = (tp+tn)/(tn+fn+fp+tp)
    # If there are no pixels that should be marked as skin, the fn_rate should be 0
    fn_rate = fn/(fn+tp) if fn+tp != 0 else 0
    # None of the images are all skin, but added a safety measure
    fp_rate = fp/(tn+fp) if tn+fp != 0 else 0
    # If there are no pixels that should be marked as skin, the sensitivity should be 1
    sensitivity = tp/(fn+tp) if fn+tp!= 0 else 1
    
    f1_score = tp/(tp+((fp+fn)*0.5))
    IoU = tp/(tp+fp+fn)
        
    if pixels: 
        return accuracy, fn_rate, fp_rate, sensitivity, f1_score

    return accuracy, fn_rate, fp_rate, sensitivity, f1_score, IoU


""" Stores an image (in form of ndarray) to the disk.
"""
# TODO: remove config as input
def save_image(config, image, path, filename, bw=False, gt=False):
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    # print(f"type(image): {type(image)}, shape: {image.shape}")
    # cv2.imwrite takes input in form height, width, channels
    if type(image) == torch.Tensor:
        # print("Shape: ", image.shape)
        # image = image.permute(1,2,0)
        image = image.to("cpu")
        if gt:
            image = cv2.cvtColor(image.numpy(), cv2.COLOR_GRAY2BGR)
            # print("New Shape: ", np.shape(image))
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
        if config.log:
            save_name = "image_" + str(index).zfill(6) + ".jpg"
            save_image(config, image, destination_path, save_name, gt=gts)
            index += 1
            
    return index