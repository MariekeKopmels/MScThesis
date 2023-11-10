import os
import cv2
import sklearn.metrics
import torch 
import numpy as np 
from torch.utils.data import DataLoader

NO_PIXELS = 224

"""Returns images in a given directory
        Format of return: list of NumPy arrays.
        NumPy arrays are of shape (224,224,3) for images and (224,224) for ground truths 
"""
def load_images(config, images, gts, image_dir_path, gt_dir_path, test=False):
    # Load list of files and directories
    image_list = os.listdir(image_dir_path)
    gt_list = os.listdir(gt_dir_path)
    
    # Not all images have a ground truth, select those that do
    dir_list = [file for file in image_list if file in gt_list]
    dir_list = sorted(dir_list, key=str.casefold)
    
    # Include as many items as requested
    if test:
        dir_list = dir_list[:config.test_size]
    else:
        dir_list = dir_list[:config.train_size]
    
    # Store images
    for file_name in dir_list:
        # Hidden files, irrelevant for this usecase
        if file_name.startswith('.'):
            continue
        # Read the images
        img_path = image_dir_path + "/" + file_name
        gt_path = gt_dir_path + "/" + file_name
        
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path)
        
        if config.colour_space == "YCrCb":
            # Convert image from RGB to YCrCb
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        
        # Convert Ground Truth from RGB to 1 channel (Black or White)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        _,gt = cv2.threshold(gt,127,1,0)
        
        # Resize images and ground truths to size 224*224
        img = cv2.resize(img, (NO_PIXELS,NO_PIXELS), interpolation=cv2.INTER_CUBIC)
        gt = cv2.resize(gt, (NO_PIXELS,NO_PIXELS), interpolation=cv2.INTER_CUBIC)
        
        #Store in list
        images.append(img)
        gts.append(gt)
        
    return images, gts

"""Returns train and test, both being data loaders with tuples of input images and corresponding ground truths.
        Format of return: Data loaders of tuples containging an input tensor and a ground truth tensor
        Image tensors are of shape (batch_size, channels, height, width)
"""
def load_data(config, train, test):
    base_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/visuAAL"

    train_images, train_gts, test_images, test_gts = [], [], [], []
    print("Loading training data...")
    train_images, train_gts = load_images(config, train_images, train_gts, base_path + "/TrainImages", base_path + "/TrainGroundTruth")
    print("Loading testing data...")
    test_images, test_gts = load_images(config, test_images, test_gts, base_path + "/TestImages", base_path + "/TestGroundTruth", test = True)
    
    # TODO: fix de eerst naar numpy en daarna pas naar tensor (sneller dan vanaf een list direct naar tensor maar nu heel lelijk)
    train = torch.utils.data.TensorDataset(torch.as_tensor(np.array(train_images)).permute(0,3,1,2), torch.as_tensor(np.array(train_gts)).permute(0,1,2))
    test = torch.utils.data.TensorDataset(torch.as_tensor(np.array(test_images)).permute(0,3,1,2), torch.as_tensor(np.array(test_gts)).permute(0,1,2))

    # Put data into dataloader
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, test_loader

def load_pixel_data(config, train, test):
    path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/Skin_NonSkin_Pixel.txt"
    train_pixels, train_label, test_pixels, test_label = [], [], [], []
    print("Loading data...")
    with open(path) as file:
       lines = [[int(num) for num in line.split()] for line in file]
    
    lines = np.array(lines)
    np.random.shuffle(lines)
    data_size = config.train_size+config.test_size
    lines = lines[:data_size,:]
    pixels = lines[:,:3] 
    labels = lines[:,3]
    
    # Labels are 1 and 2, so transform to 0 and 1
    labels = labels - 1

    split = config.train_size
    train_pixels = pixels[:split,:]
    test_pixels =  pixels[split:,:]
    train_labels = labels[:split].reshape(-1, 1)
    test_labels =  labels[split:].reshape(-1, 1)
    # print(f"Data put in the dataloader is of format [{len(train_labels)}, {train_labels[0]}].")
    # print(f"Dims: {train_pixels.shape}, {train_labels.shape}.")
    
    train = torch.utils.data.TensorDataset(torch.tensor(train_pixels), torch.tensor(train_labels))
    test = torch.utils.data.TensorDataset(torch.tensor(test_pixels), torch.tensor(test_labels))
    
    # Put data into dataloader
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, test_loader
    
"""Returns the values of the confusion matrix of true negative, false negative, true positive and false positive values
"""
def confusion_matrix(outputs, targets, test=False):
    batch_size = outputs.shape[0]
    
    outputs = outputs.to("cpu")
    outputs = outputs.detach().numpy()
    targets = targets.to("cpu")
    targets = targets.detach().numpy()
    
    matrix = [[0.0, 0.0],[0.0, 0.0]]
    
    #Compute the confusion matrix
    for i in range(batch_size):
        output = outputs[i].flatten()
        output = [1.0 if x > 0.5 else 0.0 for x in output]
        target = targets[i].flatten()
        
        # if test: 
        #     print(f"Number of 0's in output: {np.sum(output == 0)}")
        #     print(f"Number of 0's in target: {np.sum(target == 0)}")
        #     print(f"Number of 1's in output: {np.sum(output == 1)}")
        #     print(f"Number of 1's in target: {np.sum(target == 1)}")
        
        i_matrix = sklearn.metrics.confusion_matrix(target, output)
        matrix += i_matrix
    
    if test:
        print("Confusion matrix test:\n", matrix)
    
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
    
    tn = matrix[0][0]
    fn = matrix[1][0]
    tp = matrix[1][1]
    fp = matrix[0][1]
    
    return tn, fn, fp, tp

""" Computes metrics based on true/false positive/negative values.
        Returns accuracy, fn_rate, fp_rate and sensitivity.
"""
def metrics(tn, fn, fp, tp, pixels=False):
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

# """ Stores an image to the disk.
# """
# def save_image(filename, image, bw=False):
#     directory = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Output/WandB/"
#     os.chdir(directory)
    
#     # cv2.imwrite takes input in form height, width, channels
#     image = image.permute(1,2,0)
#     image = image.to("cpu")
#     if bw:
#         image = image*225
#     if type(image) != np.ndarray:
#         cv2.imwrite(filename, image.numpy())
#     else:
#         cv2.imwrite(filename, image)