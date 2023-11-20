import os
import random
import cv2
import sklearn.metrics
import torch 
import numpy as np 
from torch.utils.data import DataLoader

NO_PIXELS = 224

"""Returns images in a given directory
        Format of return: torch tensor.
        Torch tensors are of shape batch_size,3,224,224 for images and batch_size,224,224 for ground truths 
"""
def load_images(config, image_dir_path, gt_dir_path, type):
    # Load list of files and directories
    image_list = os.listdir(image_dir_path)
    gt_list = os.listdir(gt_dir_path)
    
    # Not all images have a ground truth, select those that do. Also skip the hidden files.
    dir_list = [file for file in image_list if file in gt_list and not file.startswith(".")]
    
    # Include as many items as requested. test and validation are both from the test set and should not overlap.
    if type == "train":
        dir_list = dir_list[:config.train_size]
    elif type == "validation":
        end = config.test_size + config.validation_size
        if end > len(dir_list):
            raise Exception(f"Test ({config.test_size}) and validation ({config.validation_size}) are larger than the test set of size 1157.")
        dir_list = dir_list[config.test_size:end]
    elif type == "test":
        dir_list = dir_list[:config.test_size]
            
    # Store images and ground truths in tensor format
    images = torch.empty(len(dir_list), 3, NO_PIXELS, NO_PIXELS, dtype=torch.float32)
    gts = torch.empty(len(dir_list), NO_PIXELS, NO_PIXELS, dtype=torch.float32)
    
    for i, file_name in enumerate(dir_list):
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
        
        # Add to tensor
        images[i] = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        gts[i] = torch.tensor(gt, dtype=torch.float32).unsqueeze(0)
        
    return images, gts

"""Returns train and test, both being data loaders with tuples of input images and corresponding ground truths.
        Format of return: Data loaders of tuples containging an input tensor and a ground truth tensor
        Image tensors are of shape (batch_size, channels, height, width)
"""
def load_data(config, train, test):
    if config.machine == "Mac":
        base_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/visuAAL"
    else: 
        base_path = "/home/oddity/marieke/Datasets/VisuAAL"

    print("Loading training data...")
    train_images, train_gts = load_images(config, base_path + "/TrainImages", base_path + "/TrainGroundTruth", type = "train")
    print("Loading validation data...")
    validation_images, validation_gts = load_images(config, base_path + "/TestImages", base_path + "/TestGroundTruth", type = "validation")
    print("Loading testing data...")
    test_images, test_gts = load_images(config, base_path + "/TestImages", base_path + "/TestGroundTruth", type = "test")

    # Combine images and ground truths in TensorDataset format
    train = torch.utils.data.TensorDataset(train_images, train_gts)
    validation = torch.utils.data.TensorDataset(validation_images, validation_gts)
    test = torch.utils.data.TensorDataset(test_images, test_gts)

    # Put data into dataloaders
    train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
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
    
"""Returns the values of the confusion matrix of true negative, false negative, true positive and false positive values
"""
def confusion_matrix(outputs, targets, type):
    batch_size = outputs.shape[0]
    
    outputs = outputs.to("cpu")
    outputs = outputs.detach().numpy()
    targets = targets.to("cpu")
    targets = targets.detach().numpy()
    
    matrix = [[0.0, 0.0],[0.0, 0.0]]
    
    #Compute the confusion matrix
    for i in range(batch_size):
        output = outputs[i].flatten()
        target = targets[i].flatten()
        
        # print(f"\n\nBEFORE\n")
        # print(f"Number of 0's in output: {np.sum(output == 0.0)}")
        # print(f"Number of 0's in target: {np.sum(target == 0.0)}")
        # print(f"Number of 1's in output: {np.sum(output == 1.0)}")
        # print(f"Number of 1's in target: {np.sum(target == 1.0)}")
        # print(f"Output contents: {output[:20]}")
        # print(f"Target contents: {target[:20]}")

        # print(f"\n\nBEFORE\nNumber of 0's in output: {sum(1 for value in output if value < 0.5)}")
        # print(f"Number of 0's in target: {sum(1 for value in target if value < 0.5)}")
        # print(f"Number of 1's in output: {sum(1 for value in output if value > 0.5)}")
        # print(f"Number of 1's in target: {sum(1 for value in target if value > 0.5)}")        
        # output = np.array(output) > 0.5
        output = np.array([1.0 if x > 0.5 else 0.0 for x in output])
        
        # print(f"\nAFTER\n")
        # print(f"Number of 0's in output: {np.sum(output == 0.0)}")
        # print(f"Number of 0's in target: {np.sum(target == 0.0)}") 
        # print(f"Number of 1's in output: {np.sum(output == 1.0)}")
        # print(f"Number of 1's in target: {np.sum(target == 1.0)}")
        # print(f"Output contents: {output[:20]}")
        # print(f"Target contents: {target[:20]}")
        # print(f"\nAFTER\nNumber of 0's in output: {sum(1 for value in output if value < 0.5)}")
        # print(f"Number of 0's in target: {sum(1 for value in target if value < 0.5)}")
        # print(f"Number of 1's in output: {sum(1 for value in output if value > 0.5)}")
        # print(f"Number of 1's in target: {sum(1 for value in target if value > 0.5)}")
        
        # print(f"\nContents")
        # print(f"Min value in output: {min(output)}")
        # print(f"Max value in output: {max(output)}")
        # print(f"Unique values in output: {set(output)}")
        # print(f"types of target and output: {type(target)} and {type(output)}")

        i_matrix = sklearn.metrics.confusion_matrix(target, output)
        
        # Blijkbaar bij een target met alles 0, krijg je een [[50176]] confusion matrix. Bij de volgende is deze er dus bij alle 
        # vier de indexen bij opgeteld en klopt het totaal niet meer. 
        # print("Intermediate confusion matrix:\n", i_matrix)
        # print(f"\nNumber of <0.5's in output: {sum(1 for value in output if value < 0.5)}")
        # print(f"Number of <0.5's in target: {sum(1 for value in target if value < 0.5)}")
        # print(f"Number of >0.5's in output: {sum(1 for value in output if value > 0.5)}")
        # print(f"Number of >0.5's in target: {sum(1 for value in target if value > 0.5)}")
        
        # If all target and output entries are zero, i_matrix has only one dimension as all is true negative
        # print(f"Dims: {i_matrix.shape}")
        if i_matrix.shape == (1, 1):
            matrix[0][0] += i_matrix[0][0]
        else:
            matrix += i_matrix
    
    if type == "test":
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

# def mask_images(images, outputs):
#     images[outputs > 0.5] 
#     altered_images[mask == 1]
#     return

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