from types import SimpleNamespace
import Data.DataFunctions as DataFunctions
import os
import random


default_config = SimpleNamespace(
    dims = 224,  
    colour_space = "RGB",
    log = True,
    train_split = 0.85,
    validation_split = 0.05,
    test_split = 0.1,
    dataset_folder = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/All_Skin_Datasets",
    combined_dataset_folder = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/LargeCombinedDataset",
)

# TODO: Use train/val/test splits by sklearn
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
def check_splits(config):
    total = config.train_split + config.validation_split + config.test_split
    if config.train_split + config.validation_split > 1.0:
        raise Exception(f"Train {config.train_split} and validation {config.validation_split} split are too large, will cause test set to be non-existant.")
    if total > 1.0:
        print(f"Warning: One or more splits are too large as currently {config.train_split} + {config.validation_split} + {config.test_split} > 1.0. Will cause the test split to be be cut-off!")
    if total < 1.0:
        print(f"Warning: Not all data from the dataset is used. {config.train_split} + {config.validation_split} + {config.test_split} < 1.0")

""" Combines all datasets into one
"""
def merge_datasets(config):
    folderlist = os.listdir(config.dataset_folder)
    folderlist = [folder for folder in folderlist if not folder.startswith(".")]
    # The abdomen dataset is not relevant for the current research
    folderlist.remove("Dataset8_Abdomen")
    folderlist.sort()
    
    # current_dataset_start_id refers to the  ID that is used for the data splitting such that all samples 
    # get a unique ID, identical for an image and its corresponding  ground truth.
    # TODO: Herschrijven naar een nettere manier als hier tijd voor is, dus zonder current_dataset_start_id.
    current_dataset_start_id = 0
    for folder in folderlist:
        print(f"Now processing folder {folder}")
        origin_image_path = config.dataset_folder + "/" + folder + "/original_images"
        destination_path = config.combined_dataset_folder + "/AllImages"
        _ = DataFunctions.move_images(config, current_dataset_start_id, origin_image_path, destination_path)
        origin_gt_path = config.dataset_folder + "/" + folder + "/skin_masks"
        destination_path = config.combined_dataset_folder + "/AllGroundTruths"
        current_dataset_start_id = DataFunctions.move_images(config, current_dataset_start_id, origin_gt_path, destination_path, gts=True) 
    return

""" Split the combined dataset into a train, validation and test dataset
"""
def split_dataset(config):
    image_path = config.combined_dataset_folder + "/AllImages"
    train_image_destination = config.combined_dataset_folder + "/TrainImages"
    validation_image_destination = config.combined_dataset_folder + "/ValidationImages"
    test_image_destination = config.combined_dataset_folder + "/TestImages"
    
    gt_path = config.combined_dataset_folder + "/AllGroundTruths"
    train_gt_destination = config.combined_dataset_folder + "/TrainGroundTruths"
    validation_gt_destination = config.combined_dataset_folder + "/ValidationGroundTruths"
    test_gt_destination = config.combined_dataset_folder + "/TestGroundTruths"
    
    image_list = os.listdir(image_path)
    image_list = [image for image in image_list if not image.startswith(".")]
    random.shuffle(image_list)
    
    train_end_index = int(len(image_list) * config.train_split)
    validation_end_index = int(train_end_index + len(image_list) * config.validation_split)
    test_end_index = len(image_list)

    # current_dataset_start_id refers to the  ID that is used for the data splitting such that all samples 
    # get a unique ID, identical for an image and its corresponding  ground truth.
    # TODO: Herschrijven naar een nettere manier als hier tijd voor is, dus zonder start_id.
    current_dataset_start_id = 0
    print("Splitting dataset into train...")
    _ = DataFunctions.move_images(config, current_dataset_start_id, image_path, train_image_destination, image_list[:train_end_index])
    current_dataset_start_id = DataFunctions.move_images(config, current_dataset_start_id, gt_path, train_gt_destination, image_list[:train_end_index], gts=True)
    print("validation...")
    _ = DataFunctions.move_images(config, current_dataset_start_id, image_path, validation_image_destination, image_list[train_end_index:validation_end_index])
    current_dataset_start_id = DataFunctions.move_images(config, current_dataset_start_id, gt_path, validation_gt_destination, image_list[train_end_index:validation_end_index], gts=True)
    print("and test...")
    _ = DataFunctions.move_images(config, current_dataset_start_id, image_path, test_image_destination, image_list[validation_end_index:test_end_index])
    current_dataset_start_id = DataFunctions.move_images(config, current_dataset_start_id, gt_path, test_gt_destination, image_list[validation_end_index:test_end_index], gts=True)

""" Complete pipeline of combining datasets into a new dataset.
"""
def datasetcreation_pipeline(config):
    check_splits(config)
    merge_datasets(config)
    split_dataset(config)

    return

if __name__ == '__main__':
    datasetcreation_pipeline(default_config)
    