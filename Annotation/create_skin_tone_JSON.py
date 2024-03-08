import os
import json

# Define the path to the 'annotated' folder
annotated_folder = '/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/NEWOFFICIAL_OddityData-1-refined-nms-data-0103/annotated'
json_folder = '/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/NEWOFFICIAL_OddityData-1-refined-nms-data-0103/skin_tone_labels'

# Initialize the result dictionary
result = {}

# Iterate through the subfolders ('black', 'white', 'unidentifyable')
class_list = os.listdir(annotated_folder)
class_list.sort()
for class_folder in class_list:    
    if class_folder == '.DS_Store':
        continue
        
    class_path = os.path.join(annotated_folder, class_folder)
    print(f"{class_path = }")
    
    # Iterate through video folders inside each class folder
    video_list = os.listdir(class_path)
    video_list.sort()
    for video_folder in video_list:
        if video_folder == '.DS_Store':
            continue
        
        # Determine the skin tone class based on the parent folder
        if class_folder == '0-Questionable':
            skin_tone_class = "Questionable"
        elif class_folder == '1-White':
            skin_tone_class = "White"
        elif class_folder == '2-LightBrown':
            skin_tone_class = "LightBrown"
        elif class_folder == '3-MediumBrown':
            skin_tone_class = "MediumBrown"
        elif class_folder == '4-DarkBrown':
            skin_tone_class = "DarkBrown"
        elif class_folder == '5-Black':
            skin_tone_class = "Black"
        else:
            raise ValueError('Class not recognised!')

        # Create the resulting JSON file
        result = {"class_label": skin_tone_class}
        os.makedirs(json_folder, exist_ok=True)
        output_path = os.path.join(json_folder, f'{video_folder}.json')
        with open(output_path, 'w') as output_file:
            json.dump(result, output_file)
