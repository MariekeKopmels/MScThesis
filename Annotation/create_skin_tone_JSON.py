import os
import json

# Define the path to the 'annotated' folder
annotated_folder = '/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/NEWOFFICIAL_OddityData-1-refined-nms-data-0103/annotated'
# json_folder = '/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/NEWOFFICIAL_OddityData-1-refined-nms-data-0103/labels'

# Initialize the result dictionary
result = {}

# Iterate through the subfolders ('black', 'white', 'unidentifyable')
class_list = os.listdir(annotated_folder)
class_list.sort()
for class_folder in class_list:
    # print(f"{class_folder = }")
    
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
        
        # video_json_path = os.path.join(json_folder, f'{video_folder}.json')
        
        # print(f"{video_json_path = }")
        
        # # Read the JSON file for the video
        # with open(video_json_path, 'r') as json_file:
        #     video_data = json.load(json_file)
        
        # Determine the ShoeColour class based on the parent folder
        if class_folder == '0-Questionable':
            skin_tone_class = 0.0
        elif class_folder == '1-White':
            skin_tone_class = 1.0
        elif class_folder == '2-LightBrown':
            skin_tone_class = 2.0
        elif class_folder == '3-MediumBrown':
            skin_tone_class = 3.0
        elif class_folder == '4-DarkBrown':
            skin_tone_class = 4.0
        elif class_folder == '5-Black':
            skin_tone_class = 5.0
        else:
            raise ValueError('Class not recognised!')

        # Create the video entry in the result dictionary
        result[video_folder] = {
            'SkinTone': {
                'Class': skin_tone_class
            }
        }

# Create the final JSON output file
os.chdir("/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/NEWOFFICIAL_OddityData-1-refined-nms-data-0103/")
with open('skin_tone_labels.json', 'w') as output_file:
    json.dump(result, output_file, indent=4)

print("JSON file 'skin_tone_labels.json' created successfully!")
