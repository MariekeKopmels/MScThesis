import os
import json

# Define the path to the 'annotated' folder
annotated_folder = '/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/TESTFOLDER-OddityData-1-refined-nms-data-0103/annotated'
json_folder = '/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/TESTFOLDER-OddityData-1-refined-nms-data-0103/labels'

# Initialize the result dictionary
result = {}

# Iterate through the subfolders ('black', 'white', 'unidentifyable')
for class_folder in os.listdir(annotated_folder):
    # print(f"{class_folder = }")
    
    if class_folder == '.DS_Store':
        continue
        
    class_path = os.path.join(annotated_folder, class_folder)
    print(f"{class_path = }")
    
    # Iterate through video folders inside each class folder
    for video_folder in os.listdir(class_path):
        if video_folder == '.DS_Store':
            continue
        # print(f"{video_folder = }")
        
        # Video_name not used, this would alter 
        # 01_office_inside.mp4_id-0_tbsf-0_bbox-452-661-559-1064 to 01_office_inside
        # which I don't want.  
        
        # video_name = os.path.splitext(video_folder)[0]
        video_json_path = os.path.join(json_folder, f'{video_folder}.json')
        
        print(f"{video_json_path = }")
        
        # Read the JSON file for the video
        with open(video_json_path, 'r') as json_file:
            video_data = json.load(json_file)
        
        # Not necessary, now uses the class_folder as class
        # Determine the ShoeColour class based on the parent folder
        # if class_folder == 'black':
        #     shoe_colour_class = 'Black'
        # elif class_folder == 'white':
        #     shoe_colour_class = 'White'
        # else:
        #     shoe_colour_class = 'Unidentifiable'

        # Create the video entry in the result dictionary
        result[video_folder] = {
            'Violence': {
                'Class': video_data['class_label'][0]
            },
            'ShoeColour': {
                'Class': class_folder
                # 'Class': shoe_colour_class
            }
        }

# Create the final JSON output file
os.chdir("/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/marieke-copy-refined-nms-data-0103/")
with open('output.json', 'w') as output_file:
    json.dump(result, output_file, indent=4)

print("JSON file 'output.json' created successfully!")
