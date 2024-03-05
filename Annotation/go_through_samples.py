import os
import cv2
import math

import shutil

def sort_filenames(filenames):
    # Group filenames by video ID
    video_groups = {}
    for filename in filenames:
        video_id = filename.split("_id-")[0]
        if video_id not in video_groups.keys():
            video_groups[video_id] = []
        video_groups[video_id].append(filename)
    
    # Sort each group based on the ID value
    sorted_filenames = []
    for video_id in sorted(video_groups.keys()):
        sorted_group = sorted(video_groups[video_id], key=lambda x: int(x.split("_id-")[1].split("_")[0]))
        sorted_filenames.extend(sorted_group)
    
    return sorted_filenames


def main():

    prefix = "extract"

    sample_list = os.listdir(prefix)
    sample_list = sort_filenames(sample_list)

    for sample in sample_list:
        sample_name = os.path.basename(sample)
        total_path = os.path.join(prefix, sample_name)
        total_path = total_path.replace("/n", "")
        loop = True

        while loop:
            for img in os.listdir(total_path):
                if '.txt' in img:
                    continue
                img_path = os.path.join(total_path, img)
                frame = cv2.imread(img_path)
                cv2.imshow('viewer', resize(frame))
                key = cv2.waitKey(30)
                if key == ord('v'):
                    shutil.move(total_path, "C:/Users/luukv/Desktop/ViolenceWinter2023-2024/label_possible_violence_from_unused/violence")
                    print(sample_name)
                    loop = False
                    break 
                elif key == ord('q'):
                    shutil.move(total_path, "C:/Users/luukv/Desktop/ViolenceWinter2023-2024/label_possible_violence_from_unused/questionable")
                    loop = False
                    print(sample_name)
                    break
                elif key == ord('n'):
                    shutil.move(total_path, "C:/Users/luukv/Desktop/ViolenceWinter2023-2024/label_possible_violence_from_unused/neutral")
                    loop = False
                    print(sample_name)
                    break

def resize(frame, size=300):
    width, heigth = frame.shape[:2]

    if heigth < width:
        scale_factor = size / heigth
        return cv2.resize(frame, (math.ceil(heigth * scale_factor), math.ceil(width * scale_factor)))
    else:
        scale_factor = size / width
        return cv2.resize(frame, (math.ceil(heigth * scale_factor), math.ceil(width * scale_factor)))


if __name__ == "__main__":
    main()