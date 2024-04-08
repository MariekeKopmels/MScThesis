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
    base_path = "/Users/marieke/Desktop/MScThesis/Datasets/official_dataset/"
    sample_prefix = "02_samples"
    os.chdir(base_path)
    
    # with open('trainlist.txt') as trainlist:
    #     sample_list = trainlist.readlines()
    #     sample_list = sort_filenames(sample_list)
    #     trainlist.close()
    #     print(f"Samplelist: {sample_list}")

    sample_path = os.path.join(base_path, sample_prefix)
    sample_list = os.listdir(sample_path)
    sample_list.sort()
    print(f"Length of sample_list: {len(sample_list)}")

    for sample in sample_list:
        sample_name = os.path.basename(sample)
        if sample_name == ".DS_Store":
            continue
        print(f"{sample_name = }")
        total_path = os.path.join(sample_path, sample_name)
        total_path = total_path.replace("\n", "")
        print(f"Total path: {total_path}")
        loop = True

        while loop:
            # Checken of hij de video kan vinden, anders skippen en naar de volgende
            if os.path.isdir(total_path):
                # print("path found! ")
                image_list = os.listdir(total_path)
                image_list = [image for image in image_list if not '.txt' in image]
                image_list.sort()
                if len(image_list) == 0:
                    loop = False
                for img in image_list:
                    if img.startswith(".") and img.endswith(".cloud"):
                        img = img[1:]
                        img = img[:-6]
                    img_path = os.path.join(total_path, img)
                    frame = cv2.imread(img_path)
                    cv2.moveWindow('viewer', 600, 30)
                    cv2.imshow('viewer', resize(frame))
                    key = cv2.waitKey(30)
                    if key == ord('q') or key == ord('0'):
                        shutil.move(total_path, "03_annotated_samples/0-Questionable")
                        loop = False
                        print(sample_name)
                        break
                    elif key == ord('w') or key == ord('1'):
                        shutil.move(total_path, "03_annotated_samples/1-White")
                        print(sample_name)
                        loop = False
                        break 
                    elif key == ord('l') or key == ord('2'):
                        shutil.move(total_path, "03_annotated_samples/2-LightBrown")
                        loop = False
                        print(sample_name)
                        break
                    elif key == ord('m') or key == ord('3'):
                        shutil.move(total_path, "03_annotated_samples/3-MediumBrown")
                        loop = False
                        print(sample_name)
                        break
                    elif key == ord('d') or key == ord('4'):
                        shutil.move(total_path, "03_annotated_samples/4-DarkBrown")
                        loop = False
                        print(sample_name)
                        break
                    elif key == ord('b') or key == ord('5'):
                        shutil.move(total_path, "03_annotated_samples/5-Black")
                        loop = False
                        print(sample_name)
                        break
                    elif key == ord('v') or key == ord('6'):
                        shutil.move(total_path, "03_annotated_samples/6-Violence")
                        loop = False
                        print(sample_name)
                        break
                    elif key == 27:
                        print("Exited annotation!")
                        loop = False
                        return
                        
            else:
                loop = False

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
    