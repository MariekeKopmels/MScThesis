import os
import json



def main():
    counter = 0 
    
    base_path = "/home/oddity/data/oddity-refined-data-0103"
    trainlist_path = "/home/oddity/data/oddity-refined-data-0103/testtrainlist.txt"
    
    with open(trainlist_path, "r") as file:
        trainlist = [line.strip() for line in file]
    
    reduced_trainlist = trainlist.copy()
    for example in trainlist:
        counter += 1
        temp = example.replace(".", "")
        path = base_path + temp
        print("path:", path)
        
        if os.path.isfile(path):
            print(f"Processing {file}")
            
            if counter == 5:
                reduced_trainlist.remove(example)
            
    print("Testing")
    test = "01_office_inside.mp4_id-0_tbsf-0_bbox-452-661-559-1064"
    path = base_path + test
    if os.path.isfile(path):
            print(f"Processing {file}")
    
    with open(trainlist_path, "w") as file:
        file.write("\n".join(reduced_trainlist))


























# def sort_filenames(filenames):
#     # Group filenames by video ID
#     video_groups = {}
#     for filename in filenames:
#         video_id = filename.split("_id-")[0]
#         if video_id not in video_groups.keys():
#             video_groups[video_id] = []
#         video_groups[video_id].append(filename)
    
#     # Sort each group based on the ID value
#     sorted_filenames = []
#     for video_id in sorted(video_groups.keys()):
#         sorted_group = sorted(video_groups[video_id], key=lambda x: int(x.split("_id-")[1].split("_")[0]))
#         sorted_filenames.extend(sorted_group)
    
#     return sorted_filenames

# def main():
#     base_path = "/home/oddity/data/oddity-refined-data-0103"
#     os.chdir(base_path)
    
#     trainlist_path = "/trainlist.txt"
#     samples_path = "/samples"
#     labels_path = "/labels"
#     new_labels_path = "/newlabels"
    
#     path = base_path + labels_path
#     labels_list = os.listdir(path)
#     labels_list = sort_filenames(labels_list)
    
#     path = path = base_path + new_labels_path
#     annotated_list = os.listdir(path)
#     annotated_list = sort_filenames(annotated_list)
    
#     counter = 0
#     for label_filename in labels_list:
#         print(label_filename)
#         label_filename = label_filename.replace(".json", "")
#         pwd = base_path + labels_path + "/" + label_filename
#         print("Pwd: ", pwd)
#         with open(label_filename) as file:
#             d = json.load(file)
#             print(d)
#         counter += 1
#         if label_filename in annotated_list:
#             # move to next sample
#             continue
        
#         if counter > 5:
#             print("Reached max, quitting")
#             exit()


if __name__ == "__main__":
    main()
    