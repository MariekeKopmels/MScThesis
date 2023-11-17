import os
import json
import random
import cv2

directory = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/NTU_CCTV-Fights"
os.chdir(directory)
file = open("groundtruth.json")
data = json.load(file)

directory = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Datasets/NTU_CCTV-Fights/mpeg-001-100"
os.chdir(directory)

random_video = random.choice(list(data["database"].keys()))
random_video = "fight_0003"
print("random_video: ", random_video)
random_segment = random.choice(list(data["database"][random_video]["annotations"]))
print("random_segment:", random_segment["segment"])

print("\n\n", data["database"][random_video])

filename = random_video + ".mpeg" 
print("Filename: ", filename)
vid_capture = cv2.VideoCapture(filename)

random_second = random.uniform(random_segment["segment"][0],random_segment["segment"][1])
print("random_second: ", random_second)
frame_rate = data["database"][random_video]["frame_rate"]
print("frame_rate: ", frame_rate)
random_frame = random_second * frame_rate
print("random_frame: ", random_frame)

vid_capture.set(2, random_frame)
ret, frame = vid_capture.read()

if not vid_capture.isOpened():
    print("Error: Could not open video file.")
else:
    while True:
        # Read a frame from the video
        ret, frame = vid_capture.read()

        # Check if the video has reached the end
        if not ret:
            print("At the end")
            break

        # Display the frame
        cv2.imshow('Video Frame', frame)

        # Press 'q' to exit the video playback
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break



# if ret:
#         cv2.imshow("Frame", frame)
#         cv2.waitKey(0)
# else:
#     print("Error: Failed to read a frame from the video.")
#     vid_capture.release()

# cv2.imshow("Frame", frame)
# cv2.waitKey(-1)