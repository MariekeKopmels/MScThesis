import cv2
import numpy as np
import os

if __name__ == '__main__':
    directory = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Output/Grinch"
    os.chdir(directory)
    example_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Output/Grinch/Testdata/image.jpeg"
    output_path = "/Users/mariekekopmels/Desktop/Uni/MScThesis/Code/Output/Grinch/Testdata/model_output.jpeg"
    
    example = cv2.imread(example_path)
    output = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
    _,output = cv2.threshold(output,127,1,0)
    
    grinch = np.copy(example)
    mask = output == 1
    grinch[mask] = [0,255,0]
    
    cv2.imwrite("grinch.png", grinch)
