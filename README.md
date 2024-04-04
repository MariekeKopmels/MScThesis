# MSc Thesis Marieke Kopmels - Work In Progress

This research is directed towards the development of a method to remove skin tone information from video data. The proposed method is named the Grinch method as it results in green-looking people. By means of skin detection, segments marked as skin will be given a green colour. The produced output will be video data where all people have green skin. With this approach, a model trained on the new so-called Grinch video data should not be able to base decisions on skin colour information.

The proposed *Grinch method* will be evaluated two-fold, resulting in the following research questions:
- Does the Grinch method reduce skin tone information present in video data?
- Does the Grinch method affect the accuracy of a violence detection model?

For this evaluation, three models are developed and trained. 
- The Skin Segmentation model. A segmentation model that detects which pixels in an image are skin, and which are background pixels.
- A Violence Detection model. A model that serves as a backbone to test the effect of the Grinch method on violence detection.
- A Skin tone prediction model. A model that is trained on video data, to test the effectiveness of the Grinch model, and its performance regarding skin tone information removal.

