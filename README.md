# MSc Thesis Marieke Kopmels

### Abstract
This thesis introduces the Grinch method, a novel technique designed to reduce skin tone information present in video data, aimed at enforcing deep learning models such as violence detection algorithms to rely on more fair features. By training and comparing violence detection models as well as skin tone prediction models on both regular and modified Grinch data, the effectiveness of the proposed Grinch method is evaluated. While the experiments and corresponding results indicate that the Grinch method does not harm violence detection, its ability to reduce skin tone information present in video data has not been observed. While these results have not shown the desired results, they pave the way for future research, as the results demonstrate the robustness of violence detection models. Research towards both the development of a better-performing skin segmentation model as well as other forms of visual information redaction are therefore encouraged in future research towards ethical AI.

---

This research is directed towards the development of a method to remove skin tone information from video data. The proposed method is named the Grinch method as it results in green-looking people. By means of skin detection, segments marked as skin will be given a green colour. The produced output will be video data where people have green skin. With this approach, a model trained on the new so-called Grinch video data should not be able to base decisions on skin colour information.

The proposed *Grinch method* will be evaluated two-fold, resulting in the following research questions:
- Does the Grinch method reduce skin tone information present in video data?
- Does the Grinch method affect the accuracy of a violence detection model?

For this evaluation, three models are developed and trained. 
- The Skin Segmentation model. A segmentation model that detects which pixels in an image are skin, and which are background pixels.
- A Violence Detection model. A model that serves as a backbone to test the effect of the Grinch method on violence detection.
- A Skin tone prediction model. A model that is trained on video data, to test the effectiveness of the Grinch model, and its performance regarding skin tone information removal.

