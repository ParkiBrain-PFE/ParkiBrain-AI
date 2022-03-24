# ParkiBrain - Artificial intelligence

This Repository is the result of our work to make an automatic moroccan number plate recognition software.

### Table of content
- Dataset
- Our approach
  - Data acquisition and preparation
  - Library and Model Architecture
  - Modeling
  - Post-Processing 
- How to run it
  - Colab notebooks
- Conclusion
  
## Dataset
The dataset is 1595 jpg pictures of the front or back of vehicles showing the license plate. They are of different sizes and are mostly cars. The plate license follows Moroccan standard.

For each plate corresponds a string (series of numbers and latin characters) labeled manually. The plate strings could contain a series of numbers and latin letters of different length. Because letters in Morocco license plate standard are Arabic letters, we will consider the following transliteration: a <=> أ, b <=> ب, j <=> ج, d <=> د , h <=> ه , waw <=> و, w <=> w (newly licensed cars), For example:

- The string '123 أ 20' would translate to '123a20'
- The string “123و4567” to “12345waw67”
- The string “1234567ww” to “1234567ww” (remain the same)

We offer the plate strings of 2124 jpg images for number plate optical character recognition (OCR).

- Detection dataset: https://www.kaggle.com/datasets/elmehditaf96/moroccan-vehicle-registration-plates
- number plate OCR: https://msda.um6p.ma/msda_datasets

## Our Approach
Our approach was to use Object Detection to detect plate characters from images. We have chosen to build two models separately instead of using libraries directly like easyOCR or Tesseract due to its weaknesses in handling the variance in the shapes of Moroccan License plates. The first model was trained to detect the licence plate to be then cropped from the original image, which will be then passed into the second model that was trained to detect the characters.

###  Data acquisition and preparation
After downloading the two datasets. we uploaded them to [roboflow.com](https://roboflow.com) as it offers data augmentation technics in the easiest way, as well as annotations conversion to any standard.

### Library and Model Architecture
We have chosen ```myssd_mobilenetv2_320x320``` model with ```Tensorflow``` framework. myssd_mobilenet is a Single-Shot multibox Detection (SSD) network intended to perform object detection.SSD Mobilenet V2 is a one-stage object detection model which has gained popularity for its lean network and novel depthwise separable convolutions. It is a model commonly deployed on low compute devices such as mobile (hence the name Mobilenet) with high accuracy and performance.

The both models were pretrained on the COCO dataset, because we didn’t have enough data, therefor it would only make sense to take the advantage of transfer learning of models that were trained on such a rich dataset.

### Modeling
We used the notebook ```docs/Training_Models.ipynb``` to generate the models. Then we converted the models to ```tflite``` format. 

### Post processing
Now we have two models capable of detecting and reading the number plate, but we are not done yet. 

- For detection : the model returns an array of detected license plates in a frame. we filtered that array to have just the license plate with highest confidence. then crop that frame leaving just the license plate box.
- For recognition: the model returns unordered detected characters. we wrote a function that can handle sorting those characters based on x_min value.

 ## How to run it
 To run the script just create necessary models using the notebook ```docs/Training_Models.ipynb```, and place those models in a directory called ```models```. Then install the dependencies using the following command
```
pip install -r requirements.txt
```
 After that you can run it using ```python main.py```.

 ## Conclusion
This project is a Graduation Project from High School of technology of Meknes. Made by: 

- [Ismail Ajizou](https://github.com/ismailajizou)
- [El Mehdi Salah Ben Souda](https://github.com/Mehdi-Ben-Souda)
- [Mohamed Ait Messkine](https://github.com/AitMesskine)