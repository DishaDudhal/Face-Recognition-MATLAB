# Face-Recognition-MATLAB
The file named "FaceRecognition.m" is the main MATLAB script file that contains the code for Facial Recognition in Live Stream using Machine Learning in MATLAB using the Computer Vision System Toolbox. In order to run the project you must create a folder named - 'OurClass' which will contain the training data set, i.e. images of the user in a folder with the name of the user. Another folder named 'testClass' should be created which would contain all the images which are to be tested. The additional file named as 'Facetobecropped.m' is a program for creating a Dataset with resized images as needed for accurate recognition and also feature extraction from each image. (pre-processing step). If the faces are not cropped it leads to decrease in the accuracy of the code


We have used the traditional Viola Jones algorithm for Face detection in a video frame. And uses the ECOC classifier to classify our images depending on the class labels in the dataset.
