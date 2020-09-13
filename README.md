# emotion-detector
An AI-powered emotion detector system.

This project was made as part of the challenge for Spotle AI-thon 2020 by Spotle.ai.
The dataset provided was a .csv file containing 48x48 = 2304 columns containing the individual pixel values. and a total of 10816 images.
The pillow/PIL library was used to convert the pixels to their equivalent images.
The dataset contained a certain number of non-facial images.
The converted and cleaned dataset will be provided.
The dataset was based on only 3 emotions/classes viz. FEAR, HAPPY and SAD.
For the purpose of face detection, the haar cascade xml file is used.
The cleaned dataset (10816 -> 10799) was divided into 8639 images for training and 2160 for testing/validation.
A neural network consisting of 4 convoluted layers and a couple of dense layers with the 'adam' optimizer was trained over 60 epochs to give an accuracy of about 73%.
