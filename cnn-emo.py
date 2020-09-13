# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 14:34:01 2020

@author: harsh
"""

# CREATING THE NEURAL NETWORK AND EMOTION DETECTOR

# 0. Importing the necessary libraries
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Preprocessing the images/data
train_datagen = ImageDataGenerator(rescale = 1./255)
train_set = train_datagen.flow_from_directory('C:\\Users\\harsh\\Desktop\\aithon2020-level-2\\data\\training',
                                              target_size = (48, 48),
                                              batch_size = 32,
                                              color_mode = 'grayscale',
                                              class_mode = 'categorical')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('C:\\Users\\harsh\\Desktop\\aithon2020-level-2\\data\\testing',
                                            target_size = (48, 48),
                                            batch_size = 32,
                                            color_mode = 'grayscale',
                                            class_mode = 'categorical')

# 2. Creating the model
cnn = tf.keras.models.Sequential()

#layer1
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = [48, 48, 1]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
cnn.add(tf.keras.layers.Dropout(0.25))

#layer2
cnn.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
cnn.add(tf.keras.layers.Dropout(0.25))

#layer3
cnn.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
cnn.add(tf.keras.layers.Dropout(0.25))

#layer4
cnn.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2))
cnn.add(tf.keras.layers.Dropout(0.25))

#flattening, full connection and output layer
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units = 512, activation = 'relu'))
cnn.add(tf.keras.layers.Dense(units = 3, activation = 'softmax'))

# 3. Training the model

#compiling
cnn.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#callback
callbacks = [tf.keras.callbacks.EarlyStopping(patience = 35, monitor = 'val_loss')]

#training
cnn.fit(train_set, epochs = 60, validation_data = test_set, callbacks = callbacks)

# 4. Testing on single images
#test_image = image.load_img('C:\\Users\\harsh\\Desktop\\aithon2020-level-2\\data\\im0.jpg', target_size = (48, 48))
#test_image = image.img_to_array(test_image)
#test_image = np.expand_dims(test_image, axis = 0)
#result = cnn.predict(test_image)
train_set.class_indices

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: 'fear', 1: 'happy', 2: 'sad'}

cap = cv2.VideoCapture(0)
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = cnn.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame,(1600,960), interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
