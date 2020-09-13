import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator 


def train_a_model(trainfile):
    
    raw = pd.read_csv(trainfile, skiprows = 1)
    data = raw.copy()
    
    cols = list(data.columns.values) 
    first_val = data[cols[0]].iloc[0]
    cols.pop(cols.index(first_val))
    data = data[cols + [first_val]]
   
    fear = 0
    happy = 0
    sad = 0
    
    for row in data.itertuples(index = False, name = 'Pandas'):

        pixels = row[:-1] # without label
        pixels = np.array(pixels, dtype='uint8')
        pixels = pixels.reshape((48, 48))
        image = Image.fromarray(pixels)
    
        if str(row[-1]) == 'Fear':
            image.save('/data/training/fear/im'+str(fear)+'.jpg')
            fear += 1
    
        elif str(row[-1]) == 'Happy':
            image.save('/data/training/happy/im'+str(happy)+'.jpg')
            happy += 1
        
        elif str(row[-1]) == 'Sad':
            image.save('/data/training/sad/im'+str(sad)+'.jpg')
            sad += 1

    print('done') 
    
    train_datagen = ImageDataGenerator(rescale = 1./255)
    train_set = train_datagen.flow_from_directory('data/training',
                                                  target_size = (48, 48),
                                                  batch_size = 32,
                                                  color_mode = 'grayscale',
                                                  class_mode = 'categorical')
    
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
    cnn.add(tf.keras.layers.Dropout(0.5))
   
   	#flattening, full connection and output layer
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units = 512, activation = 'relu'))
    cnn.add(tf.keras.layers.Dense(units = 3, activation = 'softmax'))
   
   	# 3. Training the model
   
   	#compiling
    cnn.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
   
   	#callback
    callbacks = [tf.keras.callbacks.EarlyStopping(patience = 35, monitor = 'val_loss')]
   
   	#training
    cnn.fit(train_set, epochs = 60, callbacks = callbacks) 
    
    return cnn



def test_the_model(testfile, cnn):
    res=[]
    
    testing=pd.DataFrame(testfile)
    if 'emotion' in testing.columns:
        testing=testing.drop('emotion',axis=1)
    testing=np.array(testing)
    testing = testing.reshape(testing.shape[0], 48, 48, 1)
    prediction=cnn.predict(testing)
    for i in prediction:
        max_index = np.argmax(i)
        emotion_detection = ('fear', 'sad', 'happy')
        emotion_prediction = emotion_detection[max_index]
        res.append(emotion_prediction)

    return res
