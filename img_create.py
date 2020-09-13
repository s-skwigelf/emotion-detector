

# CREATING IMAGES FROM PIXEL VALUES IN THE DATASET

# 0. Importing the libs
import numpy as np
import pandas as pd
from PIL import Image

# 1. Performing exploratory data analysis on the dataset
raw = pd.read_csv('dataset_file.csv') #this is just for preliminary data analysis

#checking the available emotions
col_vals = data[["emotion"]].values
unique_vals = np.unique(col_vals) 

#checking the kind of data stored
data.info()
len(data) 

'''
ANALYSIS REPORT - 

- col1 is the emotions column, rest are pixel values
- emotions contains only FEAR, HAPPY and SAD values
- there are 48x48 = 2304 columns containing the pixel values
- the pixels are all of type: int64
- dataset contains 10817 rows, meaning 10816 images in total

Interesting note :- Even though the pixels in the dataframe were of type int64,
the PIL module actually takes type uint8 for values 0-255.

DATASET ANALYSIS -

- there appear to be a certain number of non-facial images in the dataset
- 6 such in the Fear section
- 8 such in the Happy section
- 3 such in the Sad section
- 17 noisy images in total

'''
raw = pd.read_csv('dataset_file.csv', skiprows = 1) #run this for the file coonversion
data = raw.copy()

#shifting the emotions column to the end of the dataframe
cols = list(data.columns.values) 
cols.pop(cols.index('Fear'))
data = data[cols + ['Fear']]
data.head()

fear = 0
happy = 0
sad = 0

# 2. Creating the images as per their emotion

for row in data.itertuples(index = False, name = 'Pandas'):
    

    pixels = row[:-1] 
    pixels = np.array(pixels, dtype='uint8')
    pixels = pixels.reshape((48, 48))
    image = Image.fromarray(pixels)

    if str(row[-1]) == 'Fear':
        image.save('your_folder_path/data/fear/im'+str(fear)+'.jpg')
        fear += 1

    elif str(row[-1]) == 'Happy':
        image.save('your_folder_path/data/happy/im'+str(happy)+'.jpg')
        happy += 1
    
    elif str(row[-1]) == 'Sad':
        image.save('your_folder_path/data/sad/im'+str(sad)+'.jpg')
        sad += 1

print('Done') 
