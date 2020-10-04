# Image Classification Blurr and Non Blurr Images

PS- In the given dataset classify the blurr and non blurr images. 

## Installation

Use the conda environment or Google Colab to see the file
.ipynb and .py file attached::
```bash
Image_classification_blurr_nonblurr.ipynb
Image_classification_blurr_nonblurr.py
```

## Libraries Used

```python
import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
```
## Preprocessed the data into two classes blurr and nonblurr
    Splitted the data into test set and train set
    Processed the images into (224,224) pixel size
    
## Data Augmentation 
    Augmentated the test data
    
## Model Used - MobileNetV2
    Its a light weight model
    used various models but accuracy was good in this model
##  After training the model : : some results
    epochs - 10
    training accuracy- 83.23
    validation accuracy- 78.91
    Auc - 0.82
## Plotted the test history and Auc
    train loss v/s val loss
    train accuracy  v/s val accuracy
## Saved the model
    Kd_model.h5 (file included)
 
## Used the saved model to predict on some random images
    19/20 images gave the correct result
    leading to a good real time accuracy
    
    
#####  Thanks for viewing for more details Check out my ipynb Notebook..



