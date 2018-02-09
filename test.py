
import numpy as np
import pandas as pd
import sklearn
import csv
import cv2
import os, glob
import time

from sklearn.model_selection import train_test_split

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt
from subprocess import check_output

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications import VGG16

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import *
from keras.models import *
from keras.layers import Input, Dense
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
import numpy as np
from keras.utils.np_utils import to_categorical
import glob
import os.path
from keras.models import load_model
from keras.models import model_from_yaml


TRAIN_DIR = '/experiments/cameraImgs/'

MODEL_NAME = 'VGG'

TEST_MODE = True

if MODEL_NAME == 'Inception':
    print('Training Inception N/W')
    img_size = 256

elif MODEL_NAME == 'SimpleCNN' :
    print('Training Simple CNN N/W')
    img_size = 256
else:
    print('Training VGG N/W')
    img_size = 224

# modelInc = load_model('models/modelIncep.h5') # load the saved model
modelVGG = load_model('models/modelVGG_Trained2'
                      '.h5') # load the saved model
modelNN = load_model('models/modelSimpleNN1.h5') # load the saved model


TEST_OUTPUTFILE = '/experiments/kaggle/submissions/submission_VGG2.csv'
TEST_OUTPUTFILE2 = '/experiments/kaggle/submissions/analysis_VGG2.csv'

TEST_DIR = '/experiments/kaggle/test/'
sampleFileList = '/experiments/kaggle/sample_submission.csv'

num_classes = 10

cameras = {
    "LG-Nexus-5x": 0,
    "iPhone-4s": 1,
    "Motorola-Droid-Maxx": 2,
    "Motorola-Nexus-6": 3,
    "Samsung-Galaxy-S4": 4,
    "iPhone-6": 5,
    "Sony-NEX-7": 6,
    "HTC-1-M7": 7,
    "Samsung-Galaxy-Note3": 8,
    "Motorola-X": 9
}

getPhoneNames = dict(map(reversed, cameras.items()))

# ---------------------------- TESTING --------------------

def testImg(filename, imageSize, model):
    img = image.load_img(filename, target_size=(imageSize, imageSize))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # img = preprocess_input(img)
    preds = model.predict(img, batch_size=None, verbose=0)

    return preds

import csv



myFile = open(TEST_OUTPUTFILE, 'w')

stop = 0
with myFile:
    writer = csv.writer(myFile)
    myData = [["fname","camera"]]
    writer.writerows(myData)

    with open(sampleFileList) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            filename = (TEST_DIR + row[0])

            if MODEL_NAME == 'Inception':
                preds = testImg(filename, img_size, modelInc)

            elif MODEL_NAME == 'SimpleCNN':
                preds = testImg(filename, img_size, modelNN)

            else:
                preds = testImg(filename, img_size, modelVGG)



            print("PREDICTIONS: ")
            for x in preds:
                print (x)
                myData = [[x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]]]

                maxValue = 0
                id = 0
                for t in range(0, 10):
                    if (x[t] > maxValue):
                        maxValue = x[t]
                        id = t
                print(getPhoneNames[id], maxValue)

                myData =[[row[0],getPhoneNames[id]]]
                writer.writerows(myData)

# ------------------ Store Prob values for all into a separate file for ensembling later on

myFile2 = open(TEST_OUTPUTFILE2, 'w')

with myFile2:
    writer2 = csv.writer(myFile2)

    with open(sampleFileList) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            filename = (TEST_DIR + row[0])

            if MODEL_NAME == 'Inception':
                preds = testImg(filename, img_size, modelInc)

            elif MODEL_NAME == 'SimpleCNN':
                preds = testImg(filename, img_size, modelNN)

            else:
                preds = testImg(filename, img_size, modelVGG)

            print("PREDICTIONS: ")
            for x in preds:
                print (x)
                myData2 = [[row[0],x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]]]

                writer2.writerows(myData2)