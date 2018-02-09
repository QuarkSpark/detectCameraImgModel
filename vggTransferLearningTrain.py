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

# NN train:val size : 9000, 1000 [N = 1000]
# Inception: N = 600
# VGG: N =

TRAIN_DIR = '/experiments/cameraImgs/'

MODEL_NAME = 'VGG'

TEST_MODE = False

num_classes = 10

weights_path = '/experiments/kaggle/vgg16_weights.h5'
top_model_weights_path = '/experiments/kaggle/fc_model.h5'

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

# ------------- Read input data --------------------

def read_data(inp_dir, N, image_size):
    img_data_list = []
    labels = []

    directories = [d for d in os.listdir(inp_dir)
                   if os.path.isdir(os.path.join(inp_dir, d))]
    print(directories)

    for d in directories:
        print(d, cameras[d])
        label_dir = os.path.join(inp_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".jpg")]


        for f in range(0, 1000) :

            labels.append(cameras[d])

            # img = cv2.imread(file_names[f])
            # img = img[10 : image_size + 10, 10 : image_size + 10]

            img = image.load_img(file_names[f], target_size=(image_size, image_size))
            img = image.img_to_array(img)
            # img = preprocess_input(img)

            img_data_list.append(img)
            # print(np.shape(img))
            # print(f)
        print(d)
            # print(np.shape(img))


            # if stop > N:
            #     break
            # stop += 1

    return img_data_list, labels

# ------------ Train VGG , rmsprop optimizer -----------------------

def trainVGG(image_size, num_classes):

    image_input = Input(shape=(image_size, image_size, 3))
    vgg = applications.VGG16(input_tensor=image_input, weights='imagenet', include_top=False)
    for layer in vgg.layers[:-4]:
        layer.trainable = False
    for layer in vgg.layers:
        print(layer, layer.trainable)

    print('Model loaded.')

    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(vgg)

    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Show a summary of the model. Check the number of trainable parameters
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['categorical_accuracy'])
    model.summary()
    return model

    # Load pre-trained weights from VGG

    # assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    # f = h5py.File(weights_path)
    # for k in range(f.attrs['nb_layers']):
    #     if k >= len(model.layers):
    #         break
    #     g = f['layer_{}'.format(k)]
    #     weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    #     model.layers[k].set_weights(weights)
    # f.close()
    # print('Model loaded.')

    # ------------------------- include_top=False --------------


    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.


    # ------------------------- include_top=True (Doesn't work that well) --------------

    # image_input = Input(shape=(image_size, image_size, 3))
    #
    # model = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')
    #
    # model.summary()
    #
    # last_layer = model.get_layer('fc2').output
    # out = Dense(num_classes, activation='softmax', name='output')(last_layer)
    #
    # custom_vgg_model = Model(image_input, out)
    # custom_vgg_model.summary()
    #
    # for layer in custom_vgg_model.layers[:-1]:
    #     layer.trainable = False
    #
    # custom_vgg_model.layers[3].trainable
    #
    # custom_vgg_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    # return custom_vgg_model

# ------------ Train simple neural network , rmsprop optimizer -----------------------

def trainSimpleNN(image_size, num_classes):
    input_shape = (image_size, image_size, 3)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(7, 7), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    return model


# ------------ Train Inception, adam optimizer -----------------------

def trainInception(image_size, num_classes):
    input = Input(shape=(image_size, image_size, 3))

    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=input,
                                   input_shape=(image_size, image_size, 3),
                                   pooling='avg', classes=1000)
    for l in base_model.layers:
        l.trainable = False

    t = base_model(input)
    o = Dense(num_classes, activation='softmax')(t)
    model = Model(inputs=input, outputs=o)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    print(model.summary())
    return model


# ------------------- Read Input Data -------------------

print(TRAIN_DIR)

if(TEST_MODE):
    BATCH = 30
    EPOCH = 3
    N = 50
else:
    BATCH = 60
    EPOCH = 200
    N = 50

if MODEL_NAME == 'Inception':
    print('Training Inception N/W')
    img_size = 256

elif MODEL_NAME == 'SimpleCNN' :
    print('Training Simple CNN N/W')
    img_size = 256
else:
    print('Training VGG N/W')
    img_size =  224


[img_data, labels] = read_data(TRAIN_DIR, N, img_size)

print('Image Features Size: ' , np.shape(img_data))
print('Labels Size: ' , np.shape(labels))

# print(np.shape(labels))
# print(np.shape(img_data))

# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(img_data, Y, test_size=0.1, random_state=2)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

type(X_train)

print(Y)
print(np.shape(X_train))

# # ------------------ 1. INCEPTION ---------------------

if MODEL_NAME == 'Inception':

    print('Training Inception N/W')

    modelInc = trainInception(img_size, num_classes)
    history = modelInc.fit(X_train, y_train, batch_size=BATCH, epochs=EPOCH, shuffle=True, verbose=1 )

    score = modelInc.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    modelInc.save("models/modelIncep.h5")
    print("Saved model to disk")

elif MODEL_NAME == 'SimpleCNN' :
# --------------- 2. Simple NN ------------------------------

    print('Training Simple NN')

    modelNN = trainSimpleNN(img_size, num_classes)

    history = modelNN.fit(X_train, y_train, batch_size=BATCH, epochs=EPOCH,verbose=1, validation_data=(X_test, y_test))

    score = modelNN.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    modelNN.save("models/modelSimpleNN1.h5")
    print("Saved model to disk")


else:
# ---------------- 3. VGG -------------------------

    print('Training VGG')

    modelVGG = trainVGG(img_size, num_classes)

    history = modelVGG.fit(X_train, y_train, batch_size=BATCH, epochs=EPOCH,verbose=1, validation_data=(X_test, y_test))

    score = modelVGG.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    modelVGG.save("models/modelVGG_Trained2.h5")
    print("Saved model to disk")
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(len(acc))
#
# plt.plot(epochs, acc, 'b', label='Training acc')
# plt.plot(epochs, val_acc, 'r', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
#
# plt.figure()
#
# plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
#
# plt.show()
