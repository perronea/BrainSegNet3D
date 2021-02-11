from types import MethodType

import keras
from keras.layers import Input, Conv3D, Dropout, MaxPooling3D, Activation, UpSampling3D, concatenate, Conv3DTranspose
from keras.models import Model, load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.callbacks import ModelCheckpoint

from load_data import load_data

import os
import numpy as np
import nibabel as nib
import tensorflow as tf
print(tf.__version__)
print(keras.__version__)

#from keras_segmentation.models.model_utils import get_segmentation_model


def segment_model_1(input_height, input_width, input_depth, n_classes):
    

    img_input = Input(shape=(input_height,input_width, input_depth, 1))

    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D((2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D((2, 2, 2))(conv2)


    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling3D((2, 2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling3D((2, 2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv5)

    output = Conv3D( n_classes, (1, 1, 1) , padding='same')(conv5)

    output = (Activation('softmax'))(output)

    model = Model(img_input, output)
    opt = keras.optimizers.Adadelta(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    return(model)

def segment_model_2(input_height, input_width, input_depth, n_classes):
    

    img_input = Input(shape=(input_height,input_width, input_depth, 1))

    conv1 = Conv3D(64, (7, 7, 7), activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv3D(64, (7, 7, 7), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D((2, 2, 2))(conv1)

    conv2 = Conv3D(64, (7, 7, 7), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv3D(64, (7, 7, 7), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D((2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.1)(conv3)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D((2, 2, 2))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.1)(conv4)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)

    up1 = concatenate([UpSampling3D((2, 2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up1)
    conv5 = Dropout(0.1)(conv5)
    conv5 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv5)

    up2 = concatenate([UpSampling3D((2, 2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv3D(64, (7, 7, 7), activation='relu', padding='same')(up2)
    conv6 = Dropout(0.1)(conv6)
    conv6 = Conv3D(64, (7, 7, 7), activation='relu', padding='same')(conv6)

    up3 = concatenate([UpSampling3D((2, 2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv3D(64, (7, 7, 7), activation='relu', padding='same')(up3)
    conv7 = Dropout(0.1)(conv7)
    conv7 = Conv3D(64, (7, 7, 7), activation='relu', padding='same')(conv7)

    output = Conv3D( n_classes, (1, 1, 1) , padding='same')(conv7)

    output = (Activation('softmax'))(output)

    model = Model(img_input, output)
    opt = keras.optimizers.Adadelta(learning_rate=0.05)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    return(model)

    

def segment_model_3(input_height, input_width, input_depth, n_classes):
    

    img_input = Input(shape=(input_height,input_width, input_depth, 1))

    conv1 = Conv3D(64, (7, 7, 7), activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv3D(64, (7, 7, 7), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D((2, 2, 2))(conv1)

    conv2 = Conv3D(128, (7, 7, 7), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv3D(128, (7, 7, 7), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D((2, 2, 2))(conv2)


    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling3D((2, 2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv3D(128, (7, 7, 7), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv3D(128, (7, 7, 7), activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling3D((2, 2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv3D(64, (7, 7, 7), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv3D(64, (7, 7, 7), activation='relu', padding='same')(conv5)

    output = Conv3D( n_classes, (1, 1, 1) , padding='same')(conv5)

    output = (Activation('softmax'))(output)

    model = Model(img_input, output)
    opt = keras.optimizers.Adadelta(learning_rate=0.05)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    return(model)

def segment_model_4(input_height, input_width, input_depth, n_classes):
    

    img_input = Input(shape=(input_height,input_width, input_depth, 1))

    conv1 = Conv3D(64, (7, 7, 7), activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv3D(64, (7, 7, 7), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D((2, 2, 2))(conv1)

    conv2 = Conv3D(64, (7, 7, 7), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv3D(64, (7, 7, 7), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D((2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.1)(conv3)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D((2, 2, 2))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.1)(conv4)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)

    up1 = concatenate([Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(conv4), conv3], axis=-1)

    up2 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(up1), conv2], axis=-1)

    up3 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(up2), conv1], axis=-1)

    output = Conv3D( n_classes, (1, 1, 1) , padding='same')(up3)

    output = (Activation('softmax'))(output)

    model = Model(img_input, output)
    opt = keras.optimizers.Adadelta(learning_rate=0.05)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    return(model)

def segment_model_5(input_height, input_width, input_depth, n_classes):
    

    img_input = Input(shape=(input_height,input_width, input_depth, 1))

    conv1 = Conv3D(64, (7, 7, 7), activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv3D(64, (7, 7, 7), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D((2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D((2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.1)(conv3)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D((2, 2, 2))(conv3)

    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.1)(conv4)
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D((2, 2, 2))(conv4)

    conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Dropout(0.1)(conv5)
    conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv5)

    up1 = concatenate([Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(conv5), conv4], axis=-1)

    up2 = concatenate([Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(up1), conv3], axis=-1)

    up3 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(up2), conv2], axis=-1)

    up4 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(up3), conv1], axis=-1)

    output = Conv3D( n_classes, (1, 1, 1) , padding='same')(up4)

    output = (Activation('softmax'))(output)

    model = Model(img_input, output)
    opt = keras.optimizers.Adadelta(learning_rate=0.05)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    return(model)

def segment_model_14(input_height, input_width, input_depth, n_classes):
    

    img_input = Input(shape=(input_height,input_width, input_depth, 1))

    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.3)(conv1)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D((2, 2, 2))(conv1)

    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.3)(conv2)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D((2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.3)(conv3)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D((2, 2, 2))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.3)(conv4)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D((2, 2, 2))(conv4)

    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)

    up1 = concatenate([Conv3DTranspose(256, (3, 3, 3), strides=2, padding='same')(conv5), conv4], axis=-1)

    up2 = concatenate([Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(up1), conv3], axis=-1)

    up3 = concatenate([Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(up2), conv2], axis=-1)

    up4 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(up3), conv1], axis=-1)

    output = Conv3D( n_classes, (1, 1, 1) , padding='same')(up4)

    output = (Activation('softmax'))(output)

    model = Model(img_input, output)
    opt = keras.optimizers.Adadelta(learning_rate=0.05)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    return(model)


def segment_model_12(input_height, input_width, input_depth, n_classes):
    

    img_input = Input(shape=(input_height,input_width, input_depth, 1))

    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D((2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D((2, 2, 2))(conv2)

    #up01 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(conv2), conv1], axis=-1)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.1)(conv3)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D((2, 2, 2))(conv3)

    up12 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(conv3), conv2], axis=-1)
    up11 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(up12), conv1], axis=-1)

    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.1)(conv4)
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)
    #pool4 = MaxPooling3D((2, 2, 2))(conv4)

    #up23 = concatenate([Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(conv4), conv3], axis=-1)
    #up22 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(up23), up12, conv2], axis=-1)
    #up21 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(up22), up11, up01, conv1], axis=-1)

    #conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool4)
    #conv5 = Dropout(0.1)(conv5)
    #conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv5)

    #up1 = concatenate([Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(conv5), conv4], axis=-1)

    #up2 = concatenate([Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(up1), up23, conv3], axis=-1)
    up2 = concatenate([Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(conv4), conv3], axis=-1)

    up3 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(up2), up12, conv2], axis=-1)

    up4 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(up3) , up11,conv1], axis=-1)

    output = Conv3D( n_classes, (1, 1, 1) , padding='same')(up4)

    output = (Activation('softmax'))(output)

    model = Model(img_input, output)
    opt = keras.optimizers.Adadelta(learning_rate=0.05)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])


    return(model)


def segment_model_13(input_height, input_width, input_depth, n_classes):
    

    img_input = Input(shape=(input_height,input_width, input_depth, 1))

    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D((2, 2, 2))(conv1)

    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D((2, 2, 2))(conv2)

    #up01 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(conv2), conv1], axis=-1)

    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D((2, 2, 2))(conv3)

    up12 = concatenate([Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(conv3), conv2], axis=-1)
    up11 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(up12), conv1], axis=-1)

    conv4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv4)
    #pool4 = MaxPooling3D((2, 2, 2))(conv4)

    #up23 = concatenate([Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(conv4), conv3], axis=-1)
    #up22 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(up23), up12, conv2], axis=-1)
    #up21 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(up22), up11, up01, conv1], axis=-1)

    #conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool4)
    #conv5 = Dropout(0.1)(conv5)
    #conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv5)

    #up1 = concatenate([Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(conv5), conv4], axis=-1)

    #up2 = concatenate([Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(up1), up23, conv3], axis=-1)
    up2 = concatenate([Conv3DTranspose(256, (3, 3, 3), strides=2, padding='same')(conv4), conv3], axis=-1)

    up3 = concatenate([Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(up2), up12, conv2], axis=-1)

    up4 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(up3) , up11,conv1], axis=-1)

    output = Conv3D( n_classes, (1, 1, 1) , padding='same')(up4)

    output = (Activation('softmax'))(output)

    model = Model(img_input, output)
    opt = keras.optimizers.Adadelta(learning_rate=0.05)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])


    return(model)

def segment_model_15(input_height, input_width, input_depth, n_classes):
    

    img_input = Input(shape=(input_height,input_width, input_depth, 1))

    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D((2, 2, 2))(conv1)

    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D((2, 2, 2))(conv2)

    up01 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(conv2), conv1], axis=-1)

    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)

    up12 = concatenate([Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(conv3), conv2], axis=-1)
    up11 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(up12), up01, conv1], axis=-1)


    output = Conv3D( n_classes, (1, 1, 1) , padding='same')(up11)

    output = (Activation('softmax'))(output)

    model = Model(img_input, output)
    opt = keras.optimizers.Adadelta(learning_rate=0.05)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])


    return(model)

def segment_model_16(input_height, input_width, input_depth, n_classes):
    

    img_input = Input(shape=(input_height,input_width, input_depth, 1))

    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D((2, 2, 2))(conv1)

    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D((2, 2, 2))(conv2)

    up01 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(conv2), conv1], axis=-1)

    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)

    up12 = concatenate([Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(conv3), conv2], axis=-1)
    up11 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(up12), up01, conv1], axis=-1)


    output = Conv3D( n_classes, (1, 1, 1) , padding='same')(up11)

    output = (Activation('softmax'))(output)

    model = Model(img_input, output)
    opt = keras.optimizers.Adadelta(learning_rate=0.05)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])


    return(model)

def segment_model_18(input_height, input_width, input_depth, n_classes):
    

    img_input = Input(shape=(input_height,input_width, input_depth, 1))

    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(img_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D((2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D((2, 2, 2))(conv2)

    up01 = concatenate([Conv3DTranspose(32, (3, 3, 3), strides=2, padding='same')(conv2), conv1], axis=-1)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)

    up12 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(conv3), conv2], axis=-1)
    up11 = concatenate([Conv3DTranspose(32, (3, 3, 3), strides=2, padding='same')(up12), up01, conv1], axis=-1)


    output = Conv3D( n_classes, (1, 1, 1) , padding='same')(up11)

    output = (Activation('softmax'))(output)

    model = Model(img_input, output)
    opt = keras.optimizers.Adadelta(learning_rate=0.05)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])


    return(model)

def segment_model_17(input_height, input_width, input_depth, n_classes):
    

    img_input = Input(shape=(input_height,input_width, input_depth, 1))

    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(img_input)
    #conv1 = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(conv1, training=True)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    #conv1 = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(conv1, training=True)
    pool1 = MaxPooling3D((2, 2, 2))(conv1)

    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    #conv2 = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(conv2, training=True)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    #conv2 = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(conv2, training=True)
    pool2 = MaxPooling3D((2, 2, 2))(conv2)

    #up01 = concatenate([InstanceNormalization(axis=3, center=True, epsilon=1e-5)(Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(conv2), training=True), conv1], axis=-1)
    up01 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(conv2), conv1], axis=-1)

    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
    #conv3 = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(conv3, training=True)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
    #conv3 = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(conv3, training=True)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
    #conv3 = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(conv3, training=True)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)

    #up12 = concatenate([InstanceNormalization(axis=3, center=True, epsilon=1e-5)(Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(conv3), training=True), conv2], axis=-1)
    #up11 = concatenate([InstanceNormalization(axis=3, center=True, epsilon=1e-5)(Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(up12), training=True), up01, conv1], axis=-1)
    up12 = concatenate([Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(conv3), conv2], axis=-1)
    up11 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(up12), up01, conv1], axis=-1)

    output = Conv3D( n_classes, (1, 1, 1) , padding='same')(up11)

    output = (Activation('softmax'))(output)

    model = Model(img_input, output)
    opt = keras.optimizers.Adadelta(learning_rate=0.05)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])


    return(model)

def segment_model_X(input_height, input_width, input_depth, n_classes):
    

    img_input = Input(shape=(input_height,input_width, input_depth, 1))

    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(img_input)
    #conv1 = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(conv1, training=True)
    #conv1 = Activation('relu')(conv1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    #conv1 = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(conv1, training=True)
    #conv1 = Activation('relu')(conv1)
    conv1 = Dropout(0.2)(conv1)
    #pool1 = MaxPooling3D((2, 2, 2))(conv1)

    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(MaxPooling3D((2, 2, 2))(conv1))
    #conv2 = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(conv2, training=True)
    #conv2 = Activation('relu')(conv2)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    #conv2 = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(conv2, training=True)
    #conv2 = Activation('relu')(conv2)
    conv2 = Dropout(0.2)(conv2)
    #pool2 = MaxPooling3D((2, 2, 2))(conv2)

    #up21 = Conv3DTranspose(64, (3, 3, 3), activation='relu',strides=2, padding='same')(conv2)
    #up21 = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(up21, training=True)
    #up21 = Activation('relu')(up21)
    #up21 = Dropout(0.2)(up21)
    #skip01 = concatenate([Dropout(0.2)(Conv3DTranspose(64, (3, 3, 3), activation='relu',strides=2, padding='same')(conv2)), conv1], axis=-1)

    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(MaxPooling3D((2, 2, 2))(conv2))
    #conv3 = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(conv3, training=True)
    #conv3 = Activation('relu')(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
    #conv3 = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(conv3, training=True)
    #conv3 = Activation('relu')(conv3)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
    #conv3 = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(conv3, training=True)
    #conv3 = Activation('relu')(conv3)
    conv3 = Dropout(0.2)(conv3)

    #up32 = Conv3DTranspose(128, (3, 3, 3), activation='relu', strides=2, padding='same')(conv3)
    #up32 = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(up32, training=True)
    #up32 = Activation('relu')(up32)
    #up32 = Dropout(0.2)(up32)
    #skip02 = concatenate([Dropout(0.2)(Conv3DTranspose(128, (3, 3, 3), activation='relu', strides=2, padding='same')(conv3)), conv2], axis=-1)

    #up31 = Conv3DTranspose(64, (3, 3, 3), activation='relu', strides=2, padding='same')(skip02)
    #up31 = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(up31, training=True)
    #up31 = Activation('relu')(up31)
    #up31 = Dropout(0.2)(up31)  

    #skip11 = concatenate([Dropout(0.2)(Conv3DTranspose(64, (3, 3, 3), activation='relu', strides=2, padding='same')(concatenate([Dropout(0.2)(Conv3DTranspose(128, (3, 3, 3), activation='relu', strides=2, padding='same')(conv3)), conv2], axis=-1))), concatenate([Dropout(0.2)(Conv3DTranspose(64, (3, 3, 3), activation='relu',strides=2, padding='same')(conv2)), conv1], axis=-1), conv1], axis=-1)

    output = Conv3D( n_classes, (1, 1, 1) , padding='same')(concatenate([Dropout(0.2)(Conv3DTranspose(64, (3, 3, 3), activation='relu', strides=2, padding='same')(concatenate([Dropout(0.2)(Conv3DTranspose(128, (3, 3, 3), activation='relu', strides=2, padding='same')(conv3)), conv2], axis=-1))), concatenate([Dropout(0.2)(Conv3DTranspose(64, (3, 3, 3), activation='relu',strides=2, padding='same')(conv2)), conv1], axis=-1), conv1], axis=-1))

    output = (Activation('softmax'))(output)

    model = Model(img_input, output)
    opt = keras.optimizers.Adadelta(learning_rate=0.05)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])


    return(model)

def segment_model_6(input_height, input_width, input_depth, n_classes):
    

    img_input = Input(shape=(input_height,input_width, input_depth, 1))

    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(img_input)
    #conv1 = Dropout(0.1)(conv1)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D((2, 2, 2))(conv1)

    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    #conv2 = Dropout(0.1)(conv2)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D((2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    #conv3 = Dropout(0.1)(conv3)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D((2, 2, 2))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    #conv4 = Dropout(0.1)(conv4)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D((2, 2, 2))(conv4)

    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Dropout(0.1)(conv5)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)

    up1 = concatenate([Conv3DTranspose(256, (3, 3, 3), strides=2, padding='same')(conv5), conv4], axis=-1)

    up2 = concatenate([Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(up1), conv3], axis=-1)

    up3 = concatenate([Conv3DTranspose(128, (3, 3, 3), strides=2, padding='same')(up2), conv2], axis=-1)

    up4 = concatenate([Conv3DTranspose(64, (3, 3, 3), strides=2, padding='same')(up3), conv1], axis=-1)

    output = Conv3D( n_classes, (1, 1, 1) , padding='same')(up4)

    output = (Activation('softmax'))(output)

    model = Model(img_input, output)
    opt = keras.optimizers.Adadelta(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    return(model)

def encode_block(input_layer, num_filters, kernel_size=(3, 3, 3), dropout_rate=0.1, downsample=True):
    x = Conv3D(num_filters, kernel_size, padding='same')(input_layer)
    x = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    #x = Conv3D(num_filters, kernel_size, padding='same')(x)
    #x = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    #x = Activation('relu')(x)
    if downsample == True:
        x = MaxPooling3D((2, 2, 2), padding='same')(x)
    return x

def decode_block(input_layer, num_filters, kernel_size=(3, 3, 3), dropout_rate=0.1, pad='same', stride=2):
    x = Conv3DTranspose(num_filters, kernel_size, strides=stride, padding=pad)(input_layer)
    x = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x
    

def segment_model_7(input_height, input_width, input_depth, n_classes):
    
    img_input = Input(shape=(input_height,input_width, input_depth, 1))

    conv0 = encode_block(img_input, 32)
    up_conv01 = decode_block(conv0, 32)

    conv1 = encode_block(conv0, 32)
    up_conv11 = decode_block(conv1, 32)
    up_conv02 = decode_block(concatenate([up_conv11, conv0], axis=-1), 32)

    conv2 = encode_block(conv1, 32)
    up_conv21 = decode_block(conv2, 32)
    up_conv12 = decode_block(concatenate([up_conv21, conv1], axis=-1), 32)
    up_conv03 = decode_block(concatenate([up_conv12, up_conv11, conv0], axis=-1), 32)

    conv3 = encode_block(conv2, 32)
    up_conv31 = decode_block(conv3, 32)
    up_conv22 = decode_block(concatenate([up_conv31, conv2], axis=-1), 32)
    up_conv13 = decode_block(concatenate([up_conv22, up_conv21, conv1], axis=-1), 32)
    up_conv04 = decode_block(concatenate([up_conv13, up_conv12, up_conv11, conv0], axis=-1), 32)


    #conv4 = encode_block(conv3, 512, downsample=False)
    #conv4 = Conv3D(512, (3,3,3), padding='same')(conv3)
    #conv4 = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(conv4)
    #conv4 = Activation('relu')(conv4)
    #conv4 = Dropout(0.1)(conv4)

    output = Conv3D(n_classes, (1, 1, 1))(concatenate([up_conv01, up_conv02, up_conv03, up_conv04], axis=-1))
    output = Activation('softmax')(output)

    model = Model(img_input, output)
    opt = keras.optimizers.Adadelta(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    return(model)

def segment_model_8(input_height, input_width, input_depth, n_classes):
    
    img_input = Input(shape=(input_height,input_width, input_depth, 1))

    conv0 = encode_block(img_input, 16)
    up_conv01 = decode_block(conv0, 16)

    conv1 = encode_block(conv0, 32)
    up_conv11 = decode_block(conv1, 32)
    up_conv02 = decode_block(concatenate([up_conv11, conv0], axis=-1), 16)

    conv2 = encode_block(conv1, 64)
    up_conv21 = decode_block(conv2, 64)
    up_conv12 = decode_block(concatenate([up_conv21, conv1], axis=-1), 32)
    up_conv03 = decode_block(concatenate([up_conv12, up_conv11, conv0], axis=-1), 16)

    conv3 = encode_block(conv2, 128)
    up_conv31 = decode_block(conv3, 128)
    up_conv22 = decode_block(concatenate([up_conv31, conv2], axis=-1), 64)
    up_conv13 = decode_block(concatenate([up_conv22, up_conv21, conv1], axis=-1), 32)
    up_conv04 = decode_block(concatenate([up_conv13, up_conv12, up_conv11, conv0], axis=-1), 16)

    output = Conv3D(n_classes, (1, 1, 1))(concatenate([up_conv01, up_conv02, up_conv03, up_conv04], axis=-1))
    output = Activation('softmax')(output)

    model = Model(img_input, output)
    opt = keras.optimizers.Adadelta(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    return(model)

def segment_model_9(input_height, input_width, input_depth, n_classes):
    
    img_input = Input(shape=(input_height,input_width, input_depth, 1))
    conv00 = encode_block(img_input, 64, downsample=False)

    conv10 = encode_block(conv00, 64)
    #up_conv01 = decode_block(conv10, 64)
    #up_conv02 = decode_block(concatenate([up_conv11, conv0], axis=-1), 64)

    conv20 = encode_block(conv10, 64)
    up_conv11 = decode_block(conv20, 64)
    up_conv02 = decode_block(concatenate([up_conv11, conv10], axis=-1), 64)
    #up_conv03 = decode_block(concatenate([up_conv12, up_conv11, conv0], axis=-1), 64)

    conv30 = encode_block(conv20, 64)
    up_conv21 = decode_block(conv30, 64)
    up_conv12 = decode_block(concatenate([up_conv21, conv20], axis=-1), 64)
    up_conv03 = decode_block(concatenate([up_conv12, up_conv11, conv10], axis=-1), 64)
    #up_conv04 = decode_block(concatenate([up_conv13, up_conv12, up_conv11, conv0], axis=-1), 64)

    output = Conv3D(n_classes, (1, 1, 1))(concatenate([conv00, up_conv02, up_conv03], axis=-1))
    output = Activation('softmax')(output)

    model = Model(img_input, output)
    opt = keras.optimizers.Adadelta(learning_rate=0.02)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    return(model)

def segment_model_10(input_height, input_width, input_depth, n_classes):
    
    img_input = Input(shape=(input_height,input_width, input_depth, 1))
    conv00 = encode_block(img_input, 64, downsample=False)

    conv10 = encode_block(conv00, 64)
    #up_conv01 = decode_block(conv10, 64)
    #up_conv02 = decode_block(concatenate([up_conv11, conv0], axis=-1), 64)

    conv20 = encode_block(conv10, 64)
    #up_conv11 = decode_block(conv20, 64)
    #up_conv02 = decode_block(concatenate([up_conv11, conv10], axis=-1), 64)
    #up_conv03 = decode_block(concatenate([up_conv12, up_conv11, conv0], axis=-1), 64)

    #conv30 = encode_block(conv20, 64)
    #up_conv21 = decode_block(conv30, 64)
    #up_conv12 = decode_block(concatenate([up_conv21, conv20], axis=-1), 64)
    up_conv12 = decode_block(conv20, 64)
    up_conv03 = decode_block(concatenate([up_conv12, conv10], axis=-1), 64)
    #up_conv04 = decode_block(concatenate([up_conv13, up_conv12, up_conv11, conv0], axis=-1), 64)

    output = Conv3D(n_classes, (1, 1, 1))(concatenate([conv00, up_conv03], axis=-1))
    output = Activation('softmax')(output)

    model = Model(img_input, output)
    opt = keras.optimizers.Adadelta(learning_rate=0.02)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    model.summary()

    return(model)




def get_model(img_input, output):

    print(Model(img_input, out).output_shape)
    print(Model(img_input, out).input_shape)

    n_classes, output_height, output_width, output_depth = Model(img_input, out).output_shape
    input_height, input_width, input_depth = Model(img_input, out).input_shape

    output = (Activation('softmax'))(output)

    model = Model(img_input, output)
    model.output_width = output_width
    model.output_height = output_height
    model.output_depth = output_depth
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width
    model.input_depth = input_depth
    model.model_name = ""

    return model


def reconstruct_seg(preds, classes):
    num_preds, h, w, d, num_classes = preds.shape
    assert num_classes == len(classes)

    for i in range(0, num_preds):
        pred = preds[i,:,:,:,:]
        # save the highest probability from each class (a measure of confidence)
        nib.save(nib.Nifti1Image(pred.max(axis=-1), np.eye(4)), 'highest_class_probabilities.nii.gz')
        # return the enumerated class with the highest probability
        pred = pred.argmax(axis=-1)
        print(pred.shape)
        #print(np.unique(pred_mapped, return_counts=True))
        pred_mapped = np.zeros((h,w,d))
        assert pred.shape == pred_mapped.shape

        for c, class_x in enumerate(classes):
            pred_mapped = np.where(pred == c, class_x, pred)
            #class_x_idx = np.where(pred[:,:,:,c] == 1)
            #pred_seg[class_x_idx] = class_x
        print(np.unique(pred_mapped, return_counts=True))
        nib.save(nib.Nifti1Image(pred_mapped, np.eye(4)), 'first_prediction.nii.gz')
        np.save("first_prediction", pred_mapped)
        
    


def main():
    #model = get_segmentation_model(img_input ,  out ) # this would build the segmentation model
    h = 160
    w = 208
    d = 160

    data_dir = "/home/exacloud/lustre1/fnl_lab/projects/BrainSegNet3D/train_data_norm"
    validation_dir = "/home/exacloud/lustre1/fnl_lab/projects/BrainSegNet3D/test_data_norm"
    classes = []
    with open("/home/exacloud/lustre1/fnl_lab/projects/BrainSegNet3D/data/region_id.csv", "r") as c:
        for line in c:
            line = line.rstrip()
            class_id = int(line.split(',')[0])
            classes.append(class_id)
    num_classes = len(classes)
    
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = segment_model_18(h,w,d, num_classes)
        model.summary(line_length=150)

    #model = segment_model_3(h,w,d, num_classes)
    use_data_generator = True

    train = True

    if train:
        train_generator = load_data(data_dir, (h,w,d), classes, batch_size=4, subfolder="T1w_brain", generator=use_data_generator)
        #validation_generator = load_data(validation_dir, (h,w,d), classes, batch_size=4, subfolder="T1w_brain", generator=use_data_generator)

        output_dir = "/home/exacloud/lustre1/fnl_lab/projects/BrainSegNet3D/code/trials/test18"
        filepath = os.path.join(output_dir, "model-{epoch:02d}.hdf5")
        checkpoint = ModelCheckpoint(filepath, save_best_only=False, save_weights_only=False, period=10)
        callbacks_list = [checkpoint]

        model.fit_generator(generator=train_generator, epochs=300, use_multiprocessing=False, workers=1, verbose=2, callbacks=callbacks_list) 
        #model.fit_generator(generator=train_generator, validation_data=validation_generator, epochs=300, use_multiprocessing=False, workers=1, verbose=2, callbacks=callbacks_list) 
        #model.fit(train_data.batch(4), epochs=50, use_multiprocessing=True, workers=4, batch_size=4, verbose=2, callbacks=callbacks_list) 
    else:
        data_dir = "/home/exacloud/lustre1/fnl_lab/projects/BrainSegNet3D/test_data"
        train_data = load_data(data_dir, (h,w,d), classes, subfolder="T1w_brain")
        #train(model, train_images, train_annotations, input_height=h, input_width=w, input_depth=d, n_classes=classes)
    
        #model.fit(train_data["brain_images"], train_data["segmentations"], epochs=100)
        model = load_model('init_aseg_model')

        new = model.predict(train_data["brain_images"])
        print(np.unique(train_data["segmentations"], return_counts=True))
        np.save("raw_prediction.npy", new)
        reconstruct_seg(new, classes)
    
    #model.save('init_aseg_model')


if __name__ == "__main__":
    main() 



