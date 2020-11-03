from types import MethodType

import keras
from keras.layers import Input, Conv3D, Dropout, MaxPooling3D, Activation, UpSampling3D, concatenate
from keras.models import Model, load_model

from load_data import load_data

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
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
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
    h = 156
    w = 188
    d = 156

    data_dir = "/home/exacloud/lustre1/fnl_lab/projects/BrainSegNet3D/np_data"
    with open("/home/exacloud/lustre1/fnl_lab/projects/BrainSegNet3D/data/classes.txt", "r") as c:
        classes = [int(line.rstrip()) for line in c]
    num_classes = len(classes)
    
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model = segment_model_1(h,w,d, num_classes)
    #model = get_model(img_input, output)
    use_data_generator = True

    train = False

    if train:
        train_data = load_data(data_dir, (h,w,d), classes, batch_size=4, subfolder="T1w_brain", generator=True)
        model.fit(train_data, epochs=50, use_multiprocessing=False, workers=4, batch_size=4, verbose=2) 
    else:
        data_dir = "/home/exacloud/lustre1/fnl_lab/projects/BrainSegNet3D/np_data_test"
        train_data = load_data(data_dir, (h,w,d), classes, subfolder="T1w_brain")
        #train(model, train_images, train_annotations, input_height=h, input_width=w, input_depth=d, n_classes=classes)
    
        #model.fit(train_data["brain_images"], train_data["segmentations"], epochs=100)
        model = load_model('4gpu_model')

        new = model.predict(train_data["brain_images"])
        print(np.unique(train_data["segmentations"], return_counts=True))
        np.save("raw_prediction.npy", new)
        reconstruct_seg(new, classes)
    
    model.save('4gpu_model')


if __name__ == "__main__":
    main() 



