from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import sys
import os
import numpy as np
import nibabel as nib
import tensorflow as tf

import matplotlib.pyplot as plt

def reconstruct_aseg_from_prediction(preds, classes, output_dir):
    num_preds, h, w, d, num_classes = preds.shape
    assert num_classes == len(classes)

    for i in range(0, num_preds):
        pred = preds[i,:,:,:,:]
        # save the highest probability from each class (a measure of confidence)
        nib.save(nib.Nifti1Image(pred.max(axis=-1), np.eye(4)), os.path.join(output_dir, 'epoch-70_highest_class_probabilities.nii.gz'))
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
        nib.save(nib.Nifti1Image(pred_mapped, np.eye(4)), os.path.join(output_dir, 'epoch-70_aseg.nii.gz'))
        np.save(os.path.join(output_dir, "epoch-70_aseg"), pred_mapped)
    return

def decode_aseg(encoded_aseg_arr, classes):
    h,w,d,num_classes = encoded_aseg_arr.shape
    orig_aseg = np.zeros((h,w,d))
    for c, class_x in enumerate(classes):
        class_x_idx = np.where(encoded_aseg_arr[:,:,:,c] ==1)
        orig_aseg[class_x_idx] = class_x
    return(orig_aseg)

def seg_volume_plot(aseg, output_name):
    seg_counts = np.unique(aseg, return_counts=True)
    region = seg_counts[0].astype(str)
    volume = seg_counts[1]

    # Exclude blank space in bar plot
    plt.bar(region[1:], volume[1:])
    plt.savefig(output_name)

    return


def main():

    brain_img = sys.argv[1]
    aseg = sys.argv[2]
    output_dir = sys.argv[3]

    input_shape = (160, 208, 160)
    classes = []
    with open("/home/exacloud/lustre1/fnl_lab/projects/BrainSegNet3D/data/region_id.csv", "r") as c:
        for line in c:
            line = line.rstrip()
            class_id = int(line.split(',')[0])
            classes.append(class_id)
    num_classes = len(classes)

    model = load_model('/home/exacloud/lustre1/fnl_lab/projects/BrainSegNet3D/code/trials/test16/model-70.hdf5', custom_objects={'InstanceNormalization':InstanceNormalization})

    orig_brain = np.load(brain_img)
    orig_aseg = np.load(aseg)

    pred_encoded_aseg = model.predict(orig_brain[np.newaxis, :])

    reconstruct_aseg_from_prediction(pred_encoded_aseg, classes, output_dir)

    #pred_aseg = decode_aseg(pre_encoded_aseg, classes)
    
    
    #seg_volume_plot()
    

if __name__ == "__main__":
    main()


