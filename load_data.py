import os
import numpy as np
import nibabel as nib
from keras.utils import Sequence
import tensorflow as tf


def load_data(data_dir, img_shape, classes, batch_size=1, n_imgs=None, n_segs=None, generator=False, subfolder="T1w_brain"):

    imgs_path = os.path.join(data_dir, subfolder)
    segs_path = os.path.join(data_dir, "aseg")

    imgs_names = sorted(os.listdir(imgs_path))
    segs_names = sorted(os.listdir(segs_path))

    assert len(imgs_names) == len(segs_names)

    #with open("/home/exacloud/lustre1/fnl_lab/projects/BrainSegNet3D/data/classes.txt", "r") as c:
    #    classes = [int(line.rstrip()) for line in c]

    if generator:
        print("Creating data generator")
        return np_sequence(imgs_path, segs_path, imgs_names, segs_names, img_shape, classes, batch_size=batch_size)
    else:
        imgs_arr = create_np_array(imgs_names, imgs_path)
        segs_arr = create_np_array(segs_names, segs_path)
        #imgs_arr = create_image_array(imgs_names, imgs_path, img_shape, classes)
        #print("Done Loading Brain Images")
        #segs_arr = create_image_array(segs_names, segs_path, img_shape, classes)
        #print("Done Loading Segmentations")
        print(imgs_arr.shape)
        print(segs_arr.shape)
        train_dataset = tf.data.Dataset.from_tensor_slices((imgs_arr, segs_arr))
        #return {"brain_images": imgs_arr, "segmentations": segs_arr}
        return(train_dataset)

def create_np_array(imgs_names, imgs_path):
    image_arr_list = []
    for image_name in imgs_names:
        img = np.load(os.path.join(imgs_path, image_name))
        image_arr_list.append(img)
    return np.array(image_arr_list)

        

def create_image_array(image_list, image_path, img_shape, classes):
    image_arr_list = []
    for image_name in image_list:
        #image = np.array(Image.open(os.path.join(image_path, image_name)))
        img = nib.load(os.path.join(image_path, image_name))
        image = img.get_fdata()
        #print(image.shape)
        x1_pad = int(np.floor((img_shape[0]-image.shape[0])/2))
        x2_pad = int(np.ceil((img_shape[0]-image.shape[0])/2))
        y1_pad = int(np.floor((img_shape[1]-image.shape[1])/2))
        y2_pad = int(np.ceil((img_shape[1]-image.shape[1])/2))
        z1_pad = int(np.floor((img_shape[2]-image.shape[2])/2))
        z2_pad = int(np.ceil((img_shape[2]-image.shape[2])/2))
        image = np.pad(image,((x1_pad,x2_pad),(y1_pad,y2_pad),(z1_pad,z2_pad)), mode='constant')

        if "aseg" in image_path:
            print("Loading aseg: %s" % image_name)
            image = one_hot_encode(image, classes)
            #image = image[:, :, :, :, np.newaxis]
            
        else:
            print("Loading image: %s" % image_name)
            image = normalize_array(image)
            image = image[:, :, :, np.newaxis]

        image_arr_list.append(image)
    image_array = np.array(image_arr_list, dtype=object)

    return image_array

def one_hot_encode(aseg, classes):
    aseg = aseg.astype(int)
    #classes = np.unique(aseg)
    num_classes = len(classes)
    encoding_shape = aseg.shape + (num_classes,)
    encoding = np.zeros(encoding_shape)
    for i, c in enumerate(classes):
        encoding[:,:,:,i] = (aseg == c).astype(int)
    return encoding
    
  
  # If using 16 bit depth images, use the formula 'array = array / 32767.5 - 1' instead
def normalize_array(array):
    array = 2*(array - array.min())/(array.max() - array.min()) - 1
    #array = array / 127.5 - 1
    return array

class np_sequence(Sequence):

    def __init__(self, imgs_path, segs_path, imgs_names, segs_names, img_shape, classes, batch_size=1):
        self.batch_size = batch_size
        print("batch size = {}".format(self.batch_size))
        self.train_imgs = imgs_names
        self.train_segs = segs_names
        self.imgs_path = imgs_path
        self.segs_path = segs_path
        print("imgs_path = {}".format(self.imgs_path))
        print("segs_path = {}".format(self.segs_path))
        self.img_shape = img_shape
        print("img_shape = {}".format(self.img_shape))
        self.classes = classes
        print("classes = {}".format(self.classes))

    def __len__(self):
        return int(len(self.train_imgs) / int(self.batch_size))

    def __getitem__(self, idx):
        imgs_batch = self.train_imgs[idx * self.batch_size:(idx + 1) * self.batch_size]
        segs_batch = self.train_segs[idx * self.batch_size:(idx + 1) * self.batch_size]        
        imgs_arr = []
        for name in imgs_batch:
            img = np.load(os.path.join(self.imgs_path, name))
            imgs_arr.append(img)
        segs_arr = []
        for name in segs_batch:
            seg = np.load(os.path.join(self.segs_path, name))
            segs_arr.append(seg)

        print(np.array(imgs_arr).shape, np.array(segs_arr).shape)


        return np.array(imgs_arr), np.array(segs_arr)


class data_sequence(Sequence):

    def __init__(self, imgs_path, segs_path, imgs_names, segs_names, img_shape, classes, batch_size=1):  # , D_model, use_multiscale_discriminator, use_supervised_learning, REAL_LABEL):
        self.batch_size = batch_size
        self.train_imgs = imgs_names
        self.train_segs = segs_names
        self.imgs_path = imgs_path
        self.segs_path = segs_path
        self.img_shape = img_shape
        self.classes = classes

        self.shuffle = False

        self.on_epoch_end()
        #for image_name in image_list_A:
        #    if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
        #        self.train_A.append(os.path.join(trainA_path, image_name))
        #for image_name in image_list_B:
        #    if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
        #        self.train_B.append(os.path.join(trainB_path, image_name))

    def __len__(self):
        return int(len(self.train_imgs) / int(self.batch_size))

    def __getitem__(self, idx):
        #if idx >= len(self.train_imgs): 
        #    indexes = np.random.randint(len(self.train_imgs), size=self.batch_size)
        #    imgs_batch = []
        #    for i in indexes:
        #        imgs_batch.append(self.train_imgs[i])
        #    segs_batch = []
        #    for i in indexes:
        #        segs_batch.append(self.train_segs[i])
        #else:
        #print("Getting %s" % str(idx))
        imgs_batch = self.train_imgs[idx * self.batch_size:(idx + 1) * self.batch_size]
        segs_batch = self.train_segs[idx * self.batch_size:(idx + 1) * self.batch_size]        
        #print(imgs_batch)
        #print(segs_batch)
        imgs_arr = create_image_array(imgs_batch, self.imgs_path, self.img_shape, self.classes)
        segs_arr = create_image_array(segs_batch, self.segs_path, self.img_shape, self.classes)
        #print(imgs_arr.shape)
        #print(segs_arr.shape)


        return(np.array(imgs_arr), np.array(segs_arr))




if __name__ == '__main__':
    load_data()
