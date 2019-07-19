# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 19:55:00 2019

@author: SuperLee
"""

import numpy as np
import cv2
import os
#import tables
import h5py

from random import shuffle
from tqdm import tqdm


def split_list(input_list, split = 0.8, shuffle_list = True):
    
    if shuffle_list:
        
        shuffle(input_list)
        
    n_training = int(len(input_list) * split)
    train = input_list[:n_training]
    val = input_list[n_training:]
    
    return train, val

#working_path=r"E:\SARship\QD_test\data/"
mask_path = r'H:\ship_segmentation_multiclass\firstjob\IS\first_exp\data_augment\mask/'
data_path = r'H:\ship_segmentation_multiclass\firstjob\IS\first_exp\data_augment\data_pywt/'

hdf5_path = r"H:\ship_segmentation_multiclass\firstjob\IS\first_exp\data_augment/IS_trainsetfirst.hdf5"


file_list = os.listdir(mask_path)
data_split = 0.8



num_image = len(file_list)

training_list, validation_list = split_list(file_list, split = data_split,shuffle_list = True)


train_num = len(training_list)
val_num = len(validation_list)

train_shape = (train_num, 512,512,9,1)
val_shape = (val_num, 512,512,9,1)

train_mask_shape = (train_num, 512,512,1,1)
val_mask_shape = (val_num, 512,512,1,1)


hdf5_file = h5py.File(hdf5_path,mode='w')

#count = hdf5_file['train_img'].shape[0]
#valcount = hdf5_file['val_img'].shape[0]

#hdf5_file.create_dataset("train_img",train_shape, maxshape=(None,256,256,9,1),dtype = np.float32)
#hdf5_file.create_dataset("val_img",val_shape, maxshape=(None,256,256,9,1),dtype = np.float32)
#
#hdf5_file.create_dataset("train_img_mask",train_mask_shape,maxshape=(None,256,256,1,1),dtype = np.float32)
#hdf5_file.create_dataset("val_img_mask",val_mask_shape,maxshape=(None,256,256,1,1),dtype =np.float32)



hdf5_file.create_dataset("train_img",train_shape,dtype = np.float32)
hdf5_file.create_dataset("val_img",val_shape, dtype = np.float32)

hdf5_file.create_dataset("train_img_mask",train_mask_shape,dtype = np.float32)
hdf5_file.create_dataset("val_img_mask",val_mask_shape,dtype =np.float32)

count = 0


for train_file in tqdm(training_list):
    
    
    pywt_a = cv2.imread(os.path.join(data_path,train_file[:-4]+'a.tif'),-1)
    pywt_b = cv2.imread(os.path.join(data_path,train_file[:-4]+'b.tif'),-1)
    pywt_c = cv2.imread(os.path.join(data_path,train_file[:-4]+'c.tif'),-1)
    pywt_d = cv2.imread(os.path.join(data_path,train_file[:-4]+'d.tif'),-1)
    pywt_e = cv2.imread(os.path.join(data_path,train_file[:-4]+'e.tif'),-1)
    pywt_f = cv2.imread(os.path.join(data_path,train_file[:-4]+'f.tif'),-1)
    pywt_g = cv2.imread(os.path.join(data_path,train_file[:-4]+'g.tif'),-1)
    pywt_h = cv2.imread(os.path.join(data_path,train_file[:-4]+'h.tif'),-1)
    pywt_o = cv2.imread(os.path.join(data_path,train_file[:-4]+'o.tif'),-1)
    
    data_mask = cv2.imread(os.path.join(mask_path,train_file),-1)
    
    data = np.ndarray([512,512,9,1],dtype = np.float32)
    mask = np.ndarray([512,512,1,1],dtype = np.float32)
    data[:,:,0,0] = pywt_a
    data[:,:,1,0] = pywt_b
    data[:,:,2,0] = pywt_c
    data[:,:,3,0] = pywt_d
    data[:,:,4,0] = pywt_e
    data[:,:,5,0] = pywt_f
    data[:,:,6,0] = pywt_g
    data[:,:,7,0] = pywt_h
    data[:,:,8,0] = pywt_o
    mask[:,:,0,0] = data_mask
    
    hdf5_file["train_img"][count,...] = data
    hdf5_file["train_img_mask"][count,...] = mask
    
    count = count + 1 

valcount = 0


for file in tqdm(validation_list):
    
    pywt_a = cv2.imread(os.path.join(data_path,file[:-4]+'a.tif'),-1)
    pywt_b = cv2.imread(os.path.join(data_path,file[:-4]+'b.tif'),-1)
    pywt_c = cv2.imread(os.path.join(data_path,file[:-4]+'c.tif'),-1)
    pywt_d = cv2.imread(os.path.join(data_path,file[:-4]+'d.tif'),-1)
    pywt_e = cv2.imread(os.path.join(data_path,file[:-4]+'e.tif'),-1)
    pywt_f = cv2.imread(os.path.join(data_path,file[:-4]+'f.tif'),-1)
    pywt_g = cv2.imread(os.path.join(data_path,file[:-4]+'g.tif'),-1)
    pywt_h = cv2.imread(os.path.join(data_path,file[:-4]+'h.tif'),-1)
    pywt_o = cv2.imread(os.path.join(data_path,file[:-4]+'o.tif'),-1)
    
    data_mask = cv2.imread(os.path.join(mask_path,file),-1)
    
    data = np.ndarray([512,512,9,1],dtype = np.float32)
    mask = np.ndarray([512,512,1,1],dtype = np.float32)
    data[:,:,0,0] = pywt_a
    data[:,:,1,0] = pywt_b
    data[:,:,2,0] = pywt_c
    data[:,:,3,0] = pywt_d
    data[:,:,4,0] = pywt_e
    data[:,:,5,0] = pywt_f
    data[:,:,6,0] = pywt_g
    data[:,:,7,0] = pywt_h
    data[:,:,8,0] = pywt_o
    mask[:,:,0,0] = data_mask
    
    hdf5_file["val_img"][valcount,...] = data
    hdf5_file["val_img_mask"][valcount,...] = mask
    
    valcount = valcount+1
    
    
hdf5_file.close() 
    
    
    

    