from __future__ import print_function

import numpy as np

import os
from glob import glob
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, UpSampling3D, Dropout,AveragePooling3D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping,ReduceLROnPlateau
from PIL import Image
import keras.backend as K
import tensorflow as tf 
import h5py
from math import ceil
from random import shuffle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from keras.utils.np_utils import to_categorical 

from metrics import dice_coefficient_loss,get_label_dice_coefficient_function,dice_coefficient

# from history_plot import LossHistory

os.environ["CUDA_VISIBLE_DEVICES"]= '1'

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_np(y_true,y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def img_label_encoder(img_block, batch_size, n_labels):
    classes = [0., 1., 2.]
    encoder = LabelEncoder()
    encoder.fit(classes)
    label = img_block.flatten()
    label = encoder.transform(label)
    train_label = to_categorical(label,num_classes = n_labels)
    one_hot_label = train_label.reshape((batch_size,512,512,1,n_labels))

    return one_hot_label

def generate_train_data(datasets, batch_size, n_labels):
    
    train_nums = datasets["train_img"].shape[0] 
    batches = (train_nums + batch_size - 1)//batch_size
   
    while(True):
        for i in range(batches):
            start = i*batch_size
            end = min((i+1)*batch_size,train_nums)

            X = datasets["train_img"][start:end,:,:,:,:]
            X = X / 65535.0
            y = datasets["train_img_mask"][start:end,:,:,:,:]
            Y = img_label_encoder(y,batch_size,n_labels)
            
            yield (X,Y)

def generate_val_data(datasets, batch_size, n_labels):
    
    val_nums = datasets["val_img"].shape[0] 
    batches = (val_nums + batch_size-1) // batch_size

    while(True):
        for i in range(batches):
            start = i*batch_size
            end = min((i+1)*batch_size,val_nums)

            X = datasets["val_img"][start:end,:,:,:,:]
            X = X / 65535.0
            y = datasets["val_img_mask"][start:end,:,:,:,:]
            Y = img_label_encoder(y,batch_size,n_labels)

            yield (X,Y)

def get_unet(input_size, n_labels):
    inputs = Input(input_size)
    conv1 = Conv3D(8, (3,3,3),activation ='relu',padding ='same',data_format ='channels_last',kernel_initializer ='he_normal')(inputs)
    conv1 = Conv3D(8, (3,3,3),activation = 'relu',padding='same',data_format='channels_last',kernel_initializer='he_normal',dilation_rate=3)(conv1)
    pool1 = AveragePooling3D(pool_size=(2,2,1),strides = None,padding='valid',data_format=None)(conv1) # 128*128*9*8

    conv2 = Conv3D(8, (3,3,3), activation = 'relu', padding = 'same',data_format="channels_last", kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv3D(8, (3,3,3), activation = 'relu', padding = 'same',data_format="channels_last", kernel_initializer = 'he_normal',dilation_rate=3)(conv2)
    pool2 = AveragePooling3D(pool_size=(2,2,1),data_format="channels_last")(conv2)  # 64*64*9*8

    conv3 = Conv3D(8, (3,3,3), activation = 'relu', padding = 'same',data_format="channels_last", kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv3D(8,(3,3,3), activation = 'relu', padding = 'same',data_format="channels_last", kernel_initializer = 'he_normal',dilation_rate=3)(conv3)
    pool3 = AveragePooling3D(pool_size = (2,2,1),data_format = 'channels_last')(conv3) # 32*32*9*8

    conv4 = Conv3D(8,(3,3,3), activation = 'relu',padding = 'same',data_format = 'channels_last',kernel_initializer='he_normal')(pool3)
    conv4 = Conv3D(8,(3,3,3), activation = 'relu',padding = 'same',data_format = 'channels_last',kernel_initializer='he_normal',dilation_rate=3)(conv4)
    pool4 = AveragePooling3D(pool_size = (2,2,1),data_format = 'channels_last')(conv4) # 16*16*9*8

    conv5 = Conv3D(16,(3,3,3),activation = 'relu',padding = 'same',data_format = 'channels_last',kernel_initializer='he_normal')(pool4)
    conv5 = Conv3D(16,(3,3,3),activation = 'relu',padding = 'same',data_format = 'channels_last',kernel_initializer='he_normal',dilation_rate=3)(conv5)
    drop5 = Dropout(0.5)(conv5)


    up6 = Conv3D(8,2,activation = 'relu', padding = 'same', data_format="channels_last",kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,1),data_format="channels_last")(drop5))
    merge6 = concatenate([conv4,up6] ,axis = 4)   #32*32*9*16

    conv6 = Conv3D(8, (3,3,3), activation = 'relu', padding = 'same', data_format="channels_last",kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv3D(8, (3,3,3), activation = 'relu', padding = 'same',data_format="channels_last", kernel_initializer = 'he_normal',dilation_rate=2)(conv6)
    
    up7 = Conv3D(8,2,activation = 'relu', padding = 'same',data_format="channels_last", kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,1),data_format="channels_last")(conv6))
    merge7 = concatenate([conv3,up7], axis = 4)   # 64*64*9*16

    conv7 = Conv3D(8, (3,3,3), activation = 'relu', padding = 'same',data_format="channels_last", kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv3D(8, (3,3,3), activation = 'relu', padding = 'same',data_format="channels_last", kernel_initializer = 'he_normal',dilation_rate=3)(conv7)

    up8 = Conv3D(8,2,activation = 'relu', padding = 'same',data_format="channels_last", kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,1),data_format="channels_last")(conv7))
    merge8 = concatenate([conv2,up8], axis = 4)   # 128*128*9*16

    conv8 = Conv3D(8, (3,3,3), activation = 'relu', padding = 'same',data_format="channels_last", kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv3D(8, (3,3,3), activation = 'relu', padding = 'same',data_format="channels_last", kernel_initializer = 'he_normal',dilation_rate=3)(conv8)

    up9 = Conv3D(8,2,activation = 'relu', padding = 'same',data_format="channels_last", kernel_initializer = 'he_normal')(UpSampling3D(size = (2,2,1),data_format="channels_last")(conv8))
    merge9 = concatenate([conv1,up9], axis = 4)   # 256*256*9*16

    conv9 = Conv3D(8, (3,3,3), activation = 'relu', padding = 'same',data_format="channels_last", kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv3D(8, (3,3,3), activation = 'relu', padding = 'same',data_format="channels_last", kernel_initializer = 'he_normal',dilation_rate=3)(conv9)

    conv10 = Conv3D(n_labels, (1,1,9), activation ='softmax',data_format="channels_last", kernel_initializer = 'he_normal')(conv9)
    model = Model(input=inputs, output=conv10)

    # label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]

    # model.compile(optimizer = Adam(lr=0.0001), loss = dice_coefficient_loss, metrics = label_wise_dice_metrics)
    model.compile(optimizer= Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics= ['accuracy'])

    la=[layer for layer in model.layers]
    print(la)
    model.summary()
    
    return model

def train_model(input_size, n_labels, fpath, save_path,batch_size,epochs):
    if not os.path.exists(fpath):
        print("fapth is not exist.")
        exit
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print("loading data.....")
    print('loading data done.')
    print('building 3d_net....')

    data = h5py.File(fpath +'/train_datasets_muticlass_fifth.hdf5', 'r')
    train_nums = data["train_img"].shape[0]
    val_nums = data["val_img"].shape[0]
    
    model = get_unet(input_size, n_labels)
    print('ok')

    filepath = save_path + "/ship_segmentation_muticlass_fifth.hdf5"

    use_existing = True
    if use_existing:
        model.load_weights(filepath)

    model_checkpoint = ModelCheckpoint(filepath, monitor= 'val_acc',verbose=1, save_best_only=True, mode = 'max')
    reduce_lr = ReduceLROnPlateau(monitor='val_acc',factor=0.5,patience=10,mode='auto', min_lr=0.00000001, verbose = 1)
    tbCallBack = TensorBoard(log_dir=save_path+'/logs', histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=0, embeddings_metadata=None)
    
    print('Fitting model....')
    
    H = model.fit_generator(generator = generate_train_data(data, batch_size = batch_size, n_labels = n_labels),
                    steps_per_epoch = (train_nums + batch_size - 1) // batch_size,
                    epochs = epochs,
                    verbose = 1,
                    callbacks = [EarlyStopping(monitor='val_acc',patience=20, mode='max'),
                        model_checkpoint,reduce_lr,tbCallBack],
                    validation_data = generate_val_data(data, batch_size = batch_size, n_labels= n_labels),
                    validation_steps = val_nums // batch_size )
    
    data.close()

    #plt.style.use('ggplot')
    #plt.figure()
    #N = epochs
    #plt.plot(np.arange(0,N),H.history['loss'],label = 'train_loss')
    #plt.plot(np.arange(0,N),H.history['val_loss'],label = 'val_loss')
    #plt.plot(np.arange(0, N), H.history["acc"], label="train_dice")
    #plt.plot(np.arange(0, N), H.history["val_acc"], label="val_dice")
    #plt.title("Training loss and dice on 3DDC-Net ship segmentation")
    #plt.xlabel("Epoch #")
    #plt.ylabel("loss/dice_coef")
    #plt.legend(loc='upper right')
    #plt.show()
    
if __name__ == '__main__':
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.75
    session = tf.Session(config=config)

    data_path = r'F:\2018\jcli\ship_segmentation_muticlass\fifth\data'
    save_path = r'F:\2018\jcli\ship_segmentation_muticlass\fifth\model'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    input_size = (512,512,9,1)
    batch_size = 4
    epochs = 200
    n_labels = 3
    train_model(input_size,n_labels,data_path,save_path,batch_size,epochs)
    print('The process of training already be done !')
