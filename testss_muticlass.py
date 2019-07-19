import cv2
import tensorflow as tf
import os
import numpy as np
from trainss_muticlass import get_unet
from tqdm import tqdm
import h5py


os.environ['CUDA_VISIBLE_DEVICES']='1'

def predict(input_size, img_h, img_w, rows, columns, stride,n_labels, weight_path, test_data_path,pic_save_path):

    model = get_unet(input_size,n_labels)
    model.load_weights(weight_path)
    test_file = h5py.File(test_data_path,'r')
    img_num = test_file['data'].shape[0]

    color_map = np.zeros((rows * stride + img_h, columns * stride + img_w, 3), dtype = np.uint8)
    for i in tqdm(range(img_num)):

        test_data = test_file['data'][i,...]
        test_data = test_data / 65535.0
        test_data = test_data.reshape(1, 1024, 1024, 9, 1)
        
        test_result = model.predict(test_data, verbose = 0)[0]
        label_map = np.argmax(test_result, axis = 3)
        label = label_map.reshape((img_h,img_w)).astype(np.uint8)
        color_roi = np.zeros((img_h,img_w,3), dtype = np.uint8)
        
        for j in range(img_h):
            for k in range(img_w):
                if label[j][k] == 0:
                    color_roi[j, k, 0]= 240
                    color_roi[j, k, 1]= 241
                    color_roi[j, k, 2]= 242
                
                elif label[j][k] == 1:
                    
                    color_roi[j, k, 0]= 255
                    color_roi[j, k, 1]= 218
                    color_roi[j, k, 2]= 170

                elif label[j][k] == 2:
                    
                    color_roi[j, k, 0]= 174
                    color_roi[j, k, 1]= 136
                    color_roi[j, k, 2]= 192
        
        col_index = i // rows
        row_index = i % rows

        ly = row_index * stride
        lx = col_index * stride
        color_map[ly:ly + img_h, lx:lx + img_w,:] = color_roi

    test_file.close()
    cv2.imwrite(pic_save_path + '/QD_test_0521_patchsize512_stride256_fifth.tif', color_map[0:6844, 0:10835])




if __name__ == '__main__':

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.75
    session = tf.Session(config=config)
    input_size = (1024,1024,9,1)
    n_labels = 3
    batch_size = 1

    img_h = 1024
    img_w = 1024
    stride = 512
    # rows = int(6844 // stride) + 1
    # columns = int(10835 // stride) + 1

    rows = int(6844 // stride)
    columns = int(10835 // stride)

    weight_path = r'F:\2018\jcli\ship_segmentation_muticlass\fifth\model\ship_segmentation_muticlass_fifth.hdf5'
    test_data_path = r'F:\2018\jcli\SARship_data_npy\test_data_pad_1024_512stride/test_QD_1024patch_stride512.hdf5'
    pic_save_path = r'F:\2018\jcli\ship_segmentation_muticlass\fifth\result'
        
    print("Start to  predict ......")

    predict(input_size,img_h,img_w,rows,columns,stride,n_labels,weight_path,test_data_path,pic_save_path)

    print("All Predictions has been Done !!!")
    


        