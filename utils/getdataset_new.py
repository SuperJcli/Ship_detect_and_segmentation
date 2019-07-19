# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:17:08 2019

@author: SuperLee
"""

import cv2
import random
from tqdm import tqdm
import numpy as np 
import os
#import imutils


h = 512
w = 512

def mapcolor():
    '''
    read images by opencv:the channel order : [B,G,R]
    '''
    colormap =[[240,241,242],
              [255,218,170],
              [174,136,192]]
    
    mapMatrix = np.zeros(256*256*256,dtype = np.int32)
    
    for i,cm in enumerate(colormap):
        mapMatrix[cm[0]*65536+cm[1]*256+cm[2]] = i
    
    return mapMatrix

def color2label(label,mapMatrix):
    '''
    label:shape[h,w,3]
    mapMatrix: color --> label mappig matrix
    
    return: labelMatrix:shape [h,w]
    '''
    data = label.astype('int32')
    index = data[:,:,0]*65536 + data[:,:,1]*256 + data[:,:,2]
    
    return mapMatrix[index]

def rotate_muti(x,y,z,angle):

#    x = imutils.rotate_bound(x,angle)
#    y = imutils.rotate_bound(y,angle)
#    z = imutils.rotate_bound(z,angle)
    
    M_rotate = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    x = cv2.warpAffine(x,M_rotate,(w,h))
    y = cv2.warpAffine(y,M_rotate,(w,h))
    z = cv2.warpAffine(z,M_rotate,(w,h))
    
    return x,y,z


def flip(x,y,z,mode):
    
    '''
    mode = 0 : Vertical flip
    mode = 1 : horizontal flip
    mode =-1 : Vertical and horizontal flip
    '''
    x = cv2.flip(x,mode)
    y = cv2.flip(y,mode)
    z = cv2.flip(z,mode)
    
    return x,y,z

def data_augument_muticlass(X,Y,Z):

    if np.random.random() < 0.25:
        
        X,Y,Z = flip(X,Y,Z,1)
        
    if np.random.random() < 0.25:
        
        X,Y,Z = flip(X,Y,Z,-1)
        
    if np.random.random() < 0.25:
        
        X,Y,Z = flip(X,Y,Z,0)
        
    return X,Y,Z

'''
sub roi region

（row,column） [10297:12297,6992:8992]    2000*2000   land inside
（row,column） [8966:10966,20843:22843]   2000*2000   land boundary
（row,column） [8433:10433,15183:17183]   2000*2000   land inside
（row,column） [641:2641,1132:3132]       2000*2000   land boundary
（row,column） [13918:15918,23807:25807]  2000*2000    sea
（row,column） [508:2508,6993:8993]       2000*2000    land boundary
（row,column） [11696:13696,15516:17516]  2000*2000    land boundary

（row,column） [2172:4172,18779:20779]    2000*2000   land&ship&sea
（row,column） [14426:16426,10655:12655]  2000*2000     land&ship&sea
（row,column） [5636:7136,11476:16216]    1500*1500     ship&sea
（row,column） [4105:5605,14117:15617]    1500*1500     ship&sea
（row,column） [3393:4893,11711:13211]    1500*1500     ship&bridge
（row,column） [4970:5970,23810:24810]    1000*1000     ship in sea
（row,column） [7167:8167,17913:18913]    1000*1000     ship next to land

9.7% traindatas on whole image of size (17296,26606)

'''

def getdataloc():
    location = [[10297,12297,6992,8992],
                [8966,10966,20843,22843],
                [508,2508,6993,8993],
                [8433,10433,15183,17183],
                [11696,13696,15516,17516],
                [641,2641,1132,3132],
                [13918,15918,23807,25807],
                [5636,7136,11476,16216],
                [4105,5605,14117,15617],
                [3393,4893,11711,13211],
                [14426,16426,10655,12655],
                [2172,4172,18779,20779],
                [4970,5970,23810,24810],
                [7167,8167,17913,18913],
                ]
    return location

def create_dataset_on_whole(path,save_path, mode = 'original'):
    
    print('create datasets.................')
    
    src_origin = cv2.imread(path + '/IS_origin.tif',-1)
    color_origin = cv2.imread(path +'/IS_label.tif',-1)
    digital_mask = cv2.imread(path +'/IS_digital_label.tif',-1)
#    mapMatrix = mapcolor()
#    digital_mask = color2label(color_origin,mapMatrix)
#    cv2.imwrite(path+'/IS_digital_label.tif',digital_mask)
    
    location = getdataloc()
    count = 0
    for i,loc in enumerate(location):
        local_count = 0
        y_u = location[i][0]
        y_d = location[i][1]
        x_l = location[i][2]
        x_r = location[i][3]
    
        src = src_origin[y_u:y_d,x_l:x_r]
        mask = digital_mask[y_u:y_d,x_l:x_r]
        color_mask = color_origin[y_u:y_d,x_l:x_r]
    
        height,width = src.shape

        while local_count < 50:
            
            count += 1
            random_width = random.randint(0,width-w-1)
            random_height = random.randint(0,height-h-1)
            src_roi = src[random_height:random_height+h,random_width:random_width+w]
            mask_roi = mask[random_height:random_height+h,random_width:random_width+w]
            color_mask_roi = color_mask[random_height:random_height+h,random_width:random_width+w,:]
               
            if mode == 'augment':
                
                src_roi,mask_roi,color_mask_roi = data_augument_muticlass(src_roi,mask_roi,color_mask_roi)
        
            cv2.imwrite(save_path + '/data/{:s}.tif'.format(str(count).zfill(6)),src_roi)
            cv2.imwrite(save_path + '/color_mask/{:s}.tif'.format(str(count).zfill(6)),color_mask_roi)
            cv2.imwrite(save_path + '/mask/{:s}.tif'.format(str(count).zfill(6)),mask_roi)
            local_count += 1
            
            print('The {}th images has generate {} images'.format(i+1,local_count))

if __name__ == '__main__':

    root_path = r'H:\ship_segmentation_multiclass\firstjob\IS\first_exp\raw_data'
    save_path = r'H:\ship_segmentation_multiclass\firstjob\IS\first_exp\data_augment'
    mode = 'augment'
    create_dataset_on_whole(path = root_path,save_path = save_path , mode = mode)               
                    
                    