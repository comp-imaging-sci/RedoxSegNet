import lmdb
import os
import io
from PIL import Image
import torch.utils.data as data
import numpy as np
import random 

# import h5py
import torch
import glob
import torchvision.transforms as transforms



from tifffile import imread
import glob
class microscopy_new_inference(data.Dataset):

    def __init__(self, dataroot, phase,input_selection=False,input_channels=[0],sum_input=False):
        self.imgs = sorted(glob.glob(dataroot+ '/*.tif'))
        self.root = dataroot
        # self.imgs = self.imgs[:500]

        self.input_selection=input_selection
  
        self.channel_select=input_channels
        print(self.channel_select,self.input_selection)
        self.sum_input=sum_input
        if len(self.imgs)==0:
            raise(RuntimeError("file is empty"))

        
       

    
    def __getitem__(self, index):
        path = self.root
        temp_image = imread(self.imgs[index])
        temp_image=np.float32(temp_image)
        temp_name=self.imgs[index].split('/')
        if len(temp_image.shape)==2:
            temp_image = np.expand_dims(temp_image,axis=0)

        
        for slice_ind in range(temp_image.shape[0]):
            temp = temp_image[slice_ind]
            #normalization, cutting mean +-3std
            temp_mean,temp_std = temp.mean(), temp.std()
            temp[temp < temp_mean-3*temp_std] = temp_mean-3*temp_std
            # temp[temp > temp_mean+3*temp_std] = temp_mean+3*temp_std

            temp = (temp-temp.min())/(temp.max()-temp.min())
            temp_image[slice_ind] = temp

        if np.any(np.isnan(temp_image[self.channel_select])) or np.any(np.isinf(temp_image[self.channel_select])):
            print('broken data:'+temp_name[-1])
            
        data_x = temp_image
        if self.input_selection:
            data_x = temp_image[self.channel_select]
        if self.sum_input:
            data_x = data_x.sum(axis=0, keepdims=True)
        data_y = temp_image[[-1],]


        data_x=np.float32(data_x)
        data_y=np.float32(data_y)
        data_x=(data_x*1.0-0.5)*2
        data_y=(data_y*1.0-0.5)*2

        # data_x=np.expand_dims(data_x,axis=0)
        # data_y=np.expand_dims(data_y,axis=0)
        # data_y=np.expand_dims(data_y,axis=0)

        datax=torch.from_numpy(data_x)
        datay=torch.from_numpy(data_y)

        return  datax,datay,temp_name[-1]


    def __len__(self):
        return len(self.imgs)



class microscopy_new_translate(data.Dataset):

    def __init__(self, dataroot, phase,input_selection=False,input_channels=[0],out_channels=[-1],sum_input=False):
        # self.imgs = sorted(glob.glob(dataroot+ '/*.tif'))
        self.root = dataroot+phase
        self.imgs = sorted(glob.glob(self.root+ '/*.tif'))
        # self.imgs = self.imgs[:500]

        self.input_selection=input_selection
  
        self.channel_select=input_channels
        self.channel_select_out=out_channels
        print(self.channel_select+self.channel_select_out,self.input_selection)
        self.sum_input=sum_input
        if len(self.imgs)==0:
            raise(RuntimeError("file is empty"))

        self.augment = True if phase=='train' else False 
       

    
    def __getitem__(self, index):
        path = self.root
        if self.augment:
            rotate=index %6
            index=index//6

        temp_image = imread(self.imgs[index])
        temp_image=np.float32(temp_image)
        temp_name=self.imgs[index].split('/')
        if len(temp_image.shape)==2:
            temp_image = np.expand_dims(temp_image,axis=0)

        
        for slice_ind in self.channel_select+self.channel_select_out:
            temp = temp_image[slice_ind]
            #normalization, cutting mean +-3std
            temp_mean,temp_std = temp.mean(), temp.std()
            temp[temp < temp_mean-3*temp_std] = temp_mean-3*temp_std
            # temp[temp > temp_mean+3*temp_std] = temp_mean+3*temp_std

            temp = (temp-temp.min())/(temp.max()-temp.min())
            temp_image[slice_ind] = temp

        if np.any(np.isnan(temp_image[self.channel_select+self.channel_select_out])) or np.any(np.isinf(temp_image[self.channel_select+self.channel_select_out])):
            print('broken data:'+temp_name[-1])
            
        data_x = temp_image
        if self.input_selection:
            data_x = temp_image[self.channel_select]
        if self.sum_input:
            data_x = data_x.sum(axis=0, keepdims=True)
        data_y = temp_image[self.channel_select_out]

        if self.augment:
            if rotate in [1,2,3]:
                data_x=np.rot90(data_x,k=rotate,axes=(-2,-1))
                data_y=np.rot90(data_y,k=rotate,axes=(-2,-1))
            elif rotate==4:
                data_x=np.flip(data_x,axis=-1)
                data_y=np.flip(data_y,axis=-1)
            elif rotate==5:
                data_x=np.flip(data_x,axis=-2)
                data_y=np.flip(data_y,axis=-2)

        data_x=np.float32(data_x)
        data_y=np.float32(data_y)
        data_x=(data_x*1.0-0.5)*2
        data_y=(data_y*1.0-0.5)*2

        # data_x=np.expand_dims(data_x,axis=0)
        # data_y=np.expand_dims(data_y,axis=0)
        # data_y=np.expand_dims(data_y,axis=0)

        datax=torch.from_numpy(data_x.copy())
        datay=torch.from_numpy(data_y.copy())

        return  datax,datay

    def __len__(self):
        if self.augment:
            return len(self.imgs)*6
        else:
            return len(self.imgs)


