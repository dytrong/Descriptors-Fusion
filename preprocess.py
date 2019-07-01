import numpy as np
import cv2
import torch
from PIL import Image
import configparser
import h5py
import torchvision.transforms as transforms

#####change images to tensor#####
def change_images_to_tensor(H5_Patch, norm_flag=False):
    ####初始化参数
    config = configparser.ConfigParser()
    config.read('./setup.ini')
    Model_Img_size = config.getint("DEFAULT", "Model_Image_Size") 
    img_to_tensor = transforms.ToTensor()  
    img_list=[]
    #the patch image .h5 file
    Img_h5=h5py.File(H5_Patch,'r')
    for i in range(len(Img_h5)):
        img=Img_h5[str(i)][:]
        ###change image format from cv2 to Image
        img=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img=img.resize((Model_Img_size, Model_Img_size))
        img=img_to_tensor(img)
        img=img.numpy()
        img=img.reshape(3, Model_Img_size, Model_Img_size)
        img_list.append(img)
    img_array=np.array(img_list)
    #####将数据转为tensor类型
    img_tensor=torch.from_numpy(img_array)
    #####数据标准化
    if norm_flag:
        img_tensor -= 0.443728476019
        img_tensor /= 0.20197947209
    return img_tensor

#######标准化
def normalization_desc(desc):
    mean_data = desc.mean(axis=1, keepdims=True)
    std_data = desc.std(axis=1, keepdims=True)
    desc -= mean_data
    desc /= std_data
    return desc

def normalization_desc_2(desc, dataset_type, pre_trained_model_type):
    if dataset_type == 'i':
        if pre_trained_model_type == 'alexnet':
            desc -= 0.6188732
            desc /= 1.0153735
        if pre_trained_model_type == 'vgg16':
            desc -= 0.3694579
            desc /= 1.2170433
        if pre_trained_model_type == 'resnet101':
            desc -= 0.17177103
            desc /= 0.21475737
        if pre_trained_model_type == 'densenet169':
            desc -= 0.035621334
            desc /= 0.08128328
    if dataset_type == 'v':
        if pre_trained_model_type == 'alexnet':
            desc -= 0.62761325
            desc /= 1.030684
        if pre_trained_model_type == 'vgg16':
            desc -= 0.38097617
            desc /= 1.2172571
        if pre_trained_model_type == 'resnet101':
            desc -= 0.3694579
            desc /= 1.2170433
        if pre_trained_model_type == 'densenet169':
            desc -=  0.03486648
            desc /= 0.081060626
    return desc
