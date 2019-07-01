import torch.nn as nn
import torchvision.models as models  
import torch
from compute_distance import compute_cross_correlation_match
from compute_average_precision import compute_AP
from compute_keypoints_patch import * 
from forward import *
import time
from delete_dir import *
import sys
import os
import configparser
import argparse
from sklearn import preprocessing
from preprocess import *
from pre_trained_desc import *
sys.path.append('../../')
from utils.dimension_reduce_method import select_dimension_reduce_model, dimension_reduce_method

######初始化参数
config = configparser.ConfigParser()
config.read('./setup.ini')
Model_Img_size = config.getint("DEFAULT", "Model_Image_Size")
Max_kp_num = config.getint("DEFAULT", "Max_kp_num")
img_suffix = config.get("DEFAULT", "img_suffix")
txt_suffix = config.get("DEFAULT", "file_suffic")
Image_data = config.get("DATASET", "Hpatch_Image_Path")

######接收参数
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=['i','v'], required=True)
parser.add_argument("--pre_trained_descs_norm_flag", type=int, choices=[0,1], required=True)
parser.add_argument("--dimension", type=int, choices=[128,256,512,1024,2048,4096], required=True)
parser.add_argument("--reduce_flag", type=int, choices=[0,1], required=True)
parser.add_argument("--reduce_method", type=str,choices=['PCA', 'RP', 'AUTO', 'Isomap', 'noreduce'], required=True)

parser.add_argument("--autoencoder_parameter_path", type=str, required=True)
parser.add_argument("--pre_trained_model_name", type=str, choices=['alexnet','vgg16','resnet101','densenet169'])
parser.add_argument("--pre_trained_descs_path", type=str, required=False)

parser.add_argument("--fusion_flag", type=int, choices=[0,1], required=True)
parser.add_argument("--fusion_method", type=str, choices=['cat', 'CCA', 'AE', 'nofusion'], required=False)
parser.add_argument("--fusion_AE_flag", type=str, choices=['sigmoid', 'cnn','tanh', 'no'], required=False)
parser.add_argument("--fusion_dimension", type=int, choices=[0,32,64,128,256,512,1024,2048,4096], required=False)
parser.add_argument("--fusion_epoch", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--lr", type=float)
args = parser.parse_args()

#####加载CCA模型
if args.fusion_method == 'CCA':

    from CCA import CCA_model, CCA_dimension_reduce

    CCA_model = CCA_model(args.fusion_dimension)

def compute_batch_descriptors(mynet, Img_path, H5_Patch, mini_batch_size=8):

    ####generate patch img .h5 file and return valid key points
    valid_keypoints = compute_valid_keypoints(Img_path, H5_Patch, Max_kp_num)

    #####H5_patch_path用来保存每张图片的keypoint patches,方便计算CNN描述符
    input_data = change_images_to_tensor(H5_Patch)

    #####计算描述符
    desc = generate_des(mynet, input_data.cuda(), mini_batch_size, args.pre_trained_model_name).descriptor

    return valid_keypoints, desc

#######数据融合
def data_fusion(desc1, desc2, fusion_method = 'cat'):

    fusion_method = args.fusion_method

    ######拼接concatenate((array1,array2),axis=1)表示横向拼接
    if fusion_method == 'cat':

        desc = np.concatenate((desc1, desc2), axis=1)
    
    if fusion_method == 'CCA':

        desc = CCA_dimension_reduce(CCA_model, desc1, desc2)
     
    if fusion_method == 'AE':

        desc = np.concatenate((desc1, desc2), axis=1)

        desc = normalization_desc(desc)

        sys.path.append("/home/data1/daizhuang/pytorch/cnn_code/")
            
        from fusion_parameters.train_autoencoder_fusion import fusion_test_autoencoder

        AE_model_parameters_path = "/home/data1/daizhuang/pytorch/cnn_code/fusion_parameters/model_parameters/" \
                                        + args.fusion_AE_flag + "/" + args.pre_trained_model_name + '/' \
                                        + args.dataset + '/' + str(args.fusion_epoch) + "_4224_train_test_"
            
        AE_model_parameters_path = AE_model_parameters_path + args.dataset + '_' + args.pre_trained_model_name \
                                   + '_hardnet' + '_' + args.fusion_AE_flag + '_' + str(args.fusion_dimension) \
                                   + '_' + str(args.batch_size) + '_' + str(args.lr) + '.pth' 

        desc = fusion_test_autoencoder(AE_model_parameters_path, 
                                       args.fusion_dimension, 
                                       args.fusion_AE_flag, 
                                       'CPU', 
                                       desc)

        #####Fusion AE
        #AE_Fusion_model = "/home/data1/daizhuang/pytorch/cnn_code/fusion_parameters/AE_256_fusion/model_parameters/sigmoid/densenet169/v/"
        #AE_Fusion_model = AE_Fusion_model + "50_256_train_test_v_AE_fusion_sigmoid_128_64_0.0001.pth"

        #sys.path.append("/home/data1/daizhuang/pytorch/cnn_code/fusion_parameters/")
        #from AE_256_fusion.train_autoencoder_fusion import AE_fusion_test_autoencoder

        #desc = AE_fusion_test_autoencoder(AE_Fusion_model, 128, 'sigmoid', 'CPU', desc)

    return desc

def Hardnet_desc(Img_path, H5_Patch, norm_flag = True):

    #####导入HardNet代码路径
    sys.path.append('/home/data1/daizhuang/matlab/hardnet/examples/keypoint_match')

    from run_hardnet import compute_kp_and_desc

    kp, hardnet_desc = compute_kp_and_desc(Img_path, H5_Patch)

    if norm_flag:

        if args.dataset == 'i':
            hardnet_desc -= 0.0001433593
            hardnet_desc /= 0.08838826

        if args.dataset == 'v':
            hardnet_desc -= -0.000108431
            hardnet_desc /= 0.08838831

    return kp, hardnet_desc

def compute_fusion_model_descriptors(mynet,
                                     Img_path, 
                                     H5_Patch, 
                                     reduce_model, 
                                     normalization_flag = args.pre_trained_descs_norm_flag,  
                                     reduce_flag = args.reduce_flag, 
                                     fusion_flag = args.fusion_flag):

    ######pre-trained CNN descriptor
    kp, desc = compute_batch_descriptors(mynet, Img_path, H5_Patch)
    print("计算的描述符为:" + args.pre_trained_model_name)
    print("检测出的有效特征点数量为:" + str(desc.shape[0]))
    print("降维前描述符维度为:" + str(desc.shape[1]))

    ######标准化
    if normalization_flag:
        start = time.time()
        desc = normalization_desc(desc)
        print("描述符正则化耗时:"+str(time.time()-start))

    #####reduce dimension
    if reduce_flag:
        start = time.time()
        desc = dimension_reduce_method(reduce_model, desc, args.reduce_method, args.pre_trained_model_name)
        print("描述符降维耗时:"+str(time.time()-start))
        print("降维后的描述符维度:"+str(desc.shape[1])) 
    
    if fusion_flag:
        ####HardNet descriptors
        kp, hardnet_desc = Hardnet_desc(Img_path, H5_Patch)
        ###descriptor fusion
        start = time.time()
        desc = data_fusion(desc, hardnet_desc)
        #desc = [[sum((abs(row_desc[i]), abs(row_desc[i+1]))) for i in range(0, len(row_desc), 2)] for row_desc in desc]
        #desc = np.array(desc)
        #desc = preprocessing.normalize(desc, norm='l2') 
        print("描述符融合耗时:"+str(time.time()-start))
        print('融合后描述符维度:' + str(desc.shape[1]))
  
    return kp, desc

def compute_mAP(mynet, file_path, reduce_dimension_model):
    total_AP = []
    extract_desc_time = []
    compute_desc_dis_time = []
    #####子数据集地址
    base_path = Image_data + str(file_path) + '/'
    ####第１对图像对
    print("start compute the 1 pairs matches")
    Img_path_A = base_path+str(1)+img_suffix
    H5_Patch_A = './data/h5_patch/img'+str(1)+txt_suffix
    img1 = cv2.imread(Img_path_A)
    kp1,desc1 = compute_fusion_model_descriptors(mynet, Img_path_A, H5_Patch_A, reduce_dimension_model)
    for i in range(2,7):
        print("start compute the "+str(i)+" pairs matches")
        img_save_path = '/home/daizhuang/Image/' + str(file_path) + '_' + str(i) + '.jpg'
        ####ground truth of Homography 
        H_path = base_path+'H_1_'+str(i)
        ####读取图片
        Img_path_B = base_path+str(i)+img_suffix
        H5_Patch_B = './data/h5_patch/img'+str(i)+txt_suffix
        img2 = cv2.imread(Img_path_B)
        #############提取特征点，和卷积描述符
        start = time.time()
        kp2, desc2 = compute_fusion_model_descriptors(mynet, Img_path_B, H5_Patch_B, reduce_dimension_model)
        extract_desc_time.append(time.time()-start)
        ##############计算描述符之间的距离并寻找特征点匹配对
        start = time.time()
        ##### L2, cos
        distance_method = 'cos'
        match_cc = compute_cross_correlation_match(distance_method, des1=desc1, des2=desc2)
        compute_desc_dis_time.append(time.time()-start)
        ##############compute average precision(AP)
        AP = compute_AP(img1, img2, kp1, kp2, match_cc, H_path, distance_method, img_save_path, imshow=False)
        total_AP.append(AP)
    mAP = np.mean(total_AP)
    print('提取描述符平均耗时:'+str(np.mean(extract_desc_time)))
    print('计算描述符距离平均耗时:'+str(np.mean(compute_desc_dis_time)))
    print('5幅图像的平均精度:'+str(mAP))
    return mAP


if __name__ == "__main__":
    start = time.time()
    all_mAP = []
    Count = 0 
    #####选择预训练神经网络模型
    mynet = select_pretrained_cnn_model(flag = args.pre_trained_model_name)

    #####返回降维模型
    reduce_dimension_model = select_dimension_reduce_model(args.reduce_method, 
                                                           args.dimension,
                                                           args.pre_trained_descs_path,
                                                           args.autoencoder_parameter_path,
                                                           args.pre_trained_model_name
                                                          )
    #####遍历图像数据集, 输出所有数据的平均精度
    for roots, dirs, files in os.walk(Image_data):
        for Dir in dirs:
            if Dir[0] == args.dataset:
                print('读取的图像:'+Dir)
                Count = Count+1
                print('读取的图片张数:'+str(Count))
                mAP = compute_mAP(mynet, Dir, reduce_dimension_model)
                all_mAP.append(mAP)
                print('\n')

    print('所有数据的平均精度为:'+str(np.sum(all_mAP)/len(all_mAP)))
    print('总共耗时:'+str(time.time()-start))
