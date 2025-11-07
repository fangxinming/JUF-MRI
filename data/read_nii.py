import os
# from nibabel.viewers import OrthoSlicer3D
# from nibabel import nifti1
# import nibabel as nib
# from matplotlib import pylab as plt
# import matplotlib
# from PIL import Image
import numpy as np
import cv2
import SimpleITK as sitk #用于医学图像处理与分析
import torch
import torchvision.transforms
from torch.utils.tensorboard import SummaryWriter

def norm(data):
    """归一化数据"""
    data = data.astype(np.float32) #将数据类型转化为float32格式
    max = np.max(data)
    min = np.min(data)
    data = (data-min)/(max-min)  #将数据调整到0-1之间
    return data*255.               #再把数据重新映射到[0,255]范围内

#file_path = '/home/xinming/MCDudo/dataset/IXI/test/T2'
file_path = '/home/xinming/MCDudo/dataset/Brats2018/train/T2'
save_file_path = '/home/xinming/MCDudo/dataset/Brats2018/train/T2smallPNG'


step = 1
totle_number=0

file_list = sorted([f for f in os.listdir(file_path)
                    if os.path.isfile(os.path.join(file_path, f))
                    and not f.startswith('.')])  # 排除以点开头的隐藏文件夹（如 .ipynb_checkpoints）
#os.listdir用于列出指定路径下所有文件和子目录的名称
#sorted函数用于对可迭代对象进行排序，默认为升序排序
#with open("/home/lpc/dataset/IXI/T2/test.txt","r") as f:
    # open函数用于打开文件，r表示以只读模式打开文件
    #with语句用于自动管理资源，确保文件打开之后自动关闭
    # as f把打开的文件对象赋值给变量f，f可以用来操作文件
#writer = SummaryWriter("logs_qiepiantest1")
print(len(file_list))
for number, name in enumerate(file_list):
        #对每一个 file_list中的对象
    print(name)
    filename_1 = file_path+'/'+name   #路径名称加文件名
    img_1 = sitk.ReadImage(filename_1, sitk.sitkInt16)
        #读取图像，读取图像类型为16位有符号整数
    space = img_1.GetSpacing()
        #GetSpacing 用于获取图像空间分辨率的方法
        #返回的结果是一个元组 表示图像在每个维度上的物理间距
    img_1 = sitk.GetArrayFromImage(img_1)
        #把图像对象转化为numpy数组
    width, height, queue = img_1.shape
    print(width,height,queue)
    data_1 = norm(img_1)
        #对转化为numy数组的图像对象进行归一化
    skip = int((width - 20) / 2)
    for i in range(skip, skip+20, step):
            #skip为图像边缘像素值 step为步长
        totle_number = totle_number+1
        img_arr1 = data_1[i, :, :]
            #提取图像的第i个二维切片
        img_arr1 = np.expand_dims(img_arr1, axis=2)
            #将二维切片添加一个新维度，变成三位数组
        cv2.imwrite(save_file_path + '/{:08d}.png'.format(totle_number), img_arr1)
        #img_arr2 = np.expand_dims(img_arr1, axis=0)  # 添加 batch 维度
        #img_arr2 = np.transpose(img_arr2, (0, 3, 1, 2))
        # # 将形状 (batch, height, width, channels) 转换为 (batch, channels, height, width)
        #writer.add_images("PDPNG",img_arr2/255.0,i-skip)
        print('Done!'+save_file_path+'/{:08d}.png'.format(totle_number))

#writer.close()