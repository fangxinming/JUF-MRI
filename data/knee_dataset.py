import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import os
import random
import torch.fft


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors.
    Args:
        x (torch.Tensor): A PyTorch tensor.
        shift (int): Amount to roll.
        dim (int): Which dimension to roll.
    Returns:
        torch.Tensor: Rolled version of x.
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=(-2, -1)):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    Args:
        x (torch.Tensor): A PyTorch tensor.
        dim (int): Which dimension to fftshift.
    Returns:
        torch.Tensor: fftshifted version of x.
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]

    return roll(x, shift, dim)


def real_to_complex(img):
    if len(img.shape) == 3:
        data = img.unsqueeze(0)
    else:
        data = img
    y = torch.fft.fftn(data, dim=(-2, -1))
    y = fftshift(y, dim=(-2, -1))  ## (1,1,h,w)
    y_complex = torch.cat([y.real, y.imag], 1)  ## (1,2,h,w)
    if len(img.shape) == 3:
        y_complex = y_complex[0]
    return y_complex


def complex_to_real(data):
    if len(data.shape) == 3:
        data1 = data.unsqueeze(0)
    else:
        data1 = data
    h, w = data.shape[-2], data.shape[-1]
    y_real, y_imag = torch.chunk(data1, 2, dim=1)
    y = torch.complex(y_real, y_imag)
    y = fftshift(y, dim=(-2, -1))  ## (1,1,h,w)
    y = torch.fft.ifftn(y, s=(h, w), dim=(-2, -1)).abs()
    if len(data.shape) == 3:
        y = y[0]
    return y


def crop_k_data(data, scale):
    _, h, w = data.shape
    lr_h = h // scale
    lr_w = w // scale
    top_left_h = h // 2 - lr_h // 2
    top_left_w = w // 2 - lr_w // 2
    croped_data = data[:, top_left_h:(top_left_h + lr_h), top_left_w:(top_left_w + lr_w)]
    return croped_data


class knee_train(data.Dataset):
    def __init__(self, opt, train):
        # 此处的opt为一个字典类型的变量， 通常用于储存各种配置参数或选项
        # 此处的opt为datasets:分为train和val
        super(knee_train, self).__init__()
        path = opt['dataroot_GT']
        GT_list = sorted(
            [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]
        )
        self.GT_paths = [os.path.join(path, i) for i in GT_list]
        self.train = train
        if self.train:
            self.crop_size = opt['crop_size']
        self.task = opt['task']
        self.scale = int(opt['scale'])
        self.hr_in = opt['hr_in']
        self.undersample = opt['undersample']
        self.hr_in = opt['hr_in']
        self.model = opt['model']
        self.model_G = opt['model_G']
        self.diffusion = opt['method']

    def __len__(self):
        return len(self.GT_paths)

    def __getitem__(self, idx):
        GT_img_path = self.GT_paths[idx]
        # 索引处对应的目标图像路径
        ref_GT_img_path = self.GT_paths[idx].replace('PDFS', 'PD')
        # 获取与目标图像对应的参考图像路径，通过将文件名中的 T2 替换为 PD

        if self.task == 'rec':
            diffusion_img_path = None
            if self.model == 'ref-rec' and self.undersample == 'learned' and self.diffusion in ['Loupe', 'mean']:
                mask_path1 = '/home/xinming/MCDudo/dataset/fastmri/knee_singlecoil/' + self.model_G + '_mask/'
                mask_path = mask_path1 + 'x' + str(self.scale) + '_mask_hard_fastmri.png'

                # 保存已学习掩码的路径
            elif self.model == 'ref-rec' and self.undersample == 'learned' and self.diffusion == 'diffusion':
                mask_path1 = '/home/xinming/MCDudo/dataset/fastmri/knee_singlecoil/' + self.model_G + '_mask/'
                mask_path = mask_path1 + 'x' + str(self.scale) + '_mask_hard_fastmri.png'


            elif self.model == 'ref-rec' and self.undersample != 'learned':
                mask_path1 = '/home/xinming/MCDudo/dataset/fastmri/mask/'
                mask_path = mask_path1 + 'mask_' + self.undersample + '_x' + str(self.scale) + '.png'


            else:  # 说明是共同优化，共同优化没有预定义mask
                mask_path = None
                if self.diffusion == 'diffusion':
                    diffusion_img_path = self.GT_paths[idx].replace('PDFS', 'smalldiffusion')



        elif self.task == 'sr':
            if self.scale == 2:
                mask_path = '/home/lpc/dataset/IXI/MCSR/mask_sr_x2.png'
            elif self.scale == 4:
                mask_path = '/home/lpc/dataset/IXI/MCSR/mask_sr_x4.png'
            else:
                print('Wrong scale for SR!')
        # 超分与重建的mask不同

        # read image file
        im1_GT = cv2.imread(GT_img_path, cv2.IMREAD_UNCHANGED)
        # cv2.IMREAD_UNCHANGED 表示以图像的原始格式读取图像文件，不对其进行任何修改或颜色空间转换
        # cv.imread读取的图像是一个Numpy数组 256*256
        im2_GT = cv2.imread(ref_GT_img_path, cv2.IMREAD_UNCHANGED)

        if diffusion_img_path == None:
            im3_GT = None
        else:
            im3_GT = cv2.imread(diffusion_img_path, cv2.IMREAD_GRAYSCALE)

        if mask_path == None:
            mask = None
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # 掩码是一张图像吗？
        im1_GT = torch.tensor(im1_GT).unsqueeze(0).float() / 255.
        # torch.tensor把Numpy数组转化为pytorch张量，unsqueeze是在张量的第一个维度上增加一个通道
        # /255指把像素值归一化到[0,1]
        im2_GT = torch.tensor(im2_GT).unsqueeze(0).float() / 255.
        # torch.tensor把Numpy数组转化为pytorch张量，unsqueeze是在张量的第一个维度上证加一个通道
        # /255指把像素值归一化到[0,1]

        if im3_GT is not None:
            im3_GT = torch.tensor(im3_GT).unsqueeze(0).float() / 255.

        if mask is not None:
            mask = torch.tensor(mask).unsqueeze(0).float().repeat(2, 1, 1) / 255
        # .repeat指的是在第一个维度上重复两次,这个mask的通道数量为2

        # FFT
        im1_GT_k = real_to_complex(im1_GT)
        im2_GT_k = real_to_complex(im2_GT)
        # 把目标图像与k空间图像进行傅里叶变换转换到图像域 k空间数据的维度为(2,h,w)
        # apply mask
        # zero-padding
        if self.hr_in and mask != None:
            im1_LQ_k = im1_GT_k * mask
            im2_LQ_k = im2_GT_k * mask
        elif self.hr_in and mask == None:
            im1_LQ_k = im1_GT_k
            im2_LQ_k = im2_GT_k
        #  如果mask==None的话掩码在训练过程中生成 ，输入时的亚采样图像就是原图像
        # 乘号逐元素相乘
        # center_crop
        else:
            im1_LQ_k = crop_k_data(im1_GT_k, self.scale)
            im2_LQ_k = crop_k_data(im2_GT_k, self.scale)
        # 根据scale对图像进行中心裁剪，裁剪大小为(原图像长*1/scale)×(原图像宽*1/scale)
        # IFFT
        im1_LQ = complex_to_real(im1_LQ_k)
        im2_LQ = complex_to_real(im2_LQ_k)
        # 把与掩码相乘之后的k空间数据或者中心裁剪后的k空间数据转化为图像数据

        # random crop
        if self.train:
            _, H, W = im1_GT.shape
            # 在这里应该是（1，256，256）
            if self.hr_in:
                rnd_h = random.randint(0, max(0, H - self.crop_size))
                rnd_w = random.randint(0, max(0, W - self.crop_size))
                # random.randint用于生成随机整数
                im1_GT = im1_GT[:, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size]
                im2_GT = im2_GT[:, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size]
                im1_LQ = im1_LQ[:, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size]
                im2_LQ = im2_LQ[:, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size]
                # 这四行代码将高分辨率和低分辨率图像按随机生成的坐标裁剪成一个大小为 self.crop_size × self.crop_size 的子区域
            else:
                rnd_h = random.randint(0, max(0, H // self.scale - self.crop_size))
                rnd_w = random.randint(0, max(0, W // self.scale - self.crop_size))
                rnd_h_HR, rnd_w_HR = int(rnd_h * self.scale), int(rnd_w * self.scale)
                im1_LQ = im1_LQ[:, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size]
                im2_LQ = im2_LQ[:, rnd_h:rnd_h + self.crop_size, rnd_w:rnd_w + self.crop_size]
                im1_GT = im1_GT[:, rnd_h_HR:rnd_h_HR + self.crop_size * self.scale,
                         rnd_w_HR:rnd_w_HR + self.crop_size * self.scale]
                im2_GT = im2_GT[:, rnd_h_HR:rnd_h_HR + self.crop_size * self.scale,
                         rnd_w_HR:rnd_w_HR + self.crop_size * self.scale]

        if mask == None and im3_GT == None:  # joint , loupe
            return {'im1_LQ': im1_LQ, 'im1_GT': im1_GT, 'im2_LQ': im2_LQ, 'im2_GT': im2_GT}
        elif mask == None and im3_GT is not None:  # joint ,loupe
            return {'im1_LQ': im1_LQ, 'im1_GT': im1_GT, 'im2_LQ': im2_LQ, 'im2_GT': im2_GT, 'im3_GT': im3_GT}
        else:  # ref-rec,只要是ref-rec 就一定有mask
            return {'im1_LQ': im1_LQ, 'im1_GT': im1_GT, 'im2_LQ': im2_LQ, 'im2_GT': im2_GT, 'mask': mask}

