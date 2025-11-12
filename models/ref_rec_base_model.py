import logging
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from torchvision import models
import cv2
import torch.fft
from matplotlib import pyplot as plt
import os
import numpy as np

logger = logging.getLogger('base')


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
        data = img.unsqueeze(0)   #如果是三维，把他变成四维
    else:
        data = img
    y = torch.fft.fftn(data, dim=(-2, -1))  #data(B,C,H,W) 对最后两位进行傅里叶变换
    y = fftshift(y, dim=(-2, -1))  ## (1,1,h,w)
    y_complex = torch.cat([y.real, y.imag], 1)  ## (1,2,h,w)
    if len(img.shape) == 3:
        y_complex = y_complex[0]
    return y_complex

def vis_img(img, fname, ftype ,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.imshow(img, cmap='gray')
    figname = fname + '_' + ftype + '.png'
    figpath = os.path.join(output_dir, figname)
    plt.savefig(figpath)


def gaussian_kernel(kernel_size=15, sigma=3.0, channels=1):
    """ 生成 2D 高斯核（适用于 Conv2d） """
    x_coord = torch.arange(kernel_size) - kernel_size // 2
    y_coord = x_coord.clone()
    x_grid, y_grid = torch.meshgrid(x_coord, y_coord)

    kernel = torch.exp(-(x_grid ** 2 + y_grid ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()  # 归一化
    kernel = kernel.view(1, 1, kernel_size, kernel_size)  # 形状 (1,1,H,W)
    kernel = kernel.repeat(channels, 1, 1, 1)  # 复制通道

    return kernel


def gaussian_blur(img_tensor, kernel_size=15, sigma=3.0):
    """ 对 (b,1,h,w) 形状的张量进行高斯模糊 """
    b, c, h, w = img_tensor.shape
    kernel = gaussian_kernel(kernel_size, sigma, channels=c).to(img_tensor.device)
    padding = kernel_size // 2  # 保持尺寸不变
    img_low = F.conv2d(img_tensor, kernel, padding=padding, groups=c)
    return img_low



class LossWrapper(nn.Module):
    def __init__(self):
        super(LossWrapper, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.cl1_loss = nn.L1Loss()
        self.use_cl1_loss = True
    def forward(self, outputs, targets, complement=None, complement_target=None):
        l1_loss = self.l1_loss(outputs, targets)

        if self.use_cl1_loss:
            cl1_loss = self.cl1_loss(complement, complement_target)
            loss = 0.9*l1_loss + 0.1*cl1_loss
            return loss
        else:
            loss = l1_loss
            return loss

class FrequencyLoss(nn.Module):
    def __init__(self, gaussian_blur, lambda_low=0.2, lambda_high=0.8):
        """
        频率域损失（Frequency Loss）

        参数:
        - gaussian_blur: 传入的高斯模糊函数
        - lambda_low: 低频损失的权重 (默认 0.2)
        - lambda_high: 高频损失的权重 (默认 0.8)
        """
        super(FrequencyLoss, self).__init__()
        self.gaussian_blur = gaussian_blur  # 传入模糊函数
        self.lambda_low = lambda_low
        self.lambda_high = lambda_high
        self.l1_loss = nn.L1Loss()  # 标准 L1Loss

    def forward(self, fake, gt):
        """
        计算 Frequency Loss

        参数:
        - fake: 生成的图像 (b,c,h,w)
        - gt: 真实图像 (b,c,h,w)

        返回:
        - 组合损失 (l_pix)
        """
        # 计算低频成分
        fake_low = self.gaussian_blur(fake)
        gt_low = self.gaussian_blur(gt)

        # 计算高频成分
        fake_high = fake - fake_low
        gt_high = gt - gt_low

        # 计算 L1 低频和高频损失
        loss_low = torch.mean(torch.abs(fake_low - gt_low))
        loss_high = torch.mean(torch.abs(fake_high - gt_high))

        # 组合频率损失
        frequency_loss = 0.8 * loss_low + 0.2 * loss_high

        # 计算最终损失
        l_pix = 0.8 * self.l1_loss(fake, gt) + 0.2 * frequency_loss

        return l_pix




class BaseModel(BaseModel):
    def __init__(self, opt):
        super(BaseModel, self).__init__(opt)
        # 此处的opt为整个大opt


        self.rank = -1  # non dist training
        train_opt = opt['train']
        self.dwt = opt["dwt"]
        self.which_dataset = opt["mode"]
        self.model = opt["model"]
        self.method = opt["method"]
        self.pmask_slope = 5  # 掩码斜率
        self.sample_slope = 12  # 采样斜率
        self.sparsity = 1/opt['scale']
        self.image_dims = opt['network_G']['image_size']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        self.network = opt['network_G']['which_model_G']

        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss

            if self.network== "MTrans":
                loss_type= 'l1'
            elif self.network in ['TitanRNet_freloss15s3','TitanRNet_freloss15s3_Loupe','TitanRNet_freloss15s3_diffusion','TitanRNet0.5low_freloss15s3','TitanRNet0.2low_freloss15s3']:
                loss_type = 'guass'
            else:
                loss_type = train_opt['pixel_criterion']

            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'lp':
                self.cri_pix = LapLoss(max_levels=5).to(self.device)
            elif loss_type =='guass':
                self.cri_pix = FrequencyLoss(gaussian_blur).to(self.device)
            elif loss_type == 'cl1_loss':
                self.cri_pix = LossWrapper().to(self.device)

            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))


            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            # self.optimizer_G = torch.optim.SGD(optim_params, lr=train_opt['lr_G'], momentum=0.9, weight_decay=wd_G)
            self.optimizers.append(self.optimizer_G)
            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                    restarts=train_opt['restarts'],
                                                    weights=train_opt['restart_weights'],
                                                    gamma=train_opt['lr_gamma'],
                                                    clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    
    def feed_data(self, data):   

        self.im1_lr = data['im1_LQ'].to(self.device)
        self.im1_gt = data['im1_GT'].to(self.device)
        self.im2_lr = data['im2_LQ'].to(self.device)
        self.im2_gt = data['im2_GT'].to(self.device)
        if 'mask' in data:
            self.mask = data['mask'].to(self.device)
        else:
            self.mask=None
        if 'im3_GT' in data:
            self.im3_gt = data['im3_GT'].to(self.device)

        # if self.which_dataset == 'fastmri':
        #     self.im1_mean = data['mean_1'].to(self.device)
        #     self.im2_mean = data['mean_2'].to(self.device)
        #     self.im1_std = data['std_1'].to(self.device)
        #     self.im2_std = data['std_2'].to(self.device)


    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def compute_soft_edge_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W) float tensor 0-1
        return: (B,C,H,W) float tensor (未归一化)
        """
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.0:
            x = x / 255.0

        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [0., 0., 0.],
                                [1., 2., 1.]], device=x.device).view(1, 1, 3, 3)

        B, C, H, W = x.shape
        grad_x = []
        grad_y = []
        for c in range(C):
            gx = F.conv2d(x[:, c:c + 1, :, :], sobel_x, padding=1)
            gy = F.conv2d(x[:, c:c + 1, :, :], sobel_y, padding=1)
            grad_x.append(gx)
            grad_y.append(gy)
        grad_x = torch.cat(grad_x, dim=1)
        grad_y = torch.cat(grad_y, dim=1)

        denom = torch.sqrt(1.0 + grad_x ** 2 + grad_y ** 2)
        u_x = grad_x / denom
        u_y = grad_y / denom

        dux_dx = []
        duy_dy = []
        for c in range(C):
            dx = F.conv2d(u_x[:, c:c + 1, :, :], sobel_x, padding=1)
            dy = F.conv2d(u_y[:, c:c + 1, :, :], sobel_y, padding=1)
            dux_dx.append(dx)
            duy_dy.append(dy)
        dux_dx = torch.cat(dux_dx, dim=1)
        duy_dy = torch.cat(duy_dy, dim=1)
        I_edge = dux_dx + duy_dy
        I_min = I_edge.min(dim=1, keepdim=True)[0]
        I_min = I_min.min(dim=2, keepdim=True)[0]
        I_min = I_min.min(dim=3, keepdim=True)[0]

        I_max = I_edge.max(dim=1, keepdim=True)[0]
        I_max = I_max.max(dim=2, keepdim=True)[0]
        I_max = I_max.max(dim=3, keepdim=True)[0]

        I_edge_norm = (I_edge - I_min) / (I_max - I_min)

        return I_edge_norm

    def optimize_parameters(self, epoch, total_epoch):
        self.optimizer_G.zero_grad()
        if self.network == 'MTrans':
            # self.fake_1, self.ref_1 = self.netG(self.im1_lr, self.im2_gt,self.mask)
            # l_pix = self.cri_pix(self.fake_1, self.im1_gt,self.ref_1,self.im2_gt)
            self.fake_1, self.ref_1 = self.netG(self.im1_lr, self.im2_gt, mask= self.mask)
            l_pix = self.cri_pix(self.fake_1, self.im1_gt)
        elif self.network in ['edgeNet','edgeNet_v2','edgeNet_v3','edgeNet_v4','edgeNet_v5']:
            self.high_x, self.fake_1 = self.netG(self.im1_lr, self.im2_gt)
            self.real_softedge = self.compute_soft_edge_tensor(self.im1_gt)
            l_pix = 0.9*self.cri_pix(self.fake_1, self.im1_gt) +0.01*self.cri_pix(self.high_x, self.real_softedge)
        elif self.network in ['edgeNet_v6','edgeNet_v7','edgeNet_v8','edgeNet_v9','edgeNet_v10']:
            self.high_x, self.fake_1 = self.netG(self.im1_lr, self.im2_gt, mask= self.mask)
            self.real_softedge = self.compute_soft_edge_tensor(self.im1_gt)
            l_pix = 0.9*self.cri_pix(self.fake_1, self.im1_gt) +0.01*self.cri_pix(self.high_x, self.real_softedge)
        elif self.model == 'joint-optimize' and self.method in ['Loupe']:
            self.fake_1, self.mask = self.netG(self.im1_lr, self.im2_gt, self.mask)
            l_pix = self.cri_pix(self.fake_1, self.im1_gt)
            #print(self.fake_1.shape)
        elif self.model == 'joint-optimize' and self.method =='diffusion':
            self.fake_1, _ = self.netG(self.im1_lr, self.im2_gt, self.im3_gt, self.mask)
            l_pix = self.cri_pix(self.fake_1, self.im1_gt)
        else:
            self.fake_1 = self.netG(self.im1_lr, self.im2_gt, mask= self.mask)
            l_pix = self.cri_pix(self.fake_1, self.im1_gt)

        total_loss = l_pix
        total_loss.backward()
        self.optimizer_G.step()
        self.log_dict['l_pix'] = l_pix.item()

        #item是从张量中得出一个数值
    
    
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if self.network == 'MTrans':
                self.fake_H, self.ref_1 = self.netG(self.im1_lr, self.im2_gt,self.mask)
                #print(self.fake_H.shape)
            elif self.network in ['edgeNet','edgeNet_v2','edgeNet_v3','edgeNet_v4','edgeNet_v5']:
                self.high_x, self.fake_H = self.netG(self.im1_lr, self.im2_gt)
            elif self.network in ['edgeNet_v6','edgeNet_v7','edgeNet_v8','edgeNet_v9','edgeNet_v10']:
                self.high_x, self.fake_H = self.netG(self.im1_lr, self.im2_gt,self.mask)

            elif self.model == 'joint-optimize' and self.method in ['Loupe']:
                self.fake_H, self.mask= self.netG(self.im1_lr, self.im2_gt, self.mask)
            elif self.model == 'joint-optimize' and self.method == 'diffusion':
                self.fake_H, self.pmask = self.netG(self.im1_lr, self.im2_gt, self.im3_gt, self.mask)
                y2_k = real_to_complex(self.im3_gt)  # y2_k形状为(B,2,H,W)
                y1_k = real_to_complex(self.im2_gt)  # y1_k形状为(B,2,H,W)

                r = torch.abs(torch.complex(y1_k[:, 0, :, :], y1_k[:, 1, :, :]) - torch.complex(y2_k[:, 0, :, :],
                                                                                                y2_k[:, 1, :, :]))
                r = r.unsqueeze(1)
                min_r = r.min()
                max_r = r.max()

                # 归一化到 [0, 1]
                norm_r = (r - min_r) / (max_r - min_r)  # r的形状为(B,1,H,W)
                #print("self.pmask.shape:", self.pmask.shape)

                pmask_temp = self.pmask.unsqueeze(0).unsqueeze(0).repeat(norm_r.shape[0], 1, 1, 1)  # pmask_temp 的形状为 (B, 1, H, W)
                pmask_temp = pmask_temp + norm_r  # 更新临时掩码

                # self.pmask = self.pmask.unsqueeze(0).unsqueeze(0).repeat(norm_r.shape[0], 1, 1, 1) #此时pmask的形状为(B,1,H,W)
                # self.pmask=self.pmask+norm_r
                probmask = self.squash_mask(pmask_temp)  # 初始化并把数据映射到[0.01.0.99]之间

                # Sparsify, control the sampling ratio

                sparse_mask = self.sparsify(probmask)  # 把整个稀疏度变化为要求的稀疏度  整个均值等于sparsity
                # sparse_mask的形状为(B,1,256,256) 每个batch的均值为sparsify
                # generate soft mask
                #print(sparse_mask.shape)
                self.mask = self.sigmoid_beta(sparse_mask)
                # print(self.mask.shape)
            else:
                self.fake_H = self.netG(self.im1_lr, self.im2_gt, mask=self.mask)

        self.netG.train()



    def squash_mask(self, mask):
        return torch.sigmoid(self.pmask_slope * mask)

    def sparsify(self, mask):  # 此时mask的形状为(B,1,H,W)
        mask = mask.clone()  # 避免修改原始张量
        xbar = torch.mean(mask, dim=(2, 3))  # 计算整个mask的均值  代表目前掩码的稀疏度 xbar的形状为(B,1)
        r = self.sparsity / xbar  # 稀疏度除以均值    要求的稀疏度除以目前的稀疏度 r的形状也是(B,1) 广播机制
        beta = (1 - self.sparsity) / (1 - xbar)  # beta 的形状也是(B,1)
        le = (r <= 1).float()  # 如果r<=1 说明要求的稀疏度比目前的稀疏度小 说明现在采样采多了 如果r>1 说明目前采样采少了
        for i in range(mask.shape[0]):
            mask[i] = le[i] * mask[i] * r[i] + (1 - le[i]) * (1 - (1 - mask[i]) * beta[i])
        return mask
        # 如果r<=1 即采样采多了 le =1   则return   mask * r 整个稀疏度变为要求稀疏度
        # 如果 r>1 即说明采样采少了 le =0 则return   1-（1-mask）* beta 整个稀疏度变为要求稀疏度

    def sigmoid_beta(self, mask):
        random_uniform = torch.empty(*self.image_dims).uniform_(0, 1).cuda()
        # torch.empty 为创建一个具有self.image_dims 的空张量 uniform(0,1)在空张量中填充随机值，这些值在[0,1]之间

        random_uniform = random_uniform.unsqueeze(0).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)

        return torch.sigmoid(self.sample_slope * (mask - random_uniform))
        # random_uniform 的形状为(B,1,H,W)

    def get_current_log(self):
        return self.log_dict
    
    def get_current_visuals(self, need_GT=True):

        out_dict = OrderedDict()
        # if self.which_dataset == 'fastmri':
        #     out_dict['im1_restore'] = out_dict['im1_restore']*self.im1_std.cpu() + self.im1_mean.cpu()
        #     out_dict['im1_GT'] = out_dict['im1_GT']*self.im1_std.cpu() + self.im1_mean.cpu()
        # else:
        out_dict['im1_restore'] = self.fake_H.detach().float().cpu()
        out_dict['im1_GT'] = self.im1_gt.detach().float().cpu()
        out_dict['mask'] = self.mask.detach().float().cpu()
        out_dict['im1_lr'] = self.im1_lr.detach().float().cpu()
        out_dict['im2_GT'] = self.im2_gt.detach().float().cpu()

        if self.network in ['edgeNet','edgeNet_v2','edgeNet_v3','edgeNet_v4','edgeNet_v5','edgeNet_v6','edgeNet_v7','edgeNet_v8','edgeNet_v9','edgeNet_v10']:
            out_dict['high_x'] = self.high_x.detach().float().cpu()


        
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label, epoch=None):
        if epoch != None:
            self.save_network(self.netG, 'G', iter_label, str(epoch))
        else:
            self.save_network(self.netG, 'G', iter_label)
