import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, ConvTranspose2d, MaxPool2d

from .module_util import *
import functools

try:
    from models.modules.DCNv2.dcn_v2 import DCN_sep
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')


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


#  This part was built based on LOUPE: https://github.com/cagladbahadir/LOUPE/.
class LoupeLayer(nn.Module):
    def __init__(self, image_dims=[240, 240], pmask_slope=5, sample_slope=12, sparsity=0.05, hard=False, mode="relax"):
        super().__init__()

        self.image_dims = image_dims
        self.pmask_slope = pmask_slope  # 掩码斜率
        self.sample_slope = sample_slope  # 采样斜率
        self.sparsity = sparsity  # 稀疏度
        self.mode = mode
        self.eps = 0.01
        self.hard = hard  # 是否使用硬掩码

        # Mask Initial
        self.pmask = nn.Parameter(torch.FloatTensor(*self.image_dims))  # 初始化一个与输入图像尺寸相同的参数 self.pmask，这个参数会在训练过程中被优化
        # Mask is same dimension as image plus complex domain

        self.pmask.requires_grad = True

        self.pmask.data.uniform_(self.eps, 1 - self.eps)  # self.pmask 中的每个元素设置为在 [self.eps, 1 - self.eps] 区间内均匀分布的随机值
        # 在此处为[0.01,0.99]
        self.pmask.data = -torch.log(1. / self.pmask.data - 1.) / self.pmask_slope  # 数据在[-0,919,0.919]范围之间

    def squash_mask(self, mask):
        return torch.sigmoid(self.pmask_slope * mask)

    def sparsify(self, mask):   #此时mask的形状为(B,1,H,W)
        mask = mask.clone()  # 避免修改原始张量
        xbar = torch.mean(mask, dim=(2, 3))  # 计算整个mask的均值  代表目前掩码的稀疏度 xbar的形状为(B,1)
        r = self.sparsity / xbar  # 稀疏度除以均值    要求的稀疏度除以目前的稀疏度 r的形状也是(B,1) 广播机制
        beta = (1 - self.sparsity) / (1 - xbar)   #beta 的形状也是(B,1)
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
        #random_uniform 的形状为(B,1,H,W)

        return torch.sigmoid(self.sample_slope * (mask - random_uniform))
        # 从mask中减去随机数张量

    def forward(self,y1,y2):  # x1,y1,y2 形状为(B,C,H,W)
        # initialize and squash

        y2_k = real_to_complex(y2)  # y2_k形状为(B,2,H,W)
        y1_k = real_to_complex(y1)  # y1_k形状为(B,2,H,W)

        r = torch.abs(torch.complex(y1_k[:, 0, :, :], y1_k[:, 1, :, :]) - torch.complex(y2_k[:, 0, :, :], y2_k[:, 1, :, :]))
        r = r.unsqueeze(1)
        min_r = r.min()
        max_r = r.max()

        # 归一化到 [0, 1]
        norm_r = (r - min_r) / (max_r - min_r)  #r的形状为(B,1,H,W)

        pmask_temp = self.pmask.unsqueeze(0).unsqueeze(0).repeat(norm_r.shape[0], 1, 1,1) # pmask_temp 的形状为 (B, 1, H, W)
        pmask_temp = pmask_temp + norm_r  # 更新临时掩码


        #self.pmask = self.pmask.unsqueeze(0).unsqueeze(0).repeat(norm_r.shape[0], 1, 1, 1) #此时pmask的形状为(B,1,H,W)
        #self.pmask=self.pmask+norm_r
        probmask = self.squash_mask(pmask_temp)  # 初始化并把数据映射到[0.01.0.99]之间

        # Sparsify, control the sampling ratio

        sparse_mask = self.sparsify(probmask)  # 把整个稀疏度变化为要求的稀疏度  整个均值等于sparsity
        #sparse_mask的形状为(B,1,256,256) 每个batch的均值为sparsify
        # generate soft mask
        mask = self.sigmoid_beta(sparse_mask)

        # return soft mask
        return mask


class DCN_Align(nn.Module):
    def __init__(self, nf=32, groups=4):
        super(DCN_Align, self).__init__()

        self.offset_conv1_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # down1
        self.offset_conv3_1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.offset_conv4_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # down2
        self.offset_conv6_1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.offset_conv7_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # up2
        self.offset_conv1_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.offset_conv2_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # up1
        self.offset_conv3_2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.offset_conv4_2 = nn.Conv2d(nf, 32, 3, 1, 1, bias=True)

        self.dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
                               deformable_groups=4)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, fea1, fea2):
        '''align other neighboring frames to the reference frame in the feature level
        estimate offset bidirectionally
        '''
        # B,64,H,W
        offset = torch.cat([fea1, fea2], dim=1)


        offset = self.lrelu(self.offset_conv1_1(offset))
        offset1 = self.lrelu(self.offset_conv2_1(offset))
        # down1
        offset2 = self.lrelu(self.offset_conv3_1(offset1))
        offset2 = self.lrelu(self.offset_conv4_1(offset2))
        # down2
        offset3 = self.lrelu(self.offset_conv6_1(offset2))
        offset3 = self.lrelu(self.offset_conv7_1(offset3))
        # up1
        offset = F.interpolate(offset3, scale_factor=2, mode='bilinear', align_corners=False)
        offset = self.lrelu(self.offset_conv1_2(torch.cat((offset, offset2), 1)))
        offset = self.lrelu(self.offset_conv2_2(offset))
        # up2
        offset = F.interpolate(offset, scale_factor=2, mode='bilinear', align_corners=False)
        offset = self.lrelu(self.offset_conv3_2(torch.cat((offset, offset1), 1)))
        base_offset = self.offset_conv4_2(offset)
        # DCN warping
        aligned_fea = self.dcnpack(fea2, base_offset)

        return aligned_fea

class JUF_MRI_diffusion(nn.Module):
    def __init__(self, opt):
        # 此处的opt为opt_net 为opt['network_G']
        super(JUF_MRI_diffusion, self).__init__()

        self.image_channel = opt['c_image']
        # 图像通道，MRI图像通道为1

        self.channel_in = opt['nf']  # c_i, default=32
        self.channel_fea = opt['nf']  # c_f, default=32
        self.iter_num = opt['stages']  # iteration stages T, default=4
        # 四个迭代阶段

        image_size = opt['image_size']
        # 图像尺寸，对于IXI数据集来说为256*256
        acc_ratio = opt['sparsity']
        # 代表加速率 10倍加速率则为0.1

        # learnable mask from Loupe
        self.LoupeLayer = LoupeLayer(image_dims=image_size, sparsity=acc_ratio)
        self.pmask = self.LoupeLayer.pmask

        # variable initialization
        basic_block1 = functools.partial(ResidualBlock_noBN, nf=self.channel_fea)
        # basic_blockl为一个类，表示一个没有BN的残差学习块，这个残差块中的卷积层经过了He初始化
        self.init_x = make_layer(basic_block1, 10)
        # n_layers在这里是10个残差块

        basic_block2 = functools.partial(ResidualBlock_noBN, nf=self.channel_fea)
        self.map_y = make_layer(basic_block2, 10)

        basic_block3 = functools.partial(ResidualBlock_noBN, nf=self.channel_fea)
        #self.init_s = Unet_denoise(cin=self.channel_in, n_feat=self.channel_fea)
        self.init_s =make_layer(basic_block3, 10)




        ## DAS module for spatial transformation
        self.dcn_align = DCN_Align(nf=32, groups=4)
        basic_block4 = functools.partial(ResidualBlock_noBN, nf=self.channel_fea)
        self.extract_y = make_layer(basic_block4, 10)

        ## convolutional layers for feature transformation
        # map x to feature domain
        self.trans_E_x = nn.Sequential(
            *[nn.Conv2d(in_channels=self.channel_in, out_channels=self.channel_fea, kernel_size=3, padding=1), \
              nn.ReLU(), \
              nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_fea, kernel_size=3, padding=1), \
              nn.ReLU(), \
              nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_fea, kernel_size=3, padding=1)])
        # map y to feature domain
        self.trans_E_y = nn.Sequential(
            *[nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_fea, kernel_size=3, padding=1), \
              nn.ReLU(), \
              nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_fea, kernel_size=3, padding=1), \
              nn.ReLU(), \
              nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_in, kernel_size=3, padding=1)])
        # map x feature to image domain
        self.trans_D_x = nn.Sequential(
            *[nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_fea, kernel_size=3, padding=1), \
              nn.ReLU(), \
              nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_fea, kernel_size=3, padding=1), \
              nn.ReLU(), \
              nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_in, kernel_size=3, padding=1)])

        self.trans_F_x = nn.Sequential(
            *[nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_fea, kernel_size=3, padding=1), \
              nn.ReLU(), \
              nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_fea, kernel_size=3, padding=1), \
              nn.ReLU(), \
              nn.Conv2d(in_channels=self.channel_fea, out_channels=self.channel_in, kernel_size=3, padding=1)])

        ## proximal networks
        basic_block_x = functools.partial(ResidualBlock_noBN, nf=self.channel_fea)
        self.prox_x = make_layer(basic_block_x, 12)
        basic_block_s = functools.partial(ResidualBlock_noBN, nf=self.channel_fea)
        self.prox_s = make_layer(basic_block_s, 12)
        basic_block_d = functools.partial(ResidualBlock_noBN, nf=self.channel_fea)
        self.prox_d = make_layer(basic_block_d, 12)
        basic_block_img = functools.partial(ResidualBlock_noBN, nf=self.channel_fea)
        self.prox_k_img = make_layer(basic_block_img, 12)
        basic_block_real = functools.partial(ResidualBlock_noBN, nf=self.channel_fea)
        self.prox_k_real = make_layer(basic_block_real, 12)

        ## hyper-parameters
        self.mu_x = [nn.Parameter(torch.tensor(1.0)) for _ in range(self.iter_num)]
        # nn.Parameter 把张量标记为可学习的参数
        self.mu_k = [nn.Parameter(torch.tensor(1.0)) for _ in range(self.iter_num)]
        self.mu_s = [nn.Parameter(torch.tensor(1.0)) for _ in range(self.iter_num)]
        self.mu_d = [nn.Parameter(torch.tensor(1.0)) for _ in range(self.iter_num)]
        self.alpha = [nn.Parameter(torch.tensor(1.0)) for _ in range(self.iter_num)]
        self.beta = [nn.Parameter(torch.tensor(1.0)) for _ in range(self.iter_num)]
        self.gamma = [nn.Parameter(torch.tensor(0.25)) for _ in range(self.iter_num)]
        self.epsilon_k = [nn.Parameter(torch.tensor(0.25)) for _ in range(self.iter_num)]
        self.epsilon_s= [nn.Parameter(torch.tensor(0.25)) for _ in range(self.iter_num)]
        self.epsilon_d= [nn.Parameter(torch.tensor(0.25)) for _ in range(self.iter_num)]

        ## Weighted Average Layer (WAL): map C_i images to single image
        self.WAL = nn.Conv2d(in_channels=self.channel_in * 2, out_channels=self.image_channel, kernel_size=3, padding=1)

    def data_consistency_layer(self, generated, X_k, mask):

        gene_complex = real_to_complex(generated)
        output_complex = X_k + gene_complex * (1.0 - mask)
        output_img = complex_to_real(output_complex)

        return output_img

    def forward(self, x, y, y_diffusion=None, mask=None):

        # x: target image (b,1,h,w)
        # y: fully-sample reference image (b,1,h,w)
        '''
        # predifined mask with size (1,2,h,w)
        # if mask == None, x would be fully sampled target image and we should learn mask using Loupe.
        The undersampled target image will generate using the learned mask.
        # if mask != None, we use the predefined mask.
        x would be undersampled target image using the predefined mask.
        '''
        m = 1
        # CE
        x = x.repeat(1, self.channel_in, 1, 1)
        y1 = y # y1 为全采样参考图像
        y = y.repeat(1, self.channel_in, 1, 1)

        # 把x,y的通道都变为32

        # for Loupe, generate mask and get the corresponding under-sampled image
        if mask == None:
            m = None
            b, c, h, w = x.shape
            x_k = real_to_complex(x)
            ## generate mask
            mask = self.LoupeLayer(y1,y_diffusion) #mask的shape为(B,1,H,W)

            # Undersample
            x_k_under = x_k * mask
            # iFFT into image space
            x_under = complex_to_real(x_k_under)

        # for predefined mask
        else:
            mask = mask.repeat(1, self.channel_in, 1, 1)
            x_under = x
            x_k_under = real_to_complex(x)

        # initialize variables
        x_t0 = self.init_x(x_under)
        #x_tn = x_t0
        # 10个He初始化的残差块网络
        K_t0 = real_to_complex(x_t0)
        K_tn = K_t0
        # 对x0进行傅里叶变换

        # extract feature for reference
        y = self.map_y(y)


        y_dcn = self.dcn_align(x_t0, y)
        S_t0 = self.init_s(y_dcn)
        S_tn = S_t0
        D_t0 = y_dcn - S_t0
        D_tn = D_t0




        # 10个He初始化的残差块网络

        for i in range(self.iter_num):

            ########### update k ##########

            temp_k = K_t0*(1+self.epsilon_k[i]) - self.epsilon_k[i]*K_tn - self.mu_k[i] * (self.alpha[i] * (K_t0 - real_to_complex(x_t0)))

            K_t1 = torch.cat([self.prox_k_real(temp_k[:, :32]), self.prox_k_img(temp_k[:, 32:])], 1)
            # 第二个维度的前面32个为实数，后面为虚数 prox_k_real为12个残差块 prox_k_img为12个残差块


            ########## update x ###########
            # DC layer
            DC_x = self.data_consistency_layer(x_t0, x_k_under, mask)  ## DC
            # x_t0为图像经过初始化的结果 ，x_k_under为经过掩码的k空间数据，mask为掩码
            # 如果mask为1的话(原来采样到了)，那么新生成的为0 如果mask=0，原来没有采样到，那么新生成的被应用
            # refine image
            refine_image = -self.beta[i] * self.trans_D_x(
                self.trans_E_x(x_t0) - self.trans_E_y(S_t0))

            # refine k-space
            refine_k = complex_to_real(K_t0) - x_t0

            # proximal net for x
            x_t1 = self.prox_x(DC_x + self.mu_x[i] * (refine_image + refine_k))  # update x


            ########## update S and D ###########
            temp_s = S_t0*(1-self.mu_s[i]*self.gamma[i]+self.epsilon_s[i])-self.epsilon_s[i]*S_tn+self.mu_s[i]*self.gamma[i]*(self.dcn_align(x_t0,y)-D_t0)
            temp_s1 =self.trans_F_x(self.trans_D_x(self.trans_E_x(x_t0) - self.trans_E_y(S_t0)))*self.mu_s[i]*self.beta[i]
            S_t1 = self.prox_s(temp_s+temp_s1)


            ########## update D ###########
            temp_d = D_t0*(1-self.mu_d[i]*self.gamma[i]+self.epsilon_d[i])-self.epsilon_d[i]*D_tn+self.mu_d[i]*self.gamma[i]*(self.dcn_align(x_t0,y)-S_t0)
            D_t1 = self.prox_d(temp_d)

            # extract deep feature for y
            y = self.extract_y(y)

            K_tn = K_t0
            K_t0 = K_t1

            x_t0 = x_t1

            S_tn=S_t0
            S_t0=S_t1

            D_tn=D_t0
            D_t0=D_t1

        # reconstruct final image
        self.pmask = self.LoupeLayer.pmask
        x_out = self.WAL(torch.cat([x_t0, complex_to_real(K_t0)], 1))

        # return x_out, mask
        if m == None:
            return x_out, self.pmask
        else:
            return x_out
