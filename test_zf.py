import cv2,os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import util
from tqdm import tqdm



dataset_opt = {}
dataset_opt['task'] = 'rec'

dataset_opt['undersample'] = 'equ'
dataset_opt['scale'] = 30


dataset_opt['hr_in'] = True
dataset_opt['crop_size'] = 0
dataset_opt['mode'] = 'IXI' #'IXI' , 'brain=Brats2018' , 'fastmri'
dataset_opt['model']= 'ref-rec'    #[ 'ref-rec']
dataset_opt['model_G']= None
dataset_opt['method'] = None
dataset_opt['network'] = 'zero_filling'



model='_rec'
mode = dataset_opt['mode']
name=dataset_opt['network']+'_'+dataset_opt['undersample']+'_'+'x'+str(dataset_opt['scale'])+model
writer = SummaryWriter(log_dir='../tb_logger/'+dataset_opt['mode']+'_'+name)
#### create train and val dataloader
if mode == 'IXI':
    dataset_opt['test_size'] = 256
    dataset_opt['image_size'] = [256, 256]
    from data.IXI_dataset import IXI_train as D

    # D为数据集类
    dataset_opt['dataroot_GT'] = '/home/xinming/MCDudo/dataset/IXI/test/smallT2PNG'
elif mode == 'Brats2018':
    dataset_opt['test_size'] = 240
    dataset_opt['image_size'] = [240, 240]
    from data.brain_dataset import brain_train as D

    dataset_opt['dataroot_GT'] = '/home/xinming/MCDudo/dataset/Brats2018/test/smallT2PNG'
elif mode == 'fastmri':
    dataset_opt['test_size'] = 256
    dataset_opt['image_size'] = [256, 256]
    from data.knee_dataset import knee_train as D

    dataset_opt['dataroot_GT'] = '/home/xinming/MCDudo/dataset/fastmri/knee_singlecoil/test/PDFSPNG'

val_set = D(dataset_opt, train=False)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)
print('Number of val images: {:d}'.format(len(val_set)))
name=dataset_opt['mode']+'_'+dataset_opt['network']+'_'+dataset_opt['undersample']+'_'+'x'+str(dataset_opt['scale'])+model
save_path = '/home/xinming/MCDudo/test/test_'+name

avg_psnr_im1 = 0.0
avg_ssim_im1 = 0.0
avg_rmse_im1 = 0.0
idx = 0
for i, val_data in enumerate(tqdm(val_loader)):
    im1_lr = val_data['im1_LQ']
    im1_gt = val_data['im1_GT']
    im2_lr = val_data['im2_LQ']
    im2_gt = val_data['im2_GT']
    mask = val_data['mask']
    for batch_idx in range(im1_lr.size(0)):
        im1_lr_batch = im1_lr[batch_idx:batch_idx + 1]  # 选择 batch 中的单个图像
        im1_gt_batch = im1_gt[batch_idx:batch_idx + 1]
        im2_lr_batch = im2_lr[batch_idx:batch_idx + 1]
        im2_gt_batch = im2_gt[batch_idx:batch_idx + 1]
        mask_batch = mask[batch_idx:batch_idx + 1]
        im1_lr_1 = im1_lr_batch[0,0].cpu().detach().numpy()*255
        im1_gt_1 = im1_gt_batch[0, 0].cpu().detach().numpy() * 255
        mask_batch = mask_batch[0, 0].cpu().detach().numpy() * 255
        writer.add_images("LQ_"+dataset_opt['undersample']+'_'+str(dataset_opt['scale']),torch.tensor(im1_lr_1).unsqueeze(0).unsqueeze(0)/255,idx)
        writer.add_images("GT",torch.tensor(im1_gt_1).unsqueeze(0).unsqueeze(0)/255,idx)
        writer.add_images("mask", torch.tensor(mask_batch).unsqueeze(0).unsqueeze(0) / 255, idx)

        cur_psnr_im1 = util.calculate_psnr(im1_lr_1, im1_gt_1)
        avg_psnr_im1 += cur_psnr_im1
        cur_ssim_im1 = util.calculate_ssim(im1_lr_1, im1_gt_1)
        avg_ssim_im1 += cur_ssim_im1
        cur_rmse_im1 = util.calculate_rmse(im1_lr_1, im1_gt_1)
        avg_rmse_im1 += cur_rmse_im1
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(os.path.join(save_path, '{:08d}.png'.format(idx + 1)), im1_lr_1)

        idx += 1
    # calculate PSNR



avg_psnr_im1 = avg_psnr_im1 / idx
avg_ssim_im1 = avg_ssim_im1 / idx
avg_rmse_im1 = avg_rmse_im1 / idx
print(name)
print("# image1 Validation # PSNR: {:.6f}".format(avg_psnr_im1))
print("# image1 Validation # SSIM: {:.6f}".format(avg_ssim_im1))
print("# image1 Validation # RMSE: {:.6f}".format(avg_rmse_im1))

writer.close()