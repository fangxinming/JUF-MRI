import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import torch  # 现在导入 torch
from torch.utils.tensorboard import SummaryWriter
from utils import util
from tqdm import tqdm


#
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim



import time
import models.modules.JUF_MRI_Loupe as JUF_MRI_Loupe
import models.modules.JUF_MRI_diffusion as JUF_MRI_diffusion



def main():
    save_result= False
    dataset_opt = {}
    mode = 'IXI'  # "FastMRI" "Brats2018" "IXI"
    dataset_opt['scale'] = 30  # acceleration
    dataset_opt['undersample'] = 'learned'  # 'random' or 'equ' or 'learned' or 'radial' or 'spiral'
    dataset_opt['network'] = 'JUF_MRI_diffusion'  # 'JUF_MRI'  'JUF_MRI_diffusion'
    dataset_opt['mode'] = mode
    model = '_rec'

    dataset_opt['task'] = 'rec'
    dataset_opt['hr_in'] = True
    dataset_opt['crop_size'] = 0
    dataset_opt['model'] = 'ref-rec'
    dataset_opt['c_image'] = 1
    dataset_opt['nf'] = 32
    dataset_opt['stages'] = 4

    dataset_opt['sparsity'] = 1.0 / dataset_opt['scale']
    dataset_opt['model_G'] = dataset_opt['network']
    dataset_opt['method'] = dataset_opt['network'].split("_")[-1]

    name = dataset_opt['mode'] + '_' + dataset_opt['network'] + '_' + dataset_opt['undersample'] + '_' + 'x' + str(
        dataset_opt['scale']) + model
    print(name)

    if mode == 'IXI':
        dataset_opt['test_size'] = 256
        dataset_opt['image_size'] = [256, 256]
        from data.IXI_dataset import IXI_train as D
        # D为数据集类
        dataset_opt['dataroot_GT'] = '/home/xinming/MCDudo/dataset/IXI/test/smallT2PNG'
        #dataset_opt['dataroot_GT'] = '/home/xinming/JUF_MRI/JUF_MRI_dataset/IXI/test/T2PNG'
    elif mode == 'Brats2018':
        dataset_opt['test_size'] = 240
        dataset_opt['image_size'] = [240, 240]
        from data.brain_dataset import brain_train as D
        dataset_opt['dataroot_GT'] = '/home/xinming/JUF_MRI/JUF_MRI_dataset/Brats2018/test/T2PNG'
    elif mode == 'FastMRI':
        dataset_opt['test_size'] = 256
        dataset_opt['image_size'] = [256, 256]
        from data.knee_dataset import knee_train as D
        dataset_opt['dataroot_GT'] = '/home/xinming/JUF_MRI/JUF_MRI_dataset/FastMRI/test/PDFSPNG'

    val_set = D(dataset_opt, train=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False, num_workers=1, pin_memory=True)
    print('Number of val images: {:d}'.format(len(val_set)))

    if dataset_opt['undersample'] == 'learned':
        undersample = 'learned'
    else:
        undersample = 'fixed'

    model_path = (
        f"/home/xinming/JUF_MRI/Pre-trained_weights/"
        f"{mode}_Pre-trained_weights/{mode}_{undersample}_mask/"
        f"{mode}_{dataset_opt['network']}_{dataset_opt['undersample']}_x{dataset_opt['scale']}.pth"
    )
    print(model_path)

    if dataset_opt['network'] in ['JUF_MRI']:
        model = JUF_MRI_Loupe.JUF_MRI_Loupe(dataset_opt).cuda()
    elif dataset_opt['network'] in ['JUF_MRI_diffusion']:
        model = JUF_MRI_diffusion.JUF_MRI_diffusion(dataset_opt).cuda()



    model_params = util.get_model_total_params(model)
    print('Model_params: ', model_params)

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()

    with torch.no_grad():
        #### validation

        avg_psnr_im1 = 0.0
        avg_ssim_im1 = 0.0
        avg_rmse_im1 = 0.0
        idx = 0
        time_begin = time.time()
        for i, val_data in enumerate(tqdm(val_loader)):
            im1_lr = val_data['im1_LQ'].cuda()
            im1_gt = val_data['im1_GT'].cuda()
            im2_gt = val_data['im2_GT'].cuda()
            mask = val_data['mask'].cuda()

            for batch_idx in range(im1_lr.size(0)):
                im1_lr_batch = im1_lr[batch_idx:batch_idx + 1]  # 选择 batch 中的单个图像
                im1_gt_batch = im1_gt[batch_idx:batch_idx + 1]
                im2_gt_batch = im2_gt[batch_idx:batch_idx + 1]
                mask_batch = mask[batch_idx:batch_idx + 1]

                sr_img_1 = model(im1_lr_batch, im2_gt_batch, mask=mask_batch)
                sr_img_1 = torch.clamp(sr_img_1, min=0.0, max=1.0)

                sr_img_1 = sr_img_1[0, 0].cpu().detach().numpy() * 255.
                im1_gt_batch = im1_gt_batch[0, 0].cpu().detach().numpy() * 255.


                cur_psnr_im1 = calculate_psnr(sr_img_1, im1_gt_batch,data_range=255)
                avg_psnr_im1 += cur_psnr_im1
                cur_ssim_im1 = calculate_ssim(sr_img_1, im1_gt_batch,data_range=255)
                avg_ssim_im1 += cur_ssim_im1
                cur_rmse_im1 = util.calculate_rmse(sr_img_1, im1_gt_batch)
                avg_rmse_im1 += cur_rmse_im1



                if save_result:
                    save_path = '/home/xinming/JUF_MRI/test/test_' + name
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    cv2.imwrite(os.path.join(save_path, '{:08d}.png'.format(idx + 1)), sr_img_1)

                idx += 1
                time_end = time.time()
        avg_psnr_im1 = avg_psnr_im1 / idx
        avg_ssim_im1 = avg_ssim_im1 / idx
        avg_rmse_im1 = avg_rmse_im1 / idx

        print("# image1 Validation # PSNR: {:.6f}".format(avg_psnr_im1))
        print("# image1 Validation # SSIM: {:.6f}".format(avg_ssim_im1))
        print("# image1 Validation # RMSE: {:.6f}".format(avg_rmse_im1))
        print('Total time:', time_end - time_begin)

if __name__ == '__main__':
    main()
