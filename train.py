import os
import math
import argparse
import random
import logging
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler
from PIL import Image
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from data.mask_exact import squash_mask,sparsify,sigmoid_beta

#from data.diffusion_mask_exact import squash_mask,sparsify,sigmoid_beta

import data.util as data_util
import cv2
import numpy as np
import skimage.metrics as sm
import csv
from tqdm import tqdm




def main():
    #### options
    parser = argparse.ArgumentParser()
    #创建一个命令行参数解析器的对象 parser
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    #YAML文件路径
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    # 解析所有的命令行参数，并将结果存储在 args 对象中
    opt = option.parse(args.opt, is_train=True)
    rank = -1

    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    if resume_state is None:
        util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
    #创建文件夹并且重命名
        util.mkdirs((path for key, path in opt['path'].items()
                 if not key == 'experiments_root'
                    and 'pretrain_model' not in key
                 and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))
        # tensorboard logger
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        from torch.utils.tensorboard import SummaryWriter
        tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    opt = option.dict_to_nonedict(opt)
    which_model = opt['mode']
    if opt['mode'] == "fastmri":
        which_model = "fastmri/knee_singlecoil"
    ##random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)

    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True
    #对卷积操作进行性能优化
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 1.0 # dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt, train=True)
            #创建数据集，为类实例化出的对象
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size'])) # Iter number per epoch
            #math.ceil(x)为返回不小于x的最小整数，对x向上取整，train_size指的是训练一个epoch所需迭代的次数
            total_epochs = int(opt['epoch'])
            #total_iters = int(opt['train']['niter'])
            #200000 总共迭代的次数
            total_iters= train_size*total_epochs
            #total_epochs = int(math.ceil(total_iters / train_size))
            opt['logger']['save_checkpoint_freq']=int(total_iters/total_epochs)
            opt["train"]["val_freq"]= int(total_iters/total_epochs)

            train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if opt['model'] == 'joint-optimize' and opt['method'] == 'diffusion':
                dataset_opt1= dataset_opt
                dataset_opt1['phase'] = 'val'
                train_set1= create_dataset(dataset_opt1, train=False)
                train_loader1 = create_dataloader(train_set1, dataset_opt1, opt, train_sampler)
            #实例化一个DataLoader对象
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        
        elif phase == 'val':
            val_set = create_dataset(dataset_opt, train=False)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            logger.info('Number of val images: {:d}'.format(len(val_set)))

        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None
    assert val_loader is not None

    #### create model
    model = create_model(opt)
    #model为模型类


    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch'] + 1
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    performance = []
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs):
        for _, train_data in enumerate(train_loader):
            #enumerate 把可迭代对象的索引与元素关联起来
            current_step += 1
            #if current_step > total_iters:
                #break
                        #### training
            model.feed_data(train_data)
            model.optimize_parameters(epoch, total_epochs)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                #打印频率
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')>'
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar(k, v, current_step)
                logger.info(message)
            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states at the end of step: ' + str(current_step))
                model.save(current_step)
                model.save_training_state(epoch, current_step)

            #### update learning rate
            model.update_learning_rate(current_step+1, warmup_iter=opt['train']['warmup_iter'])
            #### validation
            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0:
                avg_psnr_im1 = 0.0
                avg_ssim_im1 = 0.0
                avg_rmse_im1 = 0.0
                idx = 0
                if opt['mode'] == "IXI":
                    idx1 = (current_step / opt["train"]["val_freq"] - 1) * 1140
                elif opt['mode'] == "Brats2018":
                    idx1 = (current_step / opt["train"]["val_freq"] - 1) * 560
                elif opt['mode'] == "fastmri":
                    idx1 = (current_step / opt["train"]["val_freq"] - 1) * 544
                else:
                    print("wrong dataset")



                for val_data in tqdm(val_loader): 
                    model.feed_data(val_data)
                    model.test()
                    # 给model中的self.fake_H，self.mask赋值
                    visuals = model.get_current_visuals()
                    # visuals为一个字典，包含重建图像，原始图像与mask
                    img_num = visuals["im1_GT"].shape[0]
                    # print(visuals["im1_restore"].shape, visuals["im1_GT"].shape)
                    for i in range(img_num):
                        sr_img_1 = visuals["im1_restore"][i, 0, :, :]  # (1, w, h)
                        gt_img_1 = visuals["im1_GT"][i, 0, :, :]  # (1, w, h)
                        gt_img_2 = visuals["im2_GT"][i, 0, :, :]
                        mask_1=visuals["mask"][i, 0, :, :]
                        lr_img_1 = visuals["im1_lr"][i, 0, :, :]
                        if opt['model_G'] in ['edgeNet','edgeNet_v2','edgeNet_v3','edgeNet_v4','edgeNet_v5','edgeNet_v6','edgeNet_v7','edgeNet_v8','edgeNet_v9','edgeNet_v10']:
                            lr_edge_1 = visuals["high_x"][i, 0, :, :]
                            flag_gt=gt_img_1
                            flag_ref=gt_img_2
                            gt_edge_1 = util.compute_soft_edge_tensor(flag_gt.unsqueeze(0).unsqueeze(0))
                            ref_edge_1 = util.compute_soft_edge_tensor(flag_ref.unsqueeze(0).unsqueeze(0))
                            tb_logger.add_images(opt['model_G'] + "_" + opt['undersample'] + '_' + str(
                                opt['scale']) + 'x_'+'soft_edge',
                                                 torch.tensor(lr_edge_1.numpy() * 255).unsqueeze(0).unsqueeze(0) / 255,
                                                 idx1)
                            tb_logger.add_images(opt['model_G'] + "_" + opt['undersample'] + '_' + str(
                                opt['scale']) + 'x_' + 'gt_soft_edge',
                                                 torch.tensor(gt_edge_1.numpy() * 255) / 255,
                                                 idx1)

                            tb_logger.add_images(opt['model_G'] + "_" + opt['undersample'] + '_' + str(
                                opt['scale']) + 'x_' + 'ref_soft_edge',
                                                 torch.tensor(ref_edge_1.numpy() * 255) / 255,
                                                 idx1)

                                    
                        # calculate PSNR
                        # if which_model == 'IXI' or which_model == 'Brats2018':
                        cur_psnr_im1 = util.calculate_psnr(sr_img_1.numpy()*255., gt_img_1.numpy()*255.)
                        cur_ssim_im1 = util.calculate_ssim(sr_img_1.numpy() * 255., gt_img_1.numpy() * 255.)
                        cur_rmse_im1 = util.calculate_rmse(sr_img_1.numpy() * 255., gt_img_1.numpy() * 255.)
                        # elif which_model == 'fastmri':
                        #     cur_psnr_im1 = util.calculate_psnr_fastmri(gt_img_1.numpy(), sr_img_1.numpy())
                        avg_psnr_im1 += cur_psnr_im1
                        avg_ssim_im1 += cur_ssim_im1
                        avg_rmse_im1 += cur_rmse_im1



                        tb_logger.add_images(opt['model_G'] + "_" + opt['undersample'] + '_' + str(
                            opt['scale']) + 'x', torch.tensor(sr_img_1.numpy()*255).unsqueeze(0).unsqueeze(0) / 255, idx1)
                        tb_logger.add_images(opt['model_G'] + "_" + opt['undersample'] + '_' + str(
                            opt['scale']) + 'x_'+'mask', torch.tensor(mask_1.numpy() * 255).unsqueeze(0).unsqueeze(0) / 255,
                                             idx1)
                        tb_logger.add_images(opt['model_G'] + "_" + opt['undersample'] + '_' + str(
                            opt['scale']) + 'x_'+'lr', torch.tensor(lr_img_1.numpy() * 255).unsqueeze(0).unsqueeze(0) / 255,
                                             idx1)
                        tb_logger.add_images("GT", torch.tensor(gt_img_1.numpy()*255).unsqueeze(0).unsqueeze(0) / 255, idx1)
                        tb_logger.add_images("Ref", torch.tensor(gt_img_2.numpy() * 255).unsqueeze(0).unsqueeze(0) / 255, idx1)
                        idx+=1
                        idx1+=1

                avg_psnr_im1 = avg_psnr_im1 / idx
                avg_ssim_im1 = avg_ssim_im1 /idx
                avg_rmse_im1 = avg_rmse_im1 /idx
                tb_logger.add_scalar('psnr',avg_psnr_im1,int(current_step/1000))
                tb_logger.add_scalar('ssim', avg_ssim_im1, int(current_step / 1000))
                tb_logger.add_scalar('rmse', avg_rmse_im1, int(current_step / 1000))
                # log

                logger.info("# image1 Validation # PSNR: {:.6f}".format(avg_psnr_im1))
                logger.info("# image1 Validation # SSIM: {:.6f}".format(avg_ssim_im1))
                logger.info("# image1 Validation # RMSE: {:.6f}".format(avg_rmse_im1))
                performance.append(avg_psnr_im1)



    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')
    # 假设 performance 是一个列表
    top3 = sorted(enumerate(performance), key=lambda x: x[1], reverse=True)[:3]

    for rank, (idx, val) in enumerate(top3, start=1):
        logger.info(f"Top {rank} PSNR: {val}, at index: {idx + 1}")

    #opt['name'] = opt['mode'] + '_' + opt['model_G'] + '_' + opt['undersample'] + '_x' + str(opt['scale']) + model
    # 名字 为：数据集_网络名称_亚采样方式_x倍数_任务
    if opt['model']=='joint-optimize' and opt['method'] in ['Loupe','mean']:
        model_path = "/home/xinming/MCDudo/experiments/"+opt['name']+"/models/Iter_latest.pth"
        model_learned = torch.load(model_path)
        pmask_value = model_learned['LoupeLayer.pmask']
        if opt['method']=='Loupe':
            probmask = squash_mask(mask=pmask_value)  # 初始化并把数据映射到[0.01.0.99]之间
        elif opt['method']=='mean':
            probmask = pmask_value
        else:
            raise ValueError("亚采样方法错误！")
        # Sparsify, control the sampling ratio
        sparse_mask = sparsify(mask=probmask,sparsity=1.0/opt['scale'])  # 把整个稀疏度变化为要求的稀疏度  整个均值等于sparsity
        # generate soft mask
        mask = sigmoid_beta(mask=sparse_mask,image_dims=opt['image_size'])
        mask_soft = mask.data.cpu().detach().numpy() * 255.
        mask_hard = torch.where(mask >= 0.5, 1, 0)
        mask_soft = mask_soft.squeeze()
        mask_real = mask_hard.squeeze().data.cpu().detach().numpy() * 255.0
        # mask_soft_path = '/home/xinming/MCDudo/dataset/IXI/'+opt['model_G']+'_mask/x' + str(opt['scale']) + '_mask_soft_IXI.png'
        # mask_path = '/home/xinming/MCDudo/dataset/IXI/'+opt['model_G']+'_mask/x' + str(opt['scale']) + '_mask_hard_IXI.png'
        # mask_file_path= '/home/xinming/MCDudo/dataset/IXI/'+opt['model_G']+'_mask'

        mask_soft_path = '/home/xinming/MCDudo/dataset/'+ which_model +'/' + opt['model_G'] + '_mask/x' + str(opt['scale']) + '_mask_soft_'+ opt['mode'] +'.png'
        mask_path = '/home/xinming/MCDudo/dataset/' + which_model +'/' + opt['model_G'] + '_mask/x' + str(opt['scale']) + '_mask_hard_'+ opt['mode'] +'.png'
        mask_file_path = '/home/xinming/MCDudo/dataset/'+ which_model+'/' + opt['model_G'] + '_mask'


        util.mkdir(mask_file_path)
        cv2.imwrite(mask_soft_path, mask_soft)
        cv2.imwrite(mask_path, mask_real)
        logger.info('Loupe Mask saved successfully.')
        print(mask_path)
    elif opt['model']=='joint-optimize' and opt['method']=='diffusion':
        save_file_path = '/home/xinming/MCDudo/dataset/'+which_model+'/train/'+opt['name']+'softmask_x' + str(opt['scale'])
        save_file_path1 = '/home/xinming/MCDudo/dataset/'+which_model+'/train/'+opt['name']+'hardmask_x' + str(opt['scale'])


        # save_file_path ='/home/xinming/MCDudo/dataset/IXI/train/smalldiffusionsoftmask_x'+ str(opt['scale'])
        # save_file_path1 = '/home/xinming/MCDudo/dataset/IXI/train/smalldiffusionhardmask_x'+ str(opt['scale'])
        totle_number=0
        for train_data in tqdm(train_loader1):
            model.feed_data(train_data)
            model.test()
            visuals = model.get_current_visuals()
            # visuals为一个字典，包含重建图像，原始图像与mask
            img_num = visuals["im1_GT"].shape[0]
            # print(visuals["im1_restore"].shape, visuals["im1_GT"].shape)
            for i in range(img_num):
                mask = visuals["mask"][i, 0, :, :]
                mask_soft = mask.data.cpu().detach().numpy() * 255.
                mask_hard = torch.where(mask >= 0.5, 1, 0)
                mask_soft = mask_soft.squeeze()
                mask_real = mask_hard.squeeze().data.cpu().detach().numpy() * 255.0
                totle_number=totle_number+1
                util.mkdir(save_file_path)
                util.mkdir(save_file_path1)
                cv2.imwrite(save_file_path + '/{:08d}.png'.format(totle_number), mask_soft)
                cv2.imwrite(save_file_path1 + '/{:08d}.png'.format(totle_number), mask_real)
        logger.info('Diffusion train Mask saved successfully.')


        folder_path = save_file_path1  # 替换为你的文件夹路径


        mask_path = '/home/xinming/MCDudo/dataset/'+ which_model +'/' + opt['model_G'] + '_mask/x' + str(
                opt['scale']) + '_mask_soft_IXI.png'
        sum_matrix = None
        image_count = 0


            # 遍历文件夹中的所有图像文件
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # 支持的图像格式
                image_path = os.path.join(folder_path, filename)

                # 打开图像并转换为灰度图像
                img = Image.open(image_path).convert("L")
                img_array = np.array(img, dtype=np.float32)  # 转换为浮点数类型矩阵

                    # 累加图像矩阵
                if sum_matrix is None:
                    sum_matrix = img_array
                else:
                    sum_matrix += img_array

                image_count += 1

            # 计算均值矩阵
        if image_count > 0:
            mean_matrix = sum_matrix / image_count
            mean_image = Image.fromarray(np.uint8(mean_matrix))  # 转换为灰度图像格式

                # 保存或显示均值图像
            cv2.imwrite(mask_path,mean_image)
            logger.info('Diffusion train Mask saved successfully.')
        else:
            print("文件夹中没有有效的图像文件。")

        image_path = mask_path # 替换为你的图像路径

            # 打开图像并转换为灰度图像
        img = Image.open(image_path).convert("L")

            # 转换为 NumPy 数组
        gray_matrix = np.array(img)

            # 设置二值化阈值
        threshold = 128  # 阈值（0-255之间，可调整）

            # 执行二值化操作
        binary_matrix = (gray_matrix > threshold).astype(np.uint8) * 255

            # 转换为图像格式
        binary_image = Image.fromarray(binary_matrix)

        mask_path = '/home/xinming/MCDudo/dataset/'+which_model+'/' + opt['model_G'] + '_mask/x' + str(
                opt['scale']) + '_mask_hard_IXI.png'
        cv2.imwrite(mask_path , binary_image)

            # 保存和显示二值化结果




        save_file_path = '/home/xinming/MCDudo/dataset/'+which_model+'/val/smalldiffusionsoftmask_x' + str(opt['scale'])
        save_file_path1 = '/home/xinming/MCDudo/dataset/'+which_model+'/val/smalldiffusionhardmask_x' + str(opt['scale'])
        totle_number=0
        for val_data in tqdm(val_loader):
            model.feed_data(val_data)
            model.test()
            visuals = model.get_current_visuals()
            # visuals为一个字典，包含重建图像，原始图像与mask
            img_num = visuals["im1_GT"].shape[0]
            # print(visuals["im1_restore"].shape, visuals["im1_GT"].shape)
            for i in range(img_num):
                mask = visuals["mask"][i, 0, :, :]
                mask_soft = mask.data.cpu().detach().numpy() * 255.
                mask_hard = torch.where(mask >= 0.5, 1, 0)
                mask_soft = mask_soft.squeeze()
                mask_real = mask_hard.squeeze().data.cpu().detach().numpy() * 255.0
                totle_number=totle_number+1
                util.mkdir(save_file_path)
                util.mkdir(save_file_path1)
                cv2.imwrite(save_file_path + '/{:08d}.png'.format(totle_number), mask_soft)
                cv2.imwrite(save_file_path1 + '/{:08d}.png'.format(totle_number), mask_real)
            logger.info('Diffusion val Mask saved successfully.')
        else:
            logger.info('No need to save the mask.')




if __name__ == '__main__':
    main()
