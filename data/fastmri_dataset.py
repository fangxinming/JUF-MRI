import csv
import os

import logging
import pickle
import random
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn
import pathlib

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
from transforms import build_transforms
from matplotlib import pyplot as plt
import cv2


def fetch_dir(key, data_config_file=pathlib.Path("fastmri_dirs.yaml")):
    """
    Data directory fetcher.
    This is a brute-force simple way to configure data directories for a
    project. Simply overwrite the variables for `knee_path` and `brain_path`
    and this function will retrieve the requested subsplit of the data for use.
    Args:
        key (str): key to retrieve path from data_config_file.
        data_config_file (pathlib.Path,
            default=pathlib.Path("fastmri_dirs.yaml")): Default path config
            file.
    Returns:
        pathlib.Path: The path to the specified directory.
    """
    if not data_config_file.is_file():
        default_config = dict(
            knee_path="/home/jc3/Data/",
            brain_path="/home/jc3/Data/",
        )
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        raise ValueError(f"Please populate {data_config_file} with directory paths.")

    with open(data_config_file, "r") as f:
        data_dir = yaml.safe_load(f)[key]

    data_dir = pathlib.Path(data_dir)

    if not data_dir.exists():
        raise ValueError(f"Path {data_dir} from {data_config_file} does not exist.")

    return data_dir


def et_query(
        root: etree.Element,  #root XML文档的根元素et_root
        qlist: Sequence[str],  #一个字符串列表，表示要查询的嵌套路径
        # 例如["Encoding","matrixSzie"] 表示查询路径为./Encoding/MatrixSize
        namespace: str = "http://www.ismrm.org/ISMRMRD",
        # XML文档的命名空间
) -> str:
    """
    ElementTree query function.
    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.
    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.
    Returns:
        The retrieved data as a string.
    """
    #  根据给定的路径(qlist)在XML文档中查找指定的元素，并返回该元素的文本内容
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)



### This part was built based on MTrans: https://github.com/chunmeifeng/MTrans
class SliceDataset(Dataset):
    def __init__(self, opt, train):
        # 此处的opt为datasets:分为train和val
        if train:
            self.mode = 'train'
        else:
            self.mode = 'val'

        sample_rate = 1
        self.crop_size = opt['crop_size']
        self.scale = int(opt['scale'])
        self.hr_in = opt['hr_in']  #self.hr_in=True
        self.task = opt['task']

        challenge = 'singlecoil'
        # challenge
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )

        #self.recons_key = "reconstruction_esc"

        # transform
        self.transform = build_transforms(self.mode, self.hr_in)

        #   build_transforms 返回 DataTransform( 'singlecoil' , True)
        #   DataTransform是一个类所以self.transform是一个尚未实例化的类


        self.examples = []

        self.cur_path = '/home/xinming/MCDudo/dataset/fastmri/knee_singlecoil'


        self.csv_file = os.path.join(self.cur_path, "singlecoil_" + self.mode + "_split_less.csv")

        # /home/xinming/MCDudo/dataset/fastmri/knee_singlecoil/singlecoil_train_split_less.csv

        #/ home/xinming/MCDudo/dataset/fastmri/knee_singlecoil/singlecoil_val_split_less.csv


        self.h5_file_path = os.path.join(self.cur_path, "singlecoil_" + self.mode)
        #/home/xinming/MCDudo/dataset/fastmri/knee_singlecoil/singlecoil_train
        # 训练集.h5文件夹路径
        #/home/xinming/MCDudo/dataset/fastmri/knee_singlecoil/singlecoil_val
        # 验证集.h5文件夹路径

        # 读取CSV
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)
            #逐行读取csv文件
            id = 0
            for row in reader:   #遍历csv文件的每一行
                pd_metadata, pd_num_slices = self._retrieve_metadata(os.path.join(self.h5_file_path, row[0] + '.h5'))
                #/home/xinming/MCDudo/dataset/fastmri/knee_singlecoil/singlecoil_train/row[0].h5
                # metadata 里面有 padding_left padding_right encoding_size recon_size
                # pd_num_slices 为切片数量

                pdfs_metadata, pdfs_num_slices = self._retrieve_metadata(os.path.join(self.h5_file_path, row[1] + '.h5'))

                for slice_id in range(min(pd_num_slices, pdfs_num_slices)):
                    self.examples.append(
                        (os.path.join(self.h5_file_path, row[0] + '.h5'), os.path.join(self.h5_file_path, row[1] + '.h5')
                         , slice_id, pd_metadata, pdfs_metadata, id))
                    # /home/xinming/MCDudo/dataset/fastmri/knee_singlecoil/singlecoil_trainrow[0].h5
                    # /home/xinming/MCDudo/dataset/fastmri/knee_singlecoil/singlecoil_trainrow[1].h5

                #列表中的每一项都是一个元组(pd文件路径，pdfs文件路径，所在切片的id，pd元数据，pdfs元数据，总id)

                id += 1

        if sample_rate < 1:
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[0:num_examples]

        self.mask_path = "/home/xinming/MCDudo/dataset/fastmri/mask/mask_equ_x4.png"

    def __len__(self):
        if self.mode == 'train':
            return len(self.examples)
        else:
            return len(self.examples)

    def __getitem__(self, i):

        # read pd
        pd_fname, pdfs_fname, slice, pd_metadata, pdfs_metadata, id = self.examples[i]

        with h5py.File(pd_fname, "r") as hf:
            pd_kspace = hf["kspace"][slice]
            #pd_kspace 为pd的k空间数据


            # read_mask
            pd_mask = cv2.imread(self.mask_path, cv2.IMREAD_UNCHANGED)
            pd_mask = torch.tensor(pd_mask).unsqueeze(-1).repeat(1,1,2)/255

            pd_target = hf[self.recons_key][slice] if self.recons_key in hf else None
            # self.recons_key = "reconstruction_esc"

            attrs = dict(hf.attrs)

            attrs.update(pd_metadata)

        pd_sample = self.transform(pd_kspace, pd_mask, pd_target, attrs, pd_fname, slice)

        #   self.transform = build_transforms(self.mode, self.hr_in)
        #   build_transforms 返回 DataTransform( 'singlecoil' , True)
        #   DataTransform是一个类所以self.transform是一个尚未实例化的类

        with h5py.File(pdfs_fname, "r") as hf:
            # read_data
            pdfs_kspace = hf["kspace"][slice]
            pdfs_target = hf[self.recons_key][slice] if self.recons_key in hf else None
            
            # read_mask
            pdfs_mask = cv2.imread(self.mask_path, cv2.IMREAD_UNCHANGED)
            pdfs_mask = torch.tensor(pdfs_mask).unsqueeze(-1).repeat(1,1,2)/255

            attrs = dict(hf.attrs)

            attrs.update(pdfs_metadata)
            
        pdfs_sample = self.transform(pdfs_kspace, pdfs_mask, pdfs_target, attrs, pdfs_fname, slice)

        pd_image, pd_target, pd_mean, pd_std, pd_fname, pd_slice_num = pd_sample
        pdfs_image, pdfs_target, pdfs_mean, pdfs_std, pdfs_fname, pdfs_slice_num = pdfs_sample

        # return pd_sample, pdfs_sample, id
        
        pd_image = pd_image.unsqueeze(0)
        pd_target = pd_target.unsqueeze(0)
        pdfs_image = pdfs_image.unsqueeze(0) 
        pdfs_target = pdfs_target.unsqueeze(0)

        return {'im1_LQ':pdfs_image, 'im1_GT':pdfs_target, 'im2_LQ':pd_image, 'im2_GT':pd_target, 'mean_1':pdfs_mean, 'mean_2':pd_mean, 'std_1':pdfs_std, 'std_2':pd_std, 'mask':pdfs_mask}

    def _retrieve_metadata(self, fname):
        # /home/xinming/MCDudo/dataset/fastmri/knee_singlecoil/singlecoil_train/row[0].h5
        # 输入的为训练集/验证集的.h5文件

        with h5py.File(fname, "r") as hf:
            et_root = etree.fromstring(hf["ismrmrd_header"][()])
            # hf["ismrmrd_header"][()] 读取数据集中的全部信息
            # etree.fromstring 把XML格式的字符串解析为一个ElementTree对象
            # et_root 是XML文档的根元素可以通过它访问和操作XML数据

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max

            num_slices = hf["kspace"].shape[0]

        metadata = {
            "padding_left": padding_left,
            "padding_right": padding_right,
            "encoding_size": enc_size,
            "recon_size": recon_size,
        }

        return metadata, num_slices


def to_image(data):
    image = data.numpy()
    max_image = np.max(image)
    min_image = np.min(image)
    gap = max_image - min_image
    image = (image - min_image)/gap*255.
    return image

def vis_img(img, fname, ftype ,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.imshow(img, cmap='gray')
    figname = fname + '_' + ftype + '.png'
    figpath = os.path.join(output_dir, figname)
    plt.savefig(figpath)

dataset_opt={}

mode = 'Brats2018'
dataset_opt['crop_size'] = 256
dataset_opt['scale'] = 4
dataset_opt['hr_in'] = True
dataset_opt['task'] = 'rec'


data = SliceDataset(dataset_opt, True)
for i,data_item in enumerate(data):
    pd_sample, pdfs_sample, id = data_item
    pd_image, pd_target, pd_mean, pd_std, pd_fname, pd_slice_num = pd_sample
    pdfs_image, pdfs_target, pdfs_mean, pdfs_std, pdfs_fname, pdfs_slice_num = pdfs_sample

    pdfs_image = pdfs_image * pdfs_std + pdfs_mean
    pdfs_target = pdfs_target * pdfs_std + pdfs_mean

    pd_image = pd_image * pd_std + pd_mean
    pd_target = pd_target * pd_std + pd_mean

    # print(pd_image.shape, pd_target.shape)

    # vis_img(pdfs_image, str(i), 'pdfs_lr', 'show_pdrs')
    # vis_img(pdfs_target, str(i), 'pdfs_target', 'show_pdrs')
    # vis_img(pd_image, str(i), 'pd_lr', 'show_pd')
    # vis_img(pd_target, str(i), 'pd_target', 'show_pd')
    # # print(pd_mask.shape)
    cv2.imwrite('./pd_lr'+'/{:08d}.png'.format(i), to_image(pdfs_image))
    cv2.imwrite('./pd_tar'+'/{:08d}.png'.format(i), to_image(pdfs_target))
    print(i)
    