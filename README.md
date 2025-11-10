#  Frequency Error-Guided Under-sampling Optimization for Multi-Contrast MRI Reconstruction

## Environment
- Python==3.8.19
- scikit-image==0.21.0
- torch==1.7.1
- torchvision==0.8.2
- PyYAML==6.0.2
- h5py==3.11.0
- numpy==1.26.0

```
Compile DCNv2:      
cd $ROOT/models/modules/DCNv2
sh make.sh
```
For more implementation details about DCN, please see [DCNv2].


## Datasets
### Parpare Datasets for IXI and BrainTS:


The IXI and BraTS2018 can be downloaded at:  [[IXI dataset]](https://brain-development.org/ixi-dataset/)  and  [[BrainTS dataset]](http://www.braintumorsegmentation.org/).  
(1) The original data are  _**.nii**_  data. Split your data set into training sets, validation sets, and test sets;  
(2) Read  _**.nii**_  data and save these slices as  **_.png_**  images into two different folders as:
```
python data/read_nii_to_img.py

[Ref folder:]
000001.png,  000002.png,  000003.png,  000003.png ...
[Target folder:]
000001.png,  000002.png,  000003.png,  000003.png ...
# Note that the images in the Ref and Target folders correspond one to one. The undersampled target images will be automatically generated in the training phase.
```
### 2.2. Parpare Datasets for Fastmri:

[](https://github.com/lpcccc-cv/MC-DuDoNet?tab=readme-ov-file#22-parpare-datasets-for-fastmri)

The original Fastmri dataset can be downloaded at:  [[Fastmri dataset]](https://fastmri.med.nyu.edu/).  
For the paired Fastmri data (PD and FSPD), we follow the data preparation process of MINet and MTrans. For more details, please see  [[MINet]](https://github.com/chunmeifeng/MINet), [[MTrans]](https://github.com/chunmeifeng/MTrans) and [[MC-DuDoNet]](https://github.com/lpcccc-cv/MC-DuDoNet).

If you prefer not to preprocess the datasets yourself, you can download our preprocessed versions from the following link.
[[dataset]](https://drive.google.com/drive/folders/1A3A8dZsLHmaTwm4cbn671DYfcOzsc3db?usp=drive_link).

# Training and Testing of the Model

## Train

Modify your dataset paths and training parameters in **[configs/only_reconstruction.yaml]**, 
and update your dataset and mask paths in **[data/(IXI, brain, fastmri)_dataset]**, then run：
```
sh train_rec.sh
```

## Test
Modify the test configurations in Python file **[test_metrics.py]**. Then run:

```
python test_PSNR.py
```

## Acknowledgement
Our codes are built based on [LOUPE](https://github.com/cagladbahadir/LOUPE/) , [MC-DuDoNet](https://github.com/lpcccc-cv/MC-DuDoNet) and [[MTrans]](https://github.com/chunmeifeng/MTrans). Thank them for their outstanding contributions to the community.
