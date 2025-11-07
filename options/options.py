import os
import os.path as osp
import logging
import yaml
from utils.util import OrderedYaml


Loader, Dumper = OrderedYaml() #使得Loader与Dumper支持有序字典
#loader 主要负责从YAML文件中读取数据，并将其转换为Python对象
#Dumper 主要负责将 Python 对象转换为 YAML 格式并写入文件

def parse(opt_path, is_train=True):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
        #加载这个yaml文件
        #此处的opt为整个大的opt
    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    #把指定的GPU列表转化为一个由逗号分隔的字符串
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    #表示程序将可以访问那些GPU
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)


    opt['is_train'] = is_train
    batch_size= opt['batch_size']
    scale = opt['scale']
    mode = opt['mode']
    if mode == 'IXI':
        opt['image_size'] = [256,256]
        GT = "T2"
    elif mode == 'Brats2018':
        opt['image_size'] = [240,240]
        GT = "T2"
    elif mode == 'fastmri':
        opt['image_size'] = [256, 256]
        GT = "PDFS"





    opt['network_G']['sparsity']= 1.0/scale
    opt['network_G']['which_model_G']=opt['model_G']
    opt['network_G']['image_size']=opt['image_size']
    opt['method'] = opt['model_G'].split("_")[-1]
    opt['train']['pixel_criterion'] = opt['loss']
    #print(opt['train']['pixel_criterion'])


    if opt['model']== 'ref-rec':  #代表模型的任务只是重建不涉及学习亚采样方式
        model= '_rec'
    elif opt['model'] == 'joint-optimize':  #代表重建的同时优化亚采样模式
        model= '_joint'
        opt['undersample'] = 'ING'
    else:
        print("wrong model")
    undersample = opt['undersample']

    opt['name'] = opt['mode']+'_'+opt['model_G']+'_'+opt['undersample']+'_x'+str(opt['scale'])+model
    #名字 为：数据集_网络名称_亚采样方式_x倍数_任务
    # datasets
    for phase, dataset in opt['datasets'].items():
        #.items方法用于字典所有的键值对 datasets的键为train与val ,值为对应字典
        if phase== 'train':
            dataset['batch_size']=batch_size
        dataset['mode'] = mode
        dataset['phase'] = phase
        dataset['scale'] = scale
        dataset['undersample'] = undersample
        dataset['model']= opt['model']
        dataset['model_G']= opt['model_G']
        dataset['method'] = opt['method']
        if mode == "fastmri":
            dataset['dataroot_GT'] = "/home/xinming/MCDudo/dataset/" +mode+"/knee_singlecoil/"+phase+"/"+GT+"PNG"
        else:
            dataset['dataroot_GT'] = "/home/xinming/MCDudo/dataset/" + mode + "/" + phase + "/small" + GT + "PNG"

        #把大的opt中的scale赋给datasets中train与val的scale，所以调试的时候不用管这些scale
        if dataset.get('dataroot_GT', None) is not None:
            print('路径存在')

    # path
    for key, path in opt['path'].items():
        if path and key in opt['path'] and key != 'strict_load':
            opt['path'][key] = osp.expanduser(path)
    opt['path']['root'] = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    # opt['path']['root'] = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    if is_train:
        experiments_root = os.path.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = os.path.join(experiments_root, 'models')
        opt['path']['training_state'] = os.path.join(experiments_root, 'training_state')
        opt['path']['log'] = experiments_root
        opt['path']['val_images'] = os.path.join(experiments_root, 'val_images')

        # # change some options for debug mode
        # if 'debug' in opt['name']:
        #     opt['train']['val_freq'] = 8
        #     opt['logger']['print_freq'] = 1
        #     opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = os.path.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root

    # network
    opt['network_G']['scale'] = scale

    return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


# convert to NoneDict, which return None for missing key.
class NoneDict(dict):
    def __missing__(self, key):
        return None


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

def check_resume(opt, resume_iter):
    '''Check resume states and pretrain_model paths'''
    logger = logging.getLogger('base')
    if opt['path']['resume_state']:
        if opt['path'].get('pretrain_model_G', None) is not None or opt['path'].get(
                'pretrain_model_D', None) is not None:
            #如果pretrain_model_G不存在就输出None
            logger.warning('pretrain_model path will be ignored when resuming training.')

        if opt['path']['pretrain_model_G'] is None:
            opt['path']['pretrain_model_G'] = osp.join(opt['path']['models'], 'Iter_{}.pth'.format(resume_iter))
        else:
            pass

        logger.info('Set [pretrain_model_G] to ' + opt['path']['pretrain_model_G'])
        if 'gan' in opt['model']:
            opt['path']['pretrain_model_D'] = osp.join(opt['path']['models'],
                                                       '{}_D.pth'.format(resume_iter))
            logger.info('Set [pretrain_model_D] to ' + opt['path']['pretrain_model_D'])
