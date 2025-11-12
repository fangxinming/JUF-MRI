import torch
import numpy as np
import contextlib
import cv2

@contextlib.contextmanager
def temp_seed(rng, seed):
    state = rng.get_state()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.set_state(state)
def mask_func_random_unique(shape, acc = 2, seed=20):
    """
    Args:
    shape:[320, 320, 2]

    Return:
    [1, 320, 1]非0即1的tensor
    """
    if len(shape) < 3:
        raise ValueError("Shape should have 3 or more dimensions")
    
    rng = np.random
    with temp_seed(rng, seed):
        num_cols = shape[-2]
        if acc == 4:
            center_fraction, acceleration = 0.08, 4 #中心采样比例，加速比
        elif acc == 8:
            center_fraction, acceleration = 0.04, 8
        elif acc == 2:
            center_fraction, acceleration = 0.16, 2  #这两行是我自己加的
        elif acc == 10:
            center_fraction, acceleration = 0.032, 10
        elif acc == 20:
            center_fraction, acceleration = 0.005, 20
        elif acc == 30:
            center_fraction, acceleration = 0.016, 30
        else:
            assert('accelerate rate is not implmented')

        # create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        #
        #(采样条数-中心采样条数) / 所有未采样条数 计算每一行的采样概率
        #
        prob = (num_cols / acceleration - num_low_freqs) / (
            num_cols - num_low_freqs)
        mask = rng.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad: pad + num_low_freqs] = True

        # reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32)) # mask.shape=[col, 1]
        mask = mask.repeat(shape[0], 1, 1)   
    return mask[:, :, 0]

acc=   3
mask = mask_func_random_unique([240, 240, 2],acc)
print(mask.shape)
cv2.imwrite('/home/xinming/MCDudo/dataset/Brats2018/mask/mask_random_x'+str(acc)+'.png', mask.numpy()*255)