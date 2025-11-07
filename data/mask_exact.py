import torch


def squash_mask(mask,pmask_slope=5):
    return torch.sigmoid(pmask_slope * mask)

def sparsify(mask,sparsity):
        xbar = mask.mean()  #计算整个mask的均值  代表目前掩码的稀疏度
        r = sparsity / xbar   #稀疏度除以均值    要求的稀疏度除以目前的稀疏度
        beta = (1 - sparsity) / (1 - xbar)
        le = (r <= 1).float()  #如果r<=1 说明要求的稀疏度比目前的稀疏度小 说明现在采样采多了 如果r>1 说明目前采样采少了
        return le * mask * r + (1 - le) * (1 - (1 - mask) * beta)
        # 如果r<=1 即采样采多了 le =1   则return   mask * r 整个稀疏度变为要求稀疏度
        # 如果 r>1 即说明采样采少了 le =0 则return   1-（1-mask）* beta 整个稀疏度变为要求稀疏度

def sigmoid_beta(mask,image_dims,sample_slope=12):
    random_uniform = torch.empty(*image_dims).uniform_(0, 1)
        #torch.empty 为创建一个具有self.image_dims 的空张量 uniform(0,1)在空张量中填充随机值，这些值在[0,1]之间
    return torch.sigmoid(sample_slope * (mask - random_uniform))
        # 从mask中减去随机数张量