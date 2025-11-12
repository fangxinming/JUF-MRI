
import models.modules.JUF_MRI_Loupe as JUF_MRI_Loupe
import models.modules.JUF_MRI_diffusion as JUF_MRI_diffusion

####################
# define network
####################
# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    opt_net['model'] = opt['model']
    if which_model in ['JUF_MRI_Loupe',]:
        netG = JUF_MRI_Loupe.JUF_MRI_Loupe(opt_net)
    elif which_model in ['JUF_MRI_diffusion']:
        netG = JUF_MRI_diffusion.JUF_MRI_diffusion(opt_net)

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

