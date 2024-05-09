#%%
from models.lit_naflow import LitNAFlow
import torch
from PIL import Image
from IPython.display import display
from torchvision.transforms import ToPILImage, ToTensor
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

topil = ToPILImage()
totensor = ToTensor()

#%%
checkpoint_list = [
    '/home/dj_kim/mnt/nas/projects/denoising/iclr/NAFlow/naflow/pretrained_models/naflow_sidd.ckpt'
]
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#%%
model = LitNAFlow.load_from_checkpoint(checkpoint_list[0])
model = model.to(device)


#%%
from utils.common import instantiate_from_config, load_state_dict
from omegaconf import OmegaConf

config_pth = '/home/dj_kim/mnt/nas/projects/denoising/iclr/NAFlow/naflow/configs/models/naflow.yaml'
model = instantiate_from_config(OmegaConf.load(config_pth))
load_state_dict(model, torch.load(checkpoint_list[0], map_location="cpu"), strict=True)
model = model.to(device)

#%%

img_idx = 1

img_list = [
    '/home/dj_kim/dataset/sidd/validation/input/0004-0029.png', # 4 IP_01600
    '/home/dj_kim/dataset/sidd/validation/input/0001-0004.png', # 1 GP_10000
]
img = Image.open(img_list[img_idx]).convert('RGB')
img = totensor(img).unsqueeze(0)
img = img.to(device)
display(topil(img[0]))

#%%
gt_list = [
    '/home/dj_kim/dataset/sidd/validation/groundtruth/0004-0029.png', # 4 IP_01600
    '/home/dj_kim/dataset/sidd/validation/groundtruth/0001-0004.png', # 1 GP_10000
]
gt = Image.open(gt_list[img_idx]).convert('RGB')
gt = totensor(gt).unsqueeze(0)
gt = gt.to(device)
display(topil(gt[0]))


#%%
img = model(clean_img=gt, noisy_img=img)
img = torch.clamp(img, 0.0, 1.0)
ToPILImage()(img[0]).show()
#%%
z, nll, logdet = model(input=img, z=None, mode='forward', label= ['S6_00100'])

# %%

# new_z = model.model.sample_z_using_target_class(z.shape, 'S6_03200')
# new_img, logdet = model(input=img, z=new_z, mode='reverse')
# display(topil(new_img[0]))
# %%

# # 'IP_01600', 'S6_00100','S6_03200', 
# target_labels = ['GP_00100','GP_10000']
# for tl in target_labels:
#     new_z = model.model.sample_z_using_target_class(z.shape, [tl])
#     new_img, logdet = model(input=img, z=new_z, mode='reverse')
#     print(tl)
#     display(topil(new_img[0]))
# # %%
# display(topil((img-gt).clamp(0,1)[0]))
# display(topil((new_img-gt).clamp(0,1)[0]))
# display(topil((new_img - (img-gt)).clamp(0,1)[0]))
# # %%
