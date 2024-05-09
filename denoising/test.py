import torch

from models.network_plain import DnCNN
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import save_image

totensor = ToTensor()

denoiser = DnCNN(in_nc=3, out_nc=3, nc=64, act_mode='BR')
ckpt_path = './pretrained/checkpoint_G3_00000300.pth'
checkpoint = torch.load(ckpt_path)
denoiser.load_state_dict(checkpoint['model_state_dict'])
denoiser.cuda()
denoiser.eval()

noisy_path = 'ADD TO YOUR NOISY IMAGE'

test_noisy = Image.open(noisy_path).convert('RGB')
test_noisy = totensor(test_noisy).unsqueeze(0)
test_noisy = test_noisy.cuda()

with torch.no_grad():
    denoised = denoiser(test_noisy)
    denoised = torch.clamp(denoised, 0.0, 1.0)
    save_image(denoised, 'result.png')