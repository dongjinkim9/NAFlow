import sys
sys.path.append('.')

from utils.common import instantiate_from_config, load_state_dict
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from pathlib import Path
import argparse
import torch

def sample_noisy_image(opt):	
	# load images
	topil = ToPILImage()
	totensor = ToTensor()
	
	noisy = Image.open(opt.noisy_pth).convert('RGB')
	noisy = totensor(noisy).unsqueeze(0)
	noisy = noisy.cuda()

	clean = Image.open(opt.clean_pth).convert('RGB')
	clean = totensor(clean).unsqueeze(0)
	clean = clean.cuda()

	# load model
	model = instantiate_from_config(OmegaConf.load(opt.config_pth))
	load_state_dict(model, torch.load(opt.ckpt_pth, map_location="cpu"), strict=True)
	model = model.cuda()

	# noise-aware sampling
	img = model(clean_img=clean, noisy_img=noisy)
	img = torch.clamp(img, 0.0, 1.0)
	out_dir = Path(opt.outdir) / 'output.png'
	topil(img[0]).save(out_dir)

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--clean-pth",
		type=str,
		nargs="?",
		help="path to the clean image",
		default="../assets/test_clean.png",
	)
	parser.add_argument(
		"--noisy-pth",
		type=str,
		nargs="?",
		help="path to the noisy image",
		default="../assets/test_noisy.png",
	)
	parser.add_argument(
		"--config-pth",
		type=str,
		nargs="?",
		help="path to the naflow's config",
		default="configs/models/naflow.yaml",
	)
	parser.add_argument(
		"--ckpt-pth",
		type=str,
		nargs="?",
		help="path to the naflow's checkpoint",
		default="pretrained_models/naflow_sidd.ckpt",
	)
	parser.add_argument(
		"--outdir",
		type=str,
		nargs="?",
		help="dir to write results to",
		default="../assets",
	)

	opt = parser.parse_args()

	sample_noisy_image(opt)

if __name__ == "__main__":
	main()