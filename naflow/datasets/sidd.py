import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd
import random

class crop(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size
        

    def __call__(self, *inputs):
        ih, iw = inputs[0].shape[:2]
        try:
            ix = random.randrange(0, iw - self.patch_size +1)
            iy = random.randrange(0, ih - self.patch_size +1)
        except(ValueError):
            print('>> patch size: {}'.format(self.patch_size))
            print('>> ih, iw: {}, {}'.format(ih, iw))
            exit()

        output_list = [] 
        for inp in inputs:
            output_list.append(inp[iy : iy + self.patch_size, ix : ix + self.patch_size])
        
        if(len(output_list) > 1):
            return output_list
        else:
            return output_list[0]

class augmentation(object):
    def __call__(self, *inputs):

        hor_flip = random.randrange(0,2)
        ver_flip = random.randrange(0,2)
        rot = random.randrange(0,2)
        degree = random.randrange(1,4)

        output_list = []
        for inp in inputs:
            if hor_flip:
                tmp_inp = np.fliplr(inp)
                inp = tmp_inp.copy()
                del tmp_inp
            if ver_flip:
                tmp_inp = np.flipud(inp)
                inp = tmp_inp.copy()
                del tmp_inp
            if rot:
                inp = np.rot90(inp, degree)
                # inp = inp.transpose(1, 0, 2)
            output_list.append(inp)

        if(len(output_list) > 1):
            return output_list
        else:
            return output_list[0]

def collect_metadata(dataroot):
    idx_to_metadata = {}
    img_list = []
    for img_dir in Path(dataroot).glob('**/*.PNG'):
        index = img_dir.name.split('_')[0]
        idx_to_metadata[index] = {k:v for k,v in zip(
                ['img_index', 'scene', 'camera', 'iso', 'shutter', 'temperature', 'brightness'], 
                str(img_dir.parent).split(os.sep)[-1].split('_'))}
        img_list.append(str(img_dir))
    return img_list, idx_to_metadata

def check_paired_data(dir_1, dir_2):
    for d1, d2 in zip(dir_1,dir_2):
        index_1, label_1, _, shot_1 = Path(d1).name.split('.')[0].split('_')
        index_2, label_2, _, shot_2 = Path(d2).name.split('.')[0].split('_')
        assert index_1 == index_2 and label_1 != label_2 and shot_1 == shot_2, f'{index_1=} != {index_2=} or {label_1=} != {label_2=}'

class SIDDGTLQdataset(Dataset):
    def __init__(self, 
                 dataroot:str, classes:list, patch_size:int, 
                 preload:bool, test:bool, **kwargs):
        """
        - classes: the list of class you want to include ex) ['S6_00100'] 
        if value is None, all classes will be included and it won't provide classes
        """

        super().__init__()
        self.preload = preload
        self.test = test
        self.classes = classes

        if not test:
            self.transform = [crop(patch_size), augmentation()]
            
            self.img_list, self.idx_to_metadata = collect_metadata(dataroot)
            if classes != None:
                self.img_list = self.filter_imgs(classes)
            
            self.GT_dir = sorted([img_dir for img_dir in self.img_list if 'GT' in img_dir])
            self.LQ_dir = sorted([img_dir for img_dir in self.img_list if 'NOISY' in img_dir])

            assert len(self.GT_dir) != 0
            assert len(self.GT_dir) == len(self.LQ_dir)
            check_paired_data(self.GT_dir, self.LQ_dir)
        else:
            self.transform = []

            GT_dir = Path(dataroot) / 'groundtruth'
            LQ_dir = Path(dataroot) / 'input'
            GT_all_dir = sorted([str(GT_dir / this_path) for this_path in GT_dir.iterdir()])
            LQ_all_dir = sorted([str(LQ_dir/ this_path) for this_path in LQ_dir.iterdir()])
            assert len(GT_all_dir) == len(LQ_all_dir)
            assert len(GT_all_dir) != 0
            self.val_labels = pd.read_csv(str(Path(dataroot) / "validation_label.csv"), header=None,names=['val_index', 'val_metadata'])
            
            self.idx_to_metadata = dict()
            valid_index = list()
            for idx, val_label in self.val_labels.iterrows():
                # ['img_index', 'scene', 'camera_type', 'iso', 'shutter', 'temperature', 'brightness']
                val_index, val_metadata = val_label['val_index'], val_label['val_metadata'].split('_')
                self.idx_to_metadata[f"{val_index:04d}"] = {k:v for k,v in zip(
                    ['img_index', 'scene', 'camera', 'iso', 'shutter', 'temperature', 'brightness'], val_metadata)}
                if classes == None:
                    valid_index.append(f"{val_index:04d}")
                else:
                    if f"{val_metadata[2]}_{val_metadata[3]}" in classes:
                        valid_index.append(f"{val_index:04d}")

            self.GT_dir = [gt for gt in GT_all_dir if Path(gt).name.split('-')[0] in valid_index]
            self.LQ_dir = [lq for lq in LQ_all_dir if Path(lq).name.split('-')[0] in valid_index]
            assert len(self.GT_dir) == len(self.LQ_dir)
            assert len(self.GT_dir) != 0
        
            
        if preload:
            print('copying training data into RAM.')
            self.GT, self.LQ = [], []
            for dir in self.GT_dir:
                self.GT.append(np.array(Image.open(dir).convert('RGB')))
            for dir in self.LQ_dir:
                self.LQ.append(np.array(Image.open(dir).convert('RGB')))

    def __len__(self):
        return len(self.GT_dir)

    def __getitem__(self, idx):
        if self.preload:
            GT = self.GT[idx]
            LQ = self.LQ[idx]
        else:
            GT = np.array(Image.open(self.GT_dir[idx]).convert('RGB'))
            LQ = np.array(Image.open(self.LQ_dir[idx]).convert('RGB'))

        if self.transform is not None:
            for tr in self.transform:
                GT, LQ = tr(GT, LQ)

        img_item = {}

        # range [0, 1]
        img_item['GT'] = GT.transpose(2, 0, 1).astype(np.float32) / 255.
        img_item['LQ'] = LQ.transpose(2, 0, 1).astype(np.float32) / 255.
        
        img_item['noise'] = img_item['LQ'] - img_item['GT']
        img_item['dir'] = self.GT_dir[idx], self.LQ_dir[idx]
        index_sep = '-' if self.test else '_'
        img_item['metadata'] = self.idx_to_metadata[Path(self.GT_dir[idx]).name.split(index_sep)[0]]
        img_item['class'] = f"{img_item['metadata']['camera']}_{img_item['metadata']['iso']}"
        img_item['txt'] = ""
        return img_item
        
    def filter_imgs(self , conds = ['S6_00100']):
        new_img_dir = []
        for img_dir in self.img_list:
            index = Path(img_dir).name.split('_')[0]
            metadata = self.idx_to_metadata[index]
            if f"{metadata['camera']}_{metadata['iso']}" in conds: 
                new_img_dir.append(img_dir)
        return new_img_dir    

    def get_class_distribution(self ,test = False):
        classes = []
        sep = '-' if test else '_'
        img_dirs = self.LQ_dir

        for img_dir in img_dirs:
            metadata = self.idx_to_metadata[Path(img_dir).name.split(sep)[0]]
            if 'GT' in img_dir:
                classes.append('GT')
            else:
                classes.append(f"{metadata['camera']}_{metadata['iso']}")
        
        return classes
