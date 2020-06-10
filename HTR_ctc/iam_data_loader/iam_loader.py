'''
@author: georgeretsi
'''

import os

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from skimage import io as img_io


from os.path import isfile

try:
    from utils.auxilary_functions import image_resize, centered
except:
    from HTR_ctc.utils.auxilary_functions import image_resize, centered

from .iam_config import *
from .iam_utils import gather_iam_info
def main_loader(set, level):

    info = gather_iam_info(set, level)

    data = []
    print(len(info))
    for i, (img_path, transcr) in enumerate(info):

        if i % 1000 == 0:
            print('imgs: [{}/{} ({:.0f}%)]'.format(i, len(info), 100. * i / len(info)))

        try:
            #img2 = img_io.imread(img_path + '.png')
            #img2 = 1 - img2.astype(np.float32) / 255.0
            img = cv2.normalize(cv2.imread(img_path + '.png', cv2.IMREAD_GRAYSCALE), None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        except:
            continue

        #if len(transcr) > 5:
        data += [(img.copy(), transcr.replace("|", " "))]

    if set == 'train':
        data = data[:int(0.8*len(data))]
    else:
        data = data[int(0.8 * len(data)):]
    print(len(data))
    return data

class IAMLoader(Dataset):

    def __init__(self, set, level='word', fixed_size=(128, None)):

        self.fixed_size = fixed_size

        save_file = dataset_path + '/' + set + '_' + level + '.pt'

        if isfile(save_file) is False:
            # hardcoded path and extension

            # if not os.path.isdir(img_save_path):
            #    os.mkdir(img_save_path)

            data = main_loader(set=set, level=level)
            torch.save(data, save_file)
        else:
            data = torch.load(save_file)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img = self.data[index][0]
        transcr = self.data[index][1]


        # add augmentation
        # ?? elastic, morphology, resizing

        nheight = self.fixed_size[0]
        nwidth = self.fixed_size[1]
        if nheight is None:
            nheight = img.shape[0]
        if nwidth is None:
            nwidth = int(np.random.uniform(.8, 1.2) * img.shape[1] * nheight / img.shape[0])

        img = image_resize(img, height=nheight-16, width=nwidth)
       # img = centered(img, (nheight, int(1.2 * nwidth) + 32))
        img = torch.Tensor(img).float().unsqueeze(0)

        return img, transcr
