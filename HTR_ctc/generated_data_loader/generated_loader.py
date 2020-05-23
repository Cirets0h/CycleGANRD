from os.path import isfile

from .generated_utils import *
from .generated_config import *
import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import time
import torch

from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian


try:
    from utils.auxilary_functions import image_resize, centered
except:
    from HTR_ctc.utils.auxilary_functions import image_resize, centered

class GeneratedLoader(Dataset):

    def __init__(self, set = 'train', augment_factor = 0, resize = False, nr_of_channels=1, fixed_size=(128, None)):
        self.resize = resize
        self.augment_factor = augment_factor
        self.fixed_size = fixed_size

        save_file = dataset_path + 'generated_' + set + '.pt'

        if isfile(save_file) is False:
            data = generateCrops(nr_of_channels,'/home/manuel/CycleGANRD/HTR_ctc/data/generated/', just_generate=False)
            torch.save(data, save_file)
        else:
            data = torch.load(save_file)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img = self.data[index][0].squeeze()
        transcr = self.data[index][1]


        # add augmentation
        # ?? elastic, morphology, resizing

        nheight = self.fixed_size[0]
        nwidth = self.fixed_size[1]
        if nheight is None:
            nheight = img.shape[0]

        if nwidth is None:
            nwidth = int(np.random.uniform(.8, 1.2) * img.shape[1] * nheight / img.shape[0])


        #augmentation
        noise = np.random.uniform(.1, 0.25) * self.augment_factor
        blur = np.random.uniform(.5, 2.0) * self.augment_factor

        #img = image_resize(img, height=2000, width=(int(2000/nheight)*nwidth))
        img = rotate(img, angle=np.random.random_integers(-5, 5), mode='constant', cval= 1, resize=True) # rotating
        img = random_noise(img, var=noise ** 2) # adding noise
        img = gaussian(img, sigma=blur, multichannel=True) # blurring



        #end augmentation


        img = image_resize(img, height=nheight-16, width=nwidth)

        img = torch.Tensor(img).float().unsqueeze(0)



        return img, transcr

