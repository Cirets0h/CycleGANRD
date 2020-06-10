import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cv2
from HTR_ctc.generated_data_loader.generated_utils import *

#todo: store all pictures, word crops and info in dataloader
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.files_A = []
        self.files_B = []
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        

        self.crops_A =  generateCrops(3, '/home/manuel/CycleGANRD/PyTorch-CycleGAN/datasets/cropped_data/', just_generate=True, crop_path='train/A/')

        self.files_B_name = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

        for name_B in self.files_B_name:
            a = cv2.imread(name_B)
            item_B = self.transform(cv2.normalize(a, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
            self.files_B.append([item_B, name_B])

        for crop_A, csvfile_A in self.crops_A:
            item_A = self.transform(cv2.normalize(crop_A, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
            self.files_A.append([item_A, csvfile_A])



    def __getitem__(self, index):
        item_A, A_csv = self.files_A[index % len(self.crops_A)]
        item_B, name_B = self.files_B[index % len(self.files_B_name)]
        return {'A': item_A, 'A_csv': A_csv, 'B': item_B, 'B_name': name_B}

    def __len__(self):
        return max(len(self.crops_A), len(self.files_B_name))
    
    def __lenA__(self):
        return len(self.crops_A)