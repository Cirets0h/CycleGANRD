import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cv2

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

    def __getitem__(self, index):
        name_A = self.files_A[index % len(self.files_A)]
        item_A = cv2.normalize(cv2.imread(name_A), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        item_A = self.transform(item_A)
        if self.unaligned:
            name_B = self.files_B[random.randint(0, len(self.files_B) - 1)]
        else:
            name_B = self.files_B[index % len(self.files_B)]

        item_B = self.transform(cv2.normalize(cv2.imread(name_B), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        return {'A': item_A, 'A_name': name_A, 'B': item_B, 'B_name': name_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))