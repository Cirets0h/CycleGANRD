from os.path import isfile

from .generated_utils import *
from .generated_config import *
import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import time
import torch
try:
    from utils.auxilary_functions import image_resize, centered
except:
    from HTR_ctc.utils.auxilary_functions import image_resize, centered

class GeneratedLoader(Dataset):

    def __init__(self, set = 'train', nr_of_channels=1, fixed_size=(128, None)):

        self.fixed_size = fixed_size

        save_file = dataset_path + 'generated_' + set + '.pt'

        if isfile(save_file) is False:
            print('Creating:  ' + save_file)
            data = []
            i = 0
            for image_name in data_image_names:
                t0 = time.time()
                if image_name[-1].lower() == 'g':  # to avoid e.g. thumbs.db files
                    if nr_of_channels == 1:  # Gray scale image -> MR image
                        image = cv2.normalize(cv2.imread(os.path.join(data_path, image_name), cv2.IMREAD_GRAYSCALE), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                        # todo: change data loader to delete one dimension in black and white, then delete squeeze in get
                        image = image[:, :, np.newaxis]
                    else:  # RGB image -> street view
                        image = cv2.normalize(cv2.imread(os.path.join(data_path, image_name)), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                    image = getRandomCrop(image, image_name)
                    word_array, info_array = cropWords(image, image_name.rsplit('.')[0] + '-crop')
                    for i in range(0, len(word_array)):
                        data.append([word_array[i].copy(), info_array[i]])

                i +=1

                if i % (len(data_image_names)//10) == 0:
                    print(str(i) + '/' + str(len(data_image_names)))
            torch.save(data, save_file)
            print('Finished:  ' + save_file)
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

        img = image_resize(img, height=nheight-16, width=nwidth)
       # img = centered(img, (nheight, int(1.2 * nwidth) + 32))
        img = torch.Tensor(img).float().unsqueeze(0)

        return img, transcr