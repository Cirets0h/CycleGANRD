import csv
import math
import random
import numpy as np
import torch
import cv2
from HTR_ctc.utils.auxilary_functions import image_resize

# Picture crop:
#  __________
# |x0y0     |
# |         |
# |    x1,y1|
#  _________

def getWordsInCrop(name, x0, y0, x1, y1, scale_factor = 1):
    delimiter = ';'
    scope_tolerance = 256 * 0.03 # normally try 0.05
    # print('x0: ' + str(x0) + ' y0: ' + str(y0) + 'x1: ' + str(x1) + ' y1: ' + str(y1))
    csvCrop = open('/home/manuel/CycleGANRD/HTR_ctc/data/generated/csv-crop/' + name + '-crop.csv', 'w+')
    word = ""
    pre = ""
    post = ""
    y_scope_out = False
    x0_new = 0
    y0_new = 0
    x1_new = 0
    y1_new = 0
    wordOver = False
    fix = False
    hasWord = False # Variable that is returned determinating if the crop contain a word (It could also be a crop thats is only a lettrine)
    csvCrop.write('word;x0;y0;x1;y1;y_scope_out;pre;post\n')
    with open('/home/manuel/CycleGANRD/HTR_ctc/data/generated/csv/' + name + '.csv') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            wordOver = False
            fix = False
            if row['char'] == ' ' or y1_new > float(row['y1']):  # word over
                if (word != "" or pre != "" or post != "") and pfloat(x0_new - x0) != 256 and pfloat(x1_new - x0) != 0: # while the last two and statements may seem obsolute, otherwise there is a bug where a random M occurs in the csv that has no length
                    hasWord = True
                    csvCrop.write(
                        word + delimiter + str(pfloat(x0_new - x0)) + delimiter + str(pfloat(y0_new - y0)) + delimiter + str(
                            pfloat(x1_new - x0)) + delimiter + str(pfloat(y1_new - y0)) + delimiter + str(
                            y_scope_out) + delimiter + pre + delimiter + post + '\n')
                word = ""
                pre = ""
                post = ""
                y_scope_out = False
                x0_new = 0
                y0_new = 0
                x1_new = 0
                y1_new = 0
                wordOver = True
            if x0 - scope_tolerance < float(row['x0']) and y0 - scope_tolerance < float(row['y0']) and x1 + scope_tolerance > float(row['x1']) and y1 + scope_tolerance > float(row['y1']) and not wordOver:
                if x0_new == 0:
                    x0_new = float(row['x0'])
                if y0_new == 0:
                    y0_new = float(row['y0'])
                    y1_new = float(row['y1'])
                if x1_new < float(row['x1']):
                    x1_new = float(row['x1'])

                if x0 < float(row['x0']) and y0 < float(row['y0']) and x1 > float(row['x1']) and y1 > float(row['y1']):
                    word = word + row['char']
                else:
                    if x0 - scope_tolerance < float(row['x0']) and not (x0 < float(row['x0'])): # not has to be checked extra, because otherwise it would also be true when y with tolerance would be true (last if)
                        pre = pre + row['char']
                        fix = True
                    if x1 + scope_tolerance > float(row['x1']) and not (x1 > float(row['x1'])):
                        post = post + row['char']
                        fix = True
                    if (y0 - scope_tolerance < float(row['y0']) and not (y0 < float(row['y0']))) or (y1 + scope_tolerance > float(row['y1'])  and not (y1  > float(row['y1']))):
                        if not fix:  # only add to word if char is not pre of postfix
                            word = word + row['char']
                        y_scope_out = True

    csvCrop.close()
    return hasWord

def cropWords(image, name, source):
    #print(name)
    if torch.is_tensor(image):
        image = image.detach().squeeze(0).transpose(0,2).transpose(0,1).cpu().numpy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[:, :, np.newaxis]

    word_array = []
    info_array = []
    with open(source + 'csv-crop/' + name + '.csv') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            # print(row['y0'])
            output = image[math.floor(float(row['y0'])) : math.ceil(float(row['y1'])), math.floor(float(row['x0'])) : math.ceil(float(row['x1'])), : ]
            word_array.append(output)
            info_array.append(row['pre'] + row['word'] + row['post'])
            # todo: insert below code and adjust Reading Discriminator for pre and postfixes
            #info_array.append({'word': row['word'], 'pre': row['pre'], 'post': row['post'], 'y_scope_out': row['y_scope_out']})

    #for x in range(0, len(word_array)):
    #    toimage((word_array[x].numpy()), cmin=-1, cmax=1).save('generate_book-crop/(' + info_array[x]['pre'] + ')' + info_array[x]['word'] + '(' + info_array[x]['post'] + ')' + info_array[x]['y_scope_out']  +'.png')
    return word_array, info_array

def pfloat(number): # positive float
    if number <= 0:
        return 0
    elif number >= 256:
        return 256
    else:
        return number



def getRandomCrop(image, image_name):
    boundingBox_size = 256
    hasWords = False
    while True:
        rand_x = random.randint(600,
                                2154)  # the text of the document is between 600 and 2154 (-256 = 2510) width
        rand_y = random.randint(582, 2937)
        # print('x: ' + str(rand_x) + ', y: ' + str(rand_y))
        #image = tf.image.crop_to_bounding_box(image, rand_y, rand_x, boundingBox_size, boundingBox_size)
        croppedImage = image[rand_y: rand_y + boundingBox_size, rand_x: rand_x + boundingBox_size,:]
        hasWords = getWordsInCrop(image_name.rsplit('.')[0], rand_x, rand_y, rand_x + boundingBox_size,
                                       rand_y + boundingBox_size)
        if np.mean(croppedImage) < 0.9 and hasWords:  # recrop if picture is too white (not enough text)
            break

    return croppedImage

#todo: delete
def normalize_array(array):
    return np.subtract(np.divide(array, 127.5), 1)

def resizeImg(img, fixed_size):
        nheight = fixed_size[0]
        nwidth = fixed_size[1]
        if nheight is None:
            nheight = img.shape[0]
        if nwidth is None:
            nwidth = int(np.random.uniform(.8, 1.2) * img.shape[1] * nheight / img.shape[0])

        img = image_resize(img, height=nheight-16, width=nwidth)
       # img = centered(img, (nheight, int(1.2 * nwidth) + 32))
        img = torch.Tensor(img).float().unsqueeze(0)
        return img


