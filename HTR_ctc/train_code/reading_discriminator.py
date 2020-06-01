import logging
from os.path import isfile

import numpy as np
import torch.cuda
from torch.autograd import Variable
from HTR_ctc.utils.auxilary_functions import torch_augm
from HTR_ctc.train_code.config import *
import editdistance
import matplotlib.pyplot as plt
import cv2

logger = logging.getLogger('Reading Discriminator')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parallelism on different graphic cards not important with batch size 1
#if torch.cuda.device_count() > 1:
#    print("Let's use", torch.cuda.device_count(), "GPUs!")
#    net = nn.DataParallel(net)

class ReadingDiscriminator():

    def __init__(self, optimizer, net, loss, lr=1e-4, load_model_name=None, rd_low_loss_learn = False):
        self.optimizer = optimizer
        self.lr = lr
        self.net = net.to(device)
        self.loss = loss
        self.step_count = 0
        self.step_freq = 1
        self.loadModel(load_model_name)
        self.rd_low_loss_learn = rd_low_loss_learn
        

    def train_on_Dataloader(self, epoch, train_loader, test_set, scheduler = None):
        if scheduler is not None:
            scheduler.step()
        self.optimizer.zero_grad()
        pos = 0
        ex = 0
        pos_5 = 0
        closs = []
        for iter_idx, (img, transcr) in enumerate(train_loader):
            loss, _ = self.train(img, transcr)
            closs += [loss.data]
            print(loss.data)
            if iter_idx % 50 == 1:
                logger.info('Epoch %d, Iteration %d: %f', epoch, iter_idx + 1, sum(closs) / len(closs))
                print(loss)
                closs = []

                tst_img, tst_transcr = test_set.__getitem__(np.random.randint(test_set.__len__()))
                estimated_word = self.getResult(tst_img)
                if estimated_word == tst_transcr:
                    pos += 1
                    if len(estimated_word) > 5:
                        pos_5 += 1
                print('orig:: ' + tst_transcr)
                print('greedy dec: ' + estimated_word)
                print('Accuracy: ' + str(pos) + '/' + str(int(iter_idx / 50 + 1)) + '= ' + str(
                    pos / int(iter_idx / 50 + 1)) + '| over 5 chars: ' + str(pos_5))
            # tdec, _, _, tdec_len = decoder.decode(tst_o.softmax(2).permute(1, 0, 2))
            # print('beam dec:: ' + ''.join([icdict[t.item()] for t in tdec[0, 0][:tdec_len[0, 0].item()]]))



    def train(self, img, transcr):
        img = Variable(img.to(device))
        # cuda augm - alternatively for cpu use it on dataloader
        img = torch_augm(img)
        np_img = img.detach().squeeze(0).permute(1,2,0).cpu().numpy()
        cv2.imwrite('/home/manuel/CycleGANRD/PyTorch-CycleGAN/output/test2.png', np_img*255)
        output = self.net(img)


        act_lens = torch.IntTensor(img.size(0) * [output.size(0)])
        try:
            labels = Variable(torch.IntTensor([cdict[c] for c in ''.join(transcr)]))
        except KeyError:
            print('Training failed because of unknown key: ' + str(KeyError))
            return -1, ''
        label_lens = torch.IntTensor([len(t) for t in transcr])

        output = output.log_softmax(2)  # .detach().requires_grad_()

        loss_val = self.loss(output.cpu(), labels, act_lens, label_lens)

        # if self.rd_low_loss_learn is true, the network only learns on a loss lower than 1
        # if off it learns on everything
        if not self.rd_low_loss_learn or loss_val < 1:
            loss_val.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            #cv2.imwrite('/home/manuel/CycleGANRD/PyTorch-CycleGAN/output/test1.png', np_img * 255)


        return loss_val, output
    
    def getResult(self, img):
        self.net.eval()
        try:

            img = img.unsqueeze(0) #.transpose(1, 3).transpose(2, 3)
            with torch.no_grad():
                tst_o = self.net(Variable(img.cuda()))
            tdec = tst_o.log_softmax(2).argmax(2).permute(1, 0).cpu().numpy().squeeze()

            # todo: create a better way than to just ignore output with size [1, 1, 80] (first 1 has to be >1
            tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
        except:
            tt = ''
            print('Error occured')
        estimated_word = ''.join([icdict[t] for t in tt]).replace('_', '')
        self.net.train()

        return estimated_word

    def test(self, epoch, test_loader):
        self.net.eval()

        logger.info('Testing at epoch %d', epoch)
        cer, wer = [], []
        for (img, transcr) in test_loader:
            transcr = transcr[0]
            img = Variable(img.to(device))
            with torch.no_grad():
                o = self.net(img)
            tdec = o.argmax(2).permute(1, 0).cpu().numpy().squeeze()
            tt = [v for j, v in enumerate(tdec) if j == 0 or v != tdec[j - 1]]
            dec_transcr = ''.join([icdict[t] for t in tt]).replace('_', '')
            # tdec, _, _, tdec_len = decoder.decode(o.softmax(2).permute(1, 0, 2))
            # dec_transcr = ''.join([icdict[t.item()] for t in tdec[0, 0][:tdec_len[0, 0].item()]])

            cer += [float(editdistance.eval(dec_transcr, transcr)) / len(transcr)]
            wer += [float(editdistance.eval(dec_transcr.split(' '), transcr.split(' '))) / len(transcr.split(' '))]

        logger.info('CER at epoch %d: %f', epoch, sum(cer) / len(cer))
        logger.info('WER at epoch %d: %f', epoch, sum(wer) / len(wer))

        self.net.train()

    def saveModel(self, filename):
        torch.save(self.net.state_dict(), model_path + filename)

    def loadModel(self, filename):
        if isfile(model_path + filename):
            load_parameters = torch.load(model_path + filename)
            self.net.load_state_dict(load_parameters)
            self.net.to(device)
            logger.info('Loading model parameters for RD successfull')
        elif filename is not None:
            logger.info('Loading model parameters failed, ' + str(model_path + filename) + 'not found')