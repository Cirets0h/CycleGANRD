
classes = '_!"#&\'()[]+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
cdict = {c:i for i,c in enumerate(classes)}
icdict = {i:c for i,c in enumerate(classes)}

data_name = 'line'
cnn_cfg = [(2, 32), 'M', (4, 64), 'M', (6, 128), 'M', (2, 256)]
rnn_cfg = (256, 1)  # (hidden , num_layers)

max_epochs = 40
batch_size = 1
iter_size = 16
# fixed_size

model_path = '/home/manuel/CycleGANRD/HTR_ctc/saved_models/'
save_model_name = 'crnn_' + data_name + '2.pt'
#load_model_name = None
load_model_name = 'crnn_' + data_name + '.pt'
