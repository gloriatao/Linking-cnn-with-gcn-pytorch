import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from utils import listDataset, logging
from config import Config
from utils.models import Av_CNN3D_model, Av_CNN_GCN_model#, Av_CNN_GCN_trans_model

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def adjust_learning_rate(optimizer, batch, steps, scales, lr, batchSz):
    for i in range(len(steps)):
        scale = scales[i] if i < len(scales) else 1
        if batch >= steps[i]:
            lr = lr * scale
            if batch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / batchSz
    return lr

def train_epoch(epoch,train_loader, config, writer=None):
    global processed_batches
    t0 = time.time()
    logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * len(train_loader.dataset), config.learning_rate))

    model.train()
    processed_batches = 0
    correct, total = 0, 0
    f = open(config.logFile, 'a')

    for batch_idx, (X_batch, Y_batch, NX_batch) in enumerate(train_loader):
        processed_batches = processed_batches + 1
        X_batch, Y_batch, NX_batch = X_batch.cuda().squeeze(0), Y_batch.cuda().squeeze(0), NX_batch.cuda().squeeze(0)

        optimizer.zero_grad()

        output = model.forward(X_batch, NX_batch)
        # if len(output.shape) == 3:
        #     output = output.reshape(config.batch_size*config.num_nodes, -1)
        #     Y_batch = Y_batch.reshape(config.batch_size * config.num_nodes)
        loss = nn.CrossEntropyLoss()(output, Y_batch)

        pred = torch.argmax(output, dim=1)
        correct += torch.sum(pred.eq(Y_batch))
        total += output.shape[0]
        acc = np.array(correct.cpu())/total

        print('epoch: %d, processed_batches: %d, loss: %f'
              % (epoch, processed_batches, loss.item()))
        print('acc:', acc)

        f.write('%0.6f' % (loss.item()) + ' ' + '%0.6f' % (acc.item()) + '\n')
        loss.backward()
        optimizer.step()

    t1 = time.time()
    logging('training with %f samples/s' % (len(train_loader.dataset) / (t1 - t0)))
    f.close()
    if (epoch + 1) % config.save_interval == 0:
        torch.save({'epoch': epoch,
                    'seen': processed_batches,
                    'state_dict': model.state_dict()},
                   '%s/%06d.pkl' % ('backup', np.int(epoch/12)))

    print("done")

if __name__ == '__main__':
    config = Config()
    model_name = config.model_name
    use_cuda = torch.cuda.is_available()

    # path-----------------------------------------------------------------------------------
    if not os.path.exists(config.backupDir):
        os.mkdir(config.backupDir)

    # GPU-----------------------------------------------------------------------------------
    kwargs = {'num_workers': config.num_workers, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda:%s" % str(config.gpus[0]) if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(config.gpus[0])
        print("GPU is available!")
    else:
        print("GPU is not available!!!")

    # Load config params-----------------------------------------------------------------------
    if model_name == 'AV_CNN3D':
        usingNeighbors = False
        model = Av_CNN3D_model(droupout_rate=config.dp, number_class=config.Num_classes)
    elif model_name == 'AV_CNN_GCN':
        model = Av_CNN_GCN_model(cnnOFeat_len=10, gcnOFeat_len=config.Num_classes,
                                 gcnNumGaussian=6, gaussian_hidden_feat=3, number_neighbors=2, droupout_rate=0.5)

    model = model.cuda()

    # tesnorboard---------------------------------------------------------------------------------
    writer = None
    if config.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(config.tensorboard_logsDir)

    # optimizer-----------------------------------------------------------------------------------
    if config.solver == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate/config.batch_size, momentum=config.momentum,
                              weight_decay=config.decay*config.batch_size, nesterov=True)
    elif config.solver == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=config.betas, weight_decay=config.decay, amsgrad=True)
    else:
        print('No %s solver! Please check your config file!' % (config.solver))
        exit()

    # weights-----------------------------------------------------------------------------------
    if config.weightFile != 'none':
        model.load_weights(config.weightFile)
    else:
        model.seen = 0
    # Data Loader-----------------------------------------------------------------------------------
    processed_batches = model.seen / config.batch_size
    train_loader = DataLoader(
        listDataset(config.imgDirPath, config.case_list_train, Num_nodes=config.num_nodes, Num_neighbour=config.Num_neighbors),
        batch_size=config.batch_size, shuffle=True, drop_last=True)

    init_epoch = 0
    for epoch in range(init_epoch, config.max_epochs):
        train_epoch(epoch, train_loader, config, writer)

    if config.tensorboard:
        writer.close()

    print('Done!')





