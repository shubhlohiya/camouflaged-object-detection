"""
 Please check references in the report/ppt for code references. 
 We have taken help of the official implementation of the paper in places
 , mostly for the architecture. (ResNet50 part)
"""


import torch
import argparse
from SINet import SINet_ResNet50
from dataloader import get_loader
from trainer import trainer, adjust_lr
from apex import amp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=40,
                        help='epoch number, default=30')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='init learning rate, try `lr=1e-4`')
    parser.add_argument('--batchsize', type=int, default=36,
                        help='training batch size (Note: ~500MB per img in GPU)')
    parser.add_argument('--trainsize', type=int, default=352, #352 was default
                        help='the size of training image, try small resolutions for speed (like 256)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='decay rate of learning rate per decay step')
    parser.add_argument('--decay_epoch', type=int, default=30,
                        help='every N epochs decay lr')
    parser.add_argument('--gpu', type=int, default=1,
                        help='choose which gpu you use')
    parser.add_argument('--save_epoch', type=int, default=10,
                        help='every N epochs save your trained snapshot')
    parser.add_argument('--save_model', type=str, default='./Snapshot/2020-CVPR-SINet/',
    	help='location where to save model')
    parser.add_argument('--train_img_dir', type=str, default='../COD10K-v3/Train/Image/',
    	help='location of training data')
    parser.add_argument('--train_gt_dir', type=str, default='../COD10K-v3/Train/GT_Object/',
    	help='location of training ground truths')
    opt = parser.parse_args()

    torch.cuda.set_device(0)

    # TIPS: you also can use deeper network for better performance like channel=64
    model_SINet = SINet_ResNet50(channel=64).cuda()
    print('-' * 30, model_SINet, '-' * 30)

    optimizer = torch.optim.Adam(model_SINet.parameters(), opt.lr)
    LogitsBCE = torch.nn.BCEWithLogitsLoss()

    net, optimizer = amp.initialize(model_SINet, optimizer, opt_level='O1')     # NOTES: Ox not 0x

    train_loader = get_loader(opt.train_img_dir, opt.train_gt_dir, batchsize=opt.batchsize,
                              trainsize=opt.trainsize, num_workers=12)
    total_step = len(train_loader)

    print('-' * 30, "\n[Training Dataset INFO]\nimg_dir: {}\ngt_dir: {}\nLearning Rate: {}\nBatch Size: {}\n"
                    "Training Save: {}\ntotal_num: {}\n".format(opt.train_img_dir, opt.train_gt_dir, opt.lr,
                                                              opt.batchsize, opt.save_model, total_step), '-' * 30)

    for epoch_iter in range(1, opt.epoch):
        adjust_lr(optimizer, epoch_iter, opt.decay_rate, opt.decay_epoch)
        trainer(train_loader=train_loader, model=model_SINet,
                optimizer=optimizer, epoch=epoch_iter,
                opt=opt, loss_func=LogitsBCE, total_step=total_step)
