import torch
from torch.autograd import Variable
from datetime import datetime
import os
#Uncomment the following lines to prevent apex error upon first time usage
#!git clone https://github.com/NVIDIA/apex
#%cd apex
#!pip install -v --no-cache-dir ./
from apex import amp
import torch.nn.functional as F


def eval_mae(y_pred, y):
    """
    returns the mean squared error for test or validation
    """
    return torch.abs(y_pred - y).mean()


def numpy2tensor(numpy):
    """
    converts a CPU numpy array to a GPU tensor
    """
    return torch.from_numpy(numpy).cuda()


def clip_gradient(optimizer, grad_clip):
    """
    gradient clipping to prevent exploding gradients
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, epoch, decay_rate=0.1, decay_epoch=30):
    """
    decrements the learning rate using a specified decay rate
    """
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def trainer(train_loader, model, optimizer, epoch, opt, loss_func, total_step):
    """
    trains the model with the given parameters
    """
    model.train()
    for step, data_pack in enumerate(train_loader):
        optimizer.zero_grad()
        images, gts = data_pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        cam_sm, cam_im = model(images)
        loss_sm = loss_func(cam_sm, gts)
        loss_im = loss_func(cam_im, gts)
        loss_total = loss_sm + loss_im

        with amp.scale_loss(loss_total, optimizer) as scale_loss:
            scale_loss.backward()

        # clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if step % 10 == 0 or step == total_step:
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_s: {:.4f} Loss_i: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss_sm.data, loss_im.data))
            #training specs for display

    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % opt.save_epoch == 0:
        torch.save(model.state_dict(), save_path + 'SINet_%d.pth' % (epoch+1))