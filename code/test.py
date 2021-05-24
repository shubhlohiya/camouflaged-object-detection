import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc  # required: pip install scipy == 1.2.2
from SINet import SINet_ResNet50
from dataloader import test_dataset
from trainer import eval_mae, numpy2tensor


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                    default='./Snapshot/2020-CVPR-SINet/SINet_40.pth', help='path of saved model')
parser.add_argument('--test_save', type=str,
                    default='./Result/2020-CVPR-SINet-New/',  help='location to save output data')
opt = parser.parse_args()

model = SINet_ResNet50().cuda()
model.load_state_dict(torch.load(opt.model_path))
model.eval()

for dataset in ['COD10K-v3']:
    save_path = opt.test_save + dataset + '/'
    os.makedirs(save_path, exist_ok=True)
    test_loader = test_dataset(image_root='../dataset/{}/Test/Image/'.format(dataset),
                               gt_root='../dataset/{}/Test/GT_Object/'.format(dataset),
                               testsize=opt.testsize)
    img_count = 1
    for iteration in range(test_loader.size):
        # load data
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        # inference
        _, cam = model(image)
        # reshape and squeeze
        cam = F.upsample(cam, size=gt.shape, mode='bilinear', align_corners=True)
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        misc.imsave(save_path+name, cam)
        # evaluate
        mae = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
        # coarse score
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae))
        img_count += 1

print("\n[Congratulations! Testing Done]")
