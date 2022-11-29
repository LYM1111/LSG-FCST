
import os
import cv2
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
from PIL import Image
from evaluator.ssim import SSIM,MSSSIM
from lpips import LPIPS
from evaluator.fid import FID
import torchvision.transforms as transform
from evaluator import fid
import torch
import torch.nn as nn
import random
from PIL import Image

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()
    example1 = np.zeros((64, 64, 3))
    sum_ssim = []
    sum_lpips = []
    sum_l1 = []
    sum_RSME = []
    sum_fid =[]
    sum_msssim = []
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference\
    
        visuals = model.get_current_visuals()  # get image results
        img_path= model.get_image_paths()     # get image paths

        #LPIPS
        criterionLPIPS = LPIPS().to(device)
        criter_lpips = criterionLPIPS(visuals['gt_images'],visuals['generated_images']).item()
        #SSIM
        criterionSSIM = SSIM(window_size=11, size_average=True, val_range=None, channel=1)
        ssim = criterionSSIM(visuals['gt_images'],visuals['generated_images']).item()
        #MSSSIM
        criterionMSSSIM = MSSSIM(weights=[0.45, 0.3, 0.25]).to(device)
        msssim = criterionMSSSIM(visuals['gt_images'],visuals['generated_images']).item()
        if np.isnan(msssim):
            msssim = 0.
        
        criterion = nn.L1Loss(reduction = 'mean')
        L1_loss = criterion(visuals['gt_images'],visuals['generated_images']).item()
        #RSME
        RSME = nn.MSELoss()
        rsme = RSME(visuals['gt_images'],visuals['generated_images'])
       
        if i % 1000 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize,criter_lpips=criter_lpips,ssim=ssim,msssim=msssim,L1_loss=L1_loss,rsme=rsme,content_mean_sim=content_mean_sim,hde_mean_sim=hde_mean_sim,style_char=style_char)
        sum_lpips.append(criter_lpips)
        sum_ssim.append(ssim)
        sum_l1.append(L1_loss)
        sum_RSME.append(rsme)
        sum_msssim.append(msssim)

    lpips_arv = sum(sum_lpips)/len(sum_lpips)
    sim_arv = sum(sum_ssim)/len(sum_ssim)
    L1_arv = sum(sum_l1)/len(sum_l1)
    mse_arv = sum(sum_RSME)/len(sum_RSME)
    msssim = sum(sum_msssim)/len(sum_msssim)
    with open('text_{}_{}_{}'.format(opt.model,opt.name,opt.phase), 'a+') as f:
        f.write('\n')
        f.write(opt.name)
        f.write(opt.phase)
        f.write('\n')
        tmp_wr = str([lpips_arv, sim_arv, L1_arv, mse_arv, msssim])
        f.write(tmp_wr)
    webpage.save()  # save the HTML

