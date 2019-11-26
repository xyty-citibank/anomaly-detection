
import mmcv
import matplotlib.pyplot as plt
from mmanomaly.models.g_d_networks import *
from mmanomaly.models.loss import GANLoss, GradientLoss
from mmanomaly.models.base_model import BaseModel
from mmanomaly.FlowNetPytorch.flownet import FlowNet
import time
import os
import copy
import numpy as np
import cv2
import random
"""
This class implements the ano_pred_cvpr2018 model, the paper's url:1712.09867.pdf
"""
class APCModel(BaseModel):
    def __init__(self, opt):
        super(APCModel, self).__init__()
        self.opt = opt
        self.generator = UNetGenerator()
        self.discriminator = Discriminator()
        self.flow_net = FlowNet(opt.train_cfg.flownet_pretrained)
        self.loss = GANLoss(opt.model.gan.ganmodel)
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()
        self.gradientloss = GradientLoss(opt.train_cfg.alpha)
        self.batch_size = opt.gpus * opt.data.videos_per_gpu
        self.num_pred = opt.data.train.num_pred,
        self.time_steps = opt.data.train.time_steps
        self.c, self.w, self.h = opt.data.train.scale_size
        self.p_s = 1000
        self.key = 0

    def forward(self, input):
        self.g_t = input[:, :self.c * self.time_steps, :, :]
        self.g_t_1 = input[:, self.c * self.time_steps:, :, :]
        self.f_g_t = input[:, (self.c * self.time_steps - 3):self.c * self.time_steps, :, :]
        self.p_t_1 = self.generator(self.g_t)
        self.real_f = self.flow_net(self.f_g_t, self.g_t_1)
        self.fake_f = self.flow_net(self.f_g_t, self.p_t_1)




    def backward_D(self):
        fake = round(random.uniform(0.0, 0.1), 1)
        real = round(random.uniform(0.9, 1.0), 1)
        # fake = 0.0
        # real = 1.0
        # label
        threashold = random.uniform(0.0, 1.0)
        if threashold < 0.05:
            r_fake = real
            f_real = fake
        else:
            r_fake = fake
            f_real = real
        self.set_requires_grad(self.discriminator, True)
        pred_fake = self.discriminator(self.p_t_1.detach())
        loss_D_fake = self.loss(pred_fake, r_fake)
        pred_real = self.discriminator(self.g_t_1)
        loss_D_real = self.loss(pred_real, f_real)
        loss_D = (loss_D_fake + loss_D_real) * 0.5


        if self.key % self.p_s == 0:
            self.key = 1
            b_s = self.p_t_1.shape[0]
            # plt.figure(figsize=(6, 2))
            big_img = None
            for i in range(0, b_s):
                img_g = self.g_t_1[i].squeeze(0).cpu().permute(1, 2, 0).numpy()
                img_p = self.p_t_1[i].squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
                img_g = mmcv.imdenormalize(img_g, 127.5, 127.5, to_bgr=False)
                img_p = mmcv.imdenormalize(img_p, 127.5, 127.5, to_bgr=False)
                img = np.hstack([img_g, img_p])
                if big_img is None:
                    big_img = img
                else:
                    big_img = np.vstack([big_img, img])
                # plt.subplot(i + 1, 1)
                # plt.plot(img_g)
                # plt.subplot(i + 1, 2)
                # plt.plot(img_p)
            name = str(int(time.time())) + '.jpg'
            mmcv.imwrite(big_img, os.path.join('./img', name))
        self.key += 1
            # plt.savefig()
            # plt.show()





        return {'loss_D': loss_D}
    def backward_G(self):
        # real = round(random.uniform(0.9, 1.0), 1)
        real = 1.0
        self.set_requires_grad(self.discriminator, False)
        pred_fake = self.discriminator(self.p_t_1)
        loss_G_GAN = self.loss(pred_fake, real) * 0.5
        loss_int = self.l2loss(self.p_t_1, self.g_t_1)
        loss_gd = self.gradientloss([self.p_t_1, self.g_t_1])
        loss_OP = self.l1loss(self.real_f, self.fake_f)
        loss_G = self.opt.train_cfg.lambda_int * loss_int + self.opt.train_cfg.lambda_gd * loss_gd + \
                 self.opt.train_cfg.lambda_op * loss_OP + self.opt.train_cfg.lambda_adv * loss_G_GAN
        return {'loss_G_GAN': loss_G_GAN, 'loss_int': loss_int, 'loss_gd': loss_gd, 'loss_OP': loss_OP, 'loss_G': loss_G}




















