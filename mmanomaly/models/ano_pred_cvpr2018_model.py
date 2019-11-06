from mmanomaly.models.g_d_networks import *
from mmanomaly.models.loss import GANLoss, GradientLoss
from mmanomaly.models.base_model import BaseModel
from mmanomaly.FlowNetPytorch.flownet import FlowNet
"""
This class implements the ano_pred_cvpr2018 model, the paper's url:1712.09867.pdf
"""
class APCModel(BaseModel):
    def __init__(self, opt):
        super(APCModel, self).__init__()
        self.opt = opt
        self.generator = UNetGenerator()
        self.discriminator = NLayerDiscriminator()
        self.flownet = FlowNet(opt.train.flownet_pretrained)
        self.loss = GANLoss(opt.gan_mode)
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()
        self.gradientloss = GradientLoss(opt.alpha)
        self.batch_size = opt.gpus * opt.data.videos_per_gpu
        self.num_pred = opt.data.train.num_pred,
        self.time_steps = opt.data.train.time_steps
        self.c, self.w, self.h = opt.data.train.scale_size

    def forward(self, input):
        input = input.resize(self.batch_size, self.w, self.h, self.c * (self.time_steps * self.num_pred))
        self.g_t = input[..., self.c * self.time_steps]
        self.g_t_1 = input[..., self.c * self.num_pred]
        self.p_t_1 = self.generator(self.g_t)
        self.real_f = self.flownet([self.g_t, self.g_t_1])
        self.fake_f = self.flownet([self.g_t, self.p_t_1])

    def backward_D(self):
        loss = dict()
        self.set_requires_grad(self.discriminator, True)
        pred_fake = self.discriminator(self.p_t_1.detach())
        loss_D_fake = self.loss(pred_fake, False)
        pred_real = self.discriminator(self.g_t_1)
        loss_D_real = self.loss(pred_real, True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss.update(loss_D)
        return loss

    def backward_G(self):
        loss = dict()
        self.set_requires_grad(self.discriminator, False)
        pred_fake = self.discriminator(self.p_t_1)
        loss_G_GAN = self.loss(pred_fake, True)
        loss_int = self.l2loss(self.p_t_1, self.g_t_1)
        loss_gd = self.gradientloss([self.p_t_1, self.g_t_1])
        loss_OP = self.l1loss(self.real_f, self.fake_f)
        loss_G = self.opt.lambda_int * loss_int + self.opt.lambda_gd * loss_gd + \
                 self.opt.lambda_op * loss_OP + self.opt.lambda_adv * loss_G_GAN
        loss.update(loss_G)
        return loss


















