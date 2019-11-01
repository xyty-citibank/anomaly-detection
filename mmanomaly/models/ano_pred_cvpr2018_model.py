from mmanomaly.models.g_d_networks import *
from mmanomaly.models.loss import GANLoss, GradientLoss
from mmanomaly.models.base_model import BaseModel
"""
This class implements the ano_pred_cvpr2018 model, the paper's url:1712.09867.pdf
"""
class APCModel(BaseModel):
    def __init__(self, opt):
        super(APCModel, self).__init__()
        self.opt = opt
        self.generator = UNetGenerator()
        self.discriminator = NLayerDiscriminator()
        self.flownet = None
        self.loss = GANLoss(opt.gan_mode)
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()
        self.gradientloss = GradientLoss(opt.alpha)



    def forward(self, input):
        self.real_A, self.real_B, self.real_C = input
        self.fake_C = self.generator(self.real_B)
        self.real_f_C = self.flownet(self.real_C)
        self.fake_f_C = self.flownet(self.fake_C)

    def backward_D(self):
        loss = dict()
        self.set_requires_grad(self.discriminator, True)
        fake_AC = torch.cat((self.real_A, self.fake_C), 1)
        pred_fake = self.discriminator(fake_AC.detach())
        loss_D_fake = self.loss(pred_fake, False)
        real_AC = torch.cat((self.real_A, self.real_C), 1)
        pred_real = self.discriminator(real_AC)
        loss_D_real = self.loss(pred_real, True)
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss.update(loss_D)
        return loss

    def backward_G(self):
        loss = dict()
        self.set_requires_grad(self.discriminator, False)
        fake_AC = torch.cat((self.real_A, self.fake_C), 1)
        pred_fake = self.discriminator(fake_AC)
        loss_G_GAN = self.loss(pred_fake, True)
        loss_int = self.l2loss(self.fake_C, self.real_C)
        loss_gd = self.gradientloss([self.fake_C, self.real_C])
        loss_OP = self.l1loss(self.real_f_C, self.fake_f_C)
        loss_G = self.opt.lambda_int * loss_int + self.opt.lambda_gd * loss_gd + \
                 self.opt.lambda_op * loss_OP + self.opt.lambda_adv * loss_G_GAN
        loss.update(loss_G)
        return loss


















