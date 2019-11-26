
import torch
import numpy as np
import torch.nn as nn

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def __call__(self, prediction, target):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = torch.tensor(target).cuda()
            target_tensor = target_tensor.expand_as(prediction)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class GradientLoss(nn.Module):
    def __init__(self, alpha):
        super(GradientLoss, self).__init__()
        self.conv2d_w = nn.Conv2d(3, 3, (1, 2))
        self.conv2d_h = nn.Conv2d(3, 3, (2, 1))
        self.alpha = alpha
        self.init_weights()
        for p in self.parameters():
            p.requires_grad = False
    def forward(self, input):
        fake_C, real_C = input
        gen_dw = torch.abs(self.conv2d_w(fake_C))
        gen_dh = torch.abs(self.conv2d_h(fake_C))
        gt_dw = torch.abs(self.conv2d_w(real_C))
        gt_dh = torch.abs(self.conv2d_h(real_C))
        grad_diff_w = torch.abs(gt_dw - gen_dw)
        grad_diff_h = torch.abs(gt_dh - gen_dh)
        return torch.mean(grad_diff_w ** self.alpha) + torch.mean(grad_diff_h ** self.alpha)
    def init_weights(self):
        w_p = np.ones([3, 3, 1, 2])
        h_p = np.ones([3, 3, 2, 1])
        w_p[:, :, :, 1] = -1
        h_p[:, :, 1, :] = -1
        w_p = torch.Tensor(w_p).float()
        h_p = torch.Tensor(h_p).float()
        self.conv2d_w.weight = torch.nn.Parameter(w_p)
        self.conv2d_h.weight = torch.nn.Parameter(h_p)




