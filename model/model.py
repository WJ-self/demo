import torch
from torch import nn

from .unet import UNetRecurrent
from .ediff.diffusion import GaussianDiffusion
from .ediff.unet import UNet as UNet_ediff


class E2VIDRecurrent(nn.Module):
    """
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """

    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetrecurrent = UNetRecurrent(unet_kwargs)
        self.prev_recs = None

    def reset_states(self):
        self.unetrecurrent.states = [None] * self.unetrecurrent.num_encoders
        self.prev_recs = None

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        if self.prev_recs is None:
            self.prev_recs = torch.zeros(event_tensor.shape[0], 1, event_tensor.shape[2], event_tensor.shape[3],
                                         device=event_tensor.device)
        output_dict = self.unetrecurrent.forward(event_tensor, self.prev_recs)
        self.prev_recs = output_dict['image'].detach()
        return output_dict

class ediff(nn.Module):
    '''Event-based Camera Image Reconstruction via conditional Diffusion (SR3)
    '''
    def __init__(self, unet_kwargs, steps=1000, sample_steps=1000):
        super().__init__()
        self.model = UNet_ediff(**unet_kwargs)
        self.diffusion = GaussianDiffusion(self.model, steps, sample_steps)

    def reset_states(self):
        pass

    def forward(self,img_X0, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :param img_X0: N x 3 x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        loss = self.diffusion(img_X0, event_tensor)
        return loss

    def sample(self, img_XT, event_tensor):
        return {'image':self.diffusion.sample(img_XT, event_tensor)}