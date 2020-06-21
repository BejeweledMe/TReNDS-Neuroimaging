from torch import nn
import torch


class W_NAE(nn.Module):
    def __init__(self, w=[0.3, 0.175, 0.175, 0.175, 0.175]):
        super().__init__()
        self.w = torch.FloatTensor(w)

    def forward(self, output, target):
        if not (target.size() == output.size()):
            raise ValueError('Target size ({}) must be the same as input size ({})'
                             .format(target.size(), output.size()))
        loss = torch.sum(
            self.w * torch.sum(torch.abs(target - output), axis=0) / torch.sum(target, axis=0)
        )

        return loss



losses_dict = {
    'mae': nn.L1Loss(),
    'w_nae': W_NAE(),
}


def loss_function(loss):
    criterion = losses_dict[loss]
    return criterion