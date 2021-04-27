import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    def __init__(self, args):
        super(DistillKL, self).__init__()
        self.T = args.temperature

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t.detach(), reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss
