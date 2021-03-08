import math
import torch.nn as nn
from .KD import DistillKL

def fit(args, bifpn):
    return HintLoss(args)

def conv1x1_act(in_planes, out_planes):
    C = [nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(out_planes),
         nn.ReLU()]
    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)



class HintLoss(nn.Module):
    def __init__(self, args):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()
        self.linear = nn.modules.ModuleList()
        for feat in args.network_channels:
            if args.bifpn == 'BiFPN':
                self.linear.append(conv1x1_act(feat, args.num_channels))
            else:
                self.linear.append(conv1x1_act(feat, args.width * feat))
        self.kd = DistillKL(args)
        self.alpha = args.alpha
        self.beta = args.beta

    def forward(self, o_s, o_t, g_s, g_t):
        loss = self.alpha * self.kd(o_s, o_t)
        loss += self.beta * sum([self.fit_loss(f_s, f_t.detach(), i) for i, (f_s, f_t) in enumerate(zip(g_s, g_t))])
        return loss

    def fit_loss(self, f_s, f_t, i):
        loss = self.crit(self.linear[i](f_s), f_t)
        return loss
