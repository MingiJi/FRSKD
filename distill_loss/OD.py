import math
import torch
import torch.nn as nn
from scipy.stats import norm
import torch.nn.functional as F
from .KD import DistillKL

def od(args, bifpn):
    return OD(args, bifpn)


def conv1x1(in_planes, out_planes):
    C = [nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(out_planes)]
    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)


class OD(nn.Module):
    def __init__(self, args, bifpn):
        super(OD, self).__init__()
        self.p = 2
        self.kd = DistillKL(args)
        self.alpha = args.alpha
        self.beta = args.beta
        self.linear = nn.ModuleList()
        for feat, feat_out in zip(args.network_channels, bifpn.net_channels):
            self.linear.append(conv1x1(feat, feat_out))

        if torch.cuda.device_count() > 1:
            teacher_bns = bifpn.module.get_bn_before_relu()
        else:
            teacher_bns = bifpn.get_bn_before_relu()

        margins = [self.get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i + 1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

    def forward(self, o_s, o_t, g_s, g_t):
        loss = self.alpha * self.kd(o_s, o_t.detach())
        feat_num = len(g_s)
        for i, (f_s, f_t) in enumerate(zip(g_s, g_t)):
            loss += self.beta * self.distill_loss(f_s, f_t.detach(), i) / 2 ** (feat_num - i - 1)
        return loss

    def distill_loss(self, source, target, i):
        margin = getattr(self, 'margin%d' % (i+1))
        source_feat = self.linear[i](source)
        target = torch.max(target, margin)
        loss = F.mse_loss(source_feat, target, reduction="none") / source.size(0)
        loss = loss * ((source_feat > target) | (target > 0)).float()
        return loss.sum()

    def get_margin_from_BN(self, bn):
        margin = []
        std = bn.weight.data
        mean = bn.bias.data
        for (s, m) in zip(std, mean):
            s = abs(s.item())
            m = m.item()
            if norm.cdf(-m / s) > 0.001:
                margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
            else:
                margin.append(-3 * s)

        return torch.FloatTensor(margin).to(std.device)
