import torch
import torch.nn as nn
import torch.nn.functional as F
from classification.distill_loss.KD import DistillKL


def sla(args):
    return SLALoss(args)

def rotation(images):
    size = images.shape[1:]
    return torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1).view(-1, *size)


class SLALoss(nn.Module):
    def __init__(self, args):
        super(SLALoss, self).__init__()
        self.args = args
        self.kd = DistillKL(args)
        self.ce = nn.CrossEntropyLoss()
        self.aux_lamb = args.aux_lamb
        self.fc1 = nn.Linear(args.network_channels[-1], 4 * args.num_classes)
        self.fc2 = nn.Linear(args.network_channels[-1] * args.width, 4 * args.num_classes)

    def forward(self, model, bifpn, inputs, targets):
        bs = inputs.size(0)
        rot_inputs = rotation(inputs)

        feats, outputs = model(inputs)
        feats = feats[-self.args.num_features:]
        bi_feats, bi_outputs = bifpn(feats, self.args.distill == 'od')

        rot_feats, _ = model(rot_inputs)
        rot_feats = rot_feats[-self.args.num_features:]
        rot_bi_feats, _ = bifpn(rot_feats, self.args.distill == 'od')

        last_feat = F.adaptive_avg_pool2d(rot_feats[-1], (1, 1)).view(4 * bs, -1)
        aux_outputs = self.fc1(last_feat)
        rot_last_feat = F.adaptive_avg_pool2d(rot_bi_feats[-1], (1, 1)).view(4 * bs, -1)
        biaux_outputs = self.fc2(rot_last_feat)

        single_loss = self.ce(outputs, targets) + self.ce(bi_outputs, targets)
        aux_targets = torch.stack([targets * 4 + i for i in range(4)], 1).view(-1)
        joint_loss = self.ce(aux_outputs, aux_targets) + self.ce(biaux_outputs, aux_targets)

        aux_outputs = torch.stack([aux_outputs[i::4, i::4] for i in range(4)], 1).mean(1)
        biaux_outputs = torch.stack([biaux_outputs[i::4, i::4] for i in range(4)], 1).mean(1)

        sla_loss = self.kd(outputs, aux_outputs) + self.kd(bi_outputs, biaux_outputs)
        loss = single_loss + joint_loss + self.aux_lamb * sla_loss

        outputs = torch.cat([outputs, aux_outputs], dim=0)
        bi_outputs = torch.cat([bi_outputs, biaux_outputs], dim=0)
        feats = [torch.cat([f, rf], dim=0) for f, rf in zip(feats, rot_feats)]
        bi_feats = [torch.cat([f, rf], dim=0) for f, rf in zip(bi_feats, rot_bi_feats)]
        return outputs, bi_outputs, feats, bi_feats, loss