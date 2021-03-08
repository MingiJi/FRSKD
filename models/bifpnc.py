import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import MaxPool2dStaticSamePadding, DepthConvBlock

__all__ = ['BiFPNc']


class BiFPNc(nn.Module):
    def __init__(self, network_channel, num_classes, args):
        super(BiFPNc, self).__init__()
        repeat = args.repeat
        depth = args.depth
        width = args.width

        self.num_features = args.num_features
        self.layers = nn.ModuleList()

        self.net_channels = [x * args.width for x in network_channel]
        for i in range(repeat):
            self.layers.append(BiFPN_layer(i == 0, DepthConvBlock, network_channel, depth, width))

        self.fc = nn.Linear(self.net_channels[-1], num_classes)

    def forward(self, feats, preact=False):
        feats = feats[-self.num_features:]

        for i in range(len(self.layers)):
            layer_preact = preact and i == len(self.layers) - 1
            feats = self.layers[i](feats, layer_preact)

        out = F.adaptive_avg_pool2d(F.relu(feats[-1]), (1, 1)) # for preact case
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return feats, out

    def get_bn_before_relu(self):
        layer = self.layers[-1]
        bn = [layer.up_conv[0].conv[-1][-1]]
        for down_conv in layer.down_conv:
            bn.append(down_conv.conv[-1][-1])
        return bn


class BiFPN_layer(nn.Module):
    def __init__(self, first_time, block, network_channel, depth, width):
        super(BiFPN_layer, self).__init__()
        lat_depth, up_depth, down_depth = depth
        self.first_time = first_time

        self.lat_conv = nn.ModuleList()
        self.lat_conv2 = nn.ModuleList()

        self.up_conv = nn.ModuleList()
        self.up_weight = nn.ParameterList()

        self.down_conv = nn.ModuleList()
        self.down_weight = nn.ParameterList()
        self.down_sample = nn.ModuleList()
        self.up_sample = nn.ModuleList()

        for i, channels in enumerate(network_channel):
            if self.first_time:
                self.lat_conv.append(block(channels, channels * width, 1, 1, 0, lat_depth))

            if i != 0:
                self.lat_conv2.append(block(channels, channels * width, 1, 1, 0, lat_depth))
                self.down_conv.append(block(channels * width, channels * width, 3, 1, 1, down_depth))
                num_input = 3 if i < len(network_channel) - 1 else 2

                self.down_weight.append(nn.Parameter(torch.ones(num_input, dtype=torch.float32), requires_grad=True))
                self.down_sample.append(nn.Sequential(MaxPool2dStaticSamePadding(3, 2),
                                                      block(network_channel[i-1] * width, channels * width, 1, 1, 0, 1)))

            if i != len(network_channel) - 1:
                self.up_sample.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                                    block(network_channel[i+1] * width, channels * width, 1, 1, 0, 1)))
                self.up_conv.append(block(channels * width, channels * width, 3, 1, 1, up_depth))
                self.up_weight.append(nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True))

        self.relu = nn.ReLU()

        self.epsilon = 1e-6

    def forward(self, inputs, preact=False):
        input_trans = [self.lat_conv2[i - 1](F.relu(inputs[i])) for i in range(1, len(inputs))]
        if self.first_time:
            inputs = [self.lat_conv[i](F.relu(inputs[i])) for i in range(len(inputs))] # for od case

        # up
        up_sample = [inputs[-1]]
        out_layer = []
        for i in range(1, len(inputs)):
            w = self.relu(self.up_weight[-i])
            w = w / (torch.sum(w, dim=0) + self.epsilon)

            up_sample.insert(0,
                             self.up_conv[-i](w[0] * F.relu(inputs[-i - 1])
                                              + w[1] * self.up_sample[-i](F.relu(up_sample[0]))))

        out_layer.append(up_sample[0])

        # down
        for i in range(1, len(inputs)):
            w = self.relu(self.down_weight[i - 1])
            w = w / (torch.sum(w, dim=0) + self.epsilon)
            if i < len(inputs) - 1:
                out_layer.append(self.down_conv[i - 1](w[0] * F.relu(input_trans[i - 1])
                                                       + w[1] * F.relu(up_sample[i])
                                                       + w[2] * self.down_sample[i - 1](F.relu(out_layer[-1]))
                                                       )
                                 )
            else:
                out_layer.append(
                    self.down_conv[i - 1](w[0] * F.relu(input_trans[i - 1])
                                          + w[1] * self.down_sample[i - 1](F.relu(out_layer[-1]))
                                          )
                )

        if not preact:
            return [F.relu(f) for f in out_layer]
        return out_layer
