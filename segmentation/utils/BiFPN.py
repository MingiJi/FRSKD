from torch import nn
from torch.nn import functional as F


class BiFPNSeg(nn.Module):
    def __init__(self, num_channels, is_last=False):
        super(BiFPNSeg, self).__init__()
        self.num_channels = num_channels
        self.is_last = is_last
        out_channels = num_channels

        self.conv7up = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels),
            nn.BatchNorm2d(num_features=out_channels), nn.ReLU())
        self.conv6up = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels),
            nn.BatchNorm2d(num_features=out_channels), nn.ReLU())
        self.conv5up = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels),
            nn.BatchNorm2d(num_features=out_channels), nn.ReLU())
        self.conv4up = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels),
            nn.BatchNorm2d(num_features=out_channels), nn.ReLU())
        self.conv3up = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels),
            nn.BatchNorm2d(num_features=out_channels), nn.ReLU())
        self.conv2up = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels),
            nn.BatchNorm2d(num_features=out_channels), nn.ReLU())

        if not is_last:
            self.conv3dw = nn.Sequential(
                nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels),
                nn.BatchNorm2d(num_features=out_channels), nn.ReLU())
            self.conv4dw = nn.Sequential(
                nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels),
                nn.BatchNorm2d(num_features=out_channels), nn.ReLU())
            self.conv5dw = nn.Sequential(
                nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels),
                nn.BatchNorm2d(num_features=out_channels), nn.ReLU())
            self.conv6dw = nn.Sequential(
                nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels),
                nn.BatchNorm2d(num_features=out_channels), nn.ReLU())
            self.conv7dw = nn.Sequential(
                nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, groups=num_channels),
                nn.BatchNorm2d(num_features=out_channels), nn.ReLU())

    def forward(self, inputs):
        num_channels = self.num_channels
        P2_in, P3_in, P4_in, P5_in, P6_in, P7_in = inputs

        # upsample network
        P7_up = self.conv7up(P7_in)
        scale = (P6_in.size(3) / P7_up.size(3))
        P6_up = self.conv6up(P6_in + self.Resize(scale_factor=scale)(P7_up))
        scale = (P5_in.size(3) / P6_up.size(3))
        P5_up = self.conv5up(P5_in + self.Resize(scale_factor=scale)(P6_up))
        scale = (P4_in.size(3) / P5_up.size(3))
        P4_up = self.conv4up(P4_in + self.Resize(scale_factor=scale)(P5_up))
        scale = (P3_in.size(3) / P4_up.size(3))
        P3_up = self.conv3up(P3_in + self.Resize(scale_factor=scale)(P4_up))
        scale = (P2_in.size(3) / P3_up.size(3))
        P2_out = self.conv2up(P2_in + self.Resize(scale_factor=scale)(P3_up))

        if not self.is_last:
            # fix to downsample by interpolation
            # downsample networks
            P3_out = self.conv3dw(P3_in + P3_up + F.interpolate(P2_out, P3_up.size()[2:]))
            P4_out = self.conv4dw(P4_in + P4_up + F.interpolate(P3_out, P4_up.size()[2:]))
            P5_out = self.conv5dw(P5_in + P5_up + F.interpolate(P4_out, P5_up.size()[2:]))
            P6_out = self.conv6dw(P6_in + P6_up + F.interpolate(P5_out, P6_up.size()[2:]))
            P7_out = self.conv7dw(P7_in + P7_up + F.interpolate(P6_out, P7_up.size()[2:]))
            return P2_out, P3_out, P4_out, P5_out, P6_out, P7_out

        if self.is_last:
            return P2_out, P3_up, P4_up, P5_up, P6_up, P7_up

    @staticmethod
    def Conv(in_channels, out_channels, kernel_size, stride, padding, groups=1):
        features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=groups),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        return features

    @staticmethod
    def Resize(scale_factor=2, mode='bilinear'):
        upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        return upsample