from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch
import torch.nn.functional as F
from .BiFPN import BiFPNSeg


class EfficientDet(nn.Module):
    def __init__(self, backbone="efficientnet-b0"):
        super(EfficientDet, self).__init__()
        model = EfficientNet.from_pretrained(backbone)

        self.layer0 = nn.Sequential(model._conv_stem, model._bn0)
        if backbone == "efficientnet-b0":
            self.layer1 = nn.Sequential(model._blocks[0],model._blocks[1])
            self.layer2 = nn.Sequential(model._blocks[2],model._blocks[3])
            self.layer3 = nn.Sequential(model._blocks[4],model._blocks[5])
            self.layer4 = nn.Sequential(model._blocks[6],model._blocks[7],model._blocks[8],model._blocks[9],model._blocks[10],model._blocks[11])
            self.layer5 = nn.Sequential(model._blocks[12],model._blocks[13],model._blocks[14],model._blocks[15])
        else:
            self.layer1 = nn.Sequential(model._blocks[0],model._blocks[1],model._blocks[2])
            self.layer2 = nn.Sequential(model._blocks[3],model._blocks[4],model._blocks[5])
            self.layer3 = nn.Sequential(model._blocks[6],model._blocks[7],model._blocks[8])
            self.layer4 = nn.Sequential(model._blocks[9],model._blocks[10],model._blocks[11])
            self.layer5 = nn.Sequential(model._blocks[12],model._blocks[13],model._blocks[14],model._blocks[15]
                                        ,model._blocks[16],model._blocks[17],model._blocks[18],model._blocks[19]
                                        ,model._blocks[20],model._blocks[21],model._blocks[22])
            
        outc_candi = {"efficientnet-b0":64, "efficientnet-b1": 88, "efficientnet-b2": 112}
        outc = outc_candi[backbone]

        # Bottom-up layers

        self.conv6 = self.Conv( self.layer5[-1]._project_conv.weight.size()[0], outc, kernel_size=3, stride=2, padding=1)
        self.conv7 = self.Conv(outc, outc, kernel_size=3, stride=2, padding=1)
        # Top layer
        self.toplayer = self.Conv(self.layer5[-1]._project_conv.weight.size()[0], outc, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth1 = self.Conv(outc, outc, kernel_size=3, stride=1, padding=1)
        self.smooth2 = self.Conv(outc, outc, kernel_size=3, stride=1, padding=1)        
        # Lateral layers
        self.latlayer1 = self.Conv(self.layer3[-1]._project_conv.weight.size()[0], outc, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = self.Conv(self.layer2[-1]._project_conv.weight.size()[0], outc, kernel_size=1, stride=1, padding=0)
        # loc, conf layers
        self.latlayer3 = self.Conv(self.layer1[-1]._project_conv.weight.size()[0], outc, kernel_size=1, stride=1, padding=0)
        self.BiFPN1 = nn.Sequential(BiFPNSeg(outc), BiFPNSeg(outc), BiFPNSeg(outc))
        self.BiFPN2 = nn.Sequential(BiFPNSeg(outc), BiFPNSeg(outc, True))

        self.classifier1 = nn.Sequential(nn.Conv2d(outc, 256, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(256, eps=1e-4, momentum=0.997),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 21, kernel_size=3, padding=1))
        self.classifier2 = nn.Sequential(nn.Conv2d(outc, 256, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(256, eps=1e-4, momentum=0.997),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 21, kernel_size=3, padding=1))

    def forward(self, x):
        _, _, H, W = x.size()
        x = self.layer0(x)
        p2 = self.layer1(x)
        p3 = self.layer2(p2)
        p4 = self.layer3(p3)
        p5 = self.layer4(p4)
        p5 = self.layer5(p5)

        p6 = self.conv6(p5)
        p7 = self.conv7(p6)

        p5 = self.toplayer(p5)
        p4 = self._upsample_add(p5, self.latlayer1(p4))
        p3 = self._upsample_add(p4, self.latlayer2(p3))
        p2 = self._upsample_add(p3, self.latlayer3(p2))
        sources = [p2, p3, p4, p5, p6, p7]

        sources1 = self.BiFPN1(sources)
        sources2 = self.BiFPN2(sources1)

        output1 = self.classifier1(sources1[0])
        output2 = self.classifier2(sources2[0])

        output1 = F.upsample(output1, size=(H, W), mode='bilinear')
        output2 = F.upsample(output2, size=(H, W), mode='bilinear')
        return output1, output2, sources1, sources2

    @staticmethod
    def Conv(in_channels, out_channels, kernel_size, stride, padding, groups=1):
        features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm2d(num_features=out_channels, eps=1e-4, momentum=0.997),
            nn.ReLU(inplace=True)
        )
        return features 
    
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y
