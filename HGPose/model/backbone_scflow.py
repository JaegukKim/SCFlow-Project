import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34
from HGPose.model.convnext import convnext_tiny
from collections import OrderedDict

class ResNet18(nn.Module):
    def __init__(self, input_dim=3):
        super(ResNet18, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.input_dim = input_dim
        self.resnet.conv1 = nn.Conv2d(self.input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.block_feature_dim = {'feat1':128, 'feat2':256, 'feat3':512}
    def forward(self, x):
        f = OrderedDict()
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)  # (B, C, H/2, W/2)
        f['feat1'] = self.resnet.layer2(x)
        f['feat2'] = self.resnet.layer3(f['feat1'])
        f['feat3'] = self.resnet.layer4(f['feat2'])
        return f

class ResNet34(nn.Module):
    def __init__(self, input_dim=3):
        super(ResNet34, self).__init__()
        self.model = nn.Sequential(*(list(resnet34(pretrained=True, ).children())[:-2]))
        self.model[0] = nn.Conv2d(input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.block_feature_dim = {'feat1':128, 'feat2':256, 'feat3':512}
    def forward(self, x):
        f = OrderedDict()
        x = self.model[0](x)
        x = self.model[1](x)
        x = self.model[2](x)
        x = self.model[3](x)
        x = self.model[4](x)
        f['feat1'] = self.model[5](x)
        f['feat2'] = self.model[6](f['feat1'])
        f['feat3'] = self.model[7](f['feat2'])
        return f

class ConvNextTiny(nn.Module):
    def __init__(self, input_dim=3):
        super(ConvNextTiny, self).__init__()
        self.model = convnext_tiny(pretrained=True, in_22k=True, num_classes=21841)
        delattr(self.model, 'norm')
        delattr(self.model, 'head')
        self.model.downsample_layers[0][0] = nn.Conv2d(input_dim, 96, kernel_size=(4, 4), stride=(4, 4))
        self.block_feature_dim = {'feat1':192, 'feat2':384, 'feat3':768}
    def forward(self, x):
        f = OrderedDict()
        x = self.model.downsample_layers[0](x)
        x = self.model.stages[0](x)
        x = self.model.downsample_layers[1](x)
        x = self.model.stages[1](x)
        f['feat1'] = x
        x = self.model.downsample_layers[2](x)
        x = self.model.stages[2](x)
        f['feat2'] = x
        x = self.model.downsample_layers[3](x)  # 768 x 8 x 8
        x = self.model.stages[3](x)
        f['feat3'] = x
        return f



class UpsampleHead(nn.Module):
    def __init__(self, feature_dim=None, coord_dim=None):
        super(UpsampleHead, self).__init__()
        self.feature_dim = feature_dim
        self.coord_dim = coord_dim
        self.m = nn.ModuleDict({
            'feat3': nn.Sequential(            # 8x8 -> 16x16
                nn.ConvTranspose2d(self.feature_dim['feat3'], self.feature_dim['feat3'], 3, 2, 1, 1, bias=False),
                nn.BatchNorm2d(self.feature_dim['feat3']),
                nn.GELU(),
                nn.Conv2d(self.feature_dim['feat3'], self.feature_dim['feat3'], 3, 1, 1),
                nn.BatchNorm2d(self.feature_dim['feat3']),
                nn.GELU(),
                nn.Conv2d(self.feature_dim['feat3'], self.feature_dim['feat2'], 3, 1, 1),
                nn.BatchNorm2d(self.feature_dim['feat2']),
                nn.GELU()),
            'feat2': nn.Sequential(            # 16x16 -> 32x32
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(self.feature_dim['feat2'], self.feature_dim['feat2'], 3, 1, 1),
                nn.BatchNorm2d(self.feature_dim['feat2']),
                nn.GELU(),
                nn.Conv2d(self.feature_dim['feat2'], self.feature_dim['feat1'], 3, 1, 1),
                nn.BatchNorm2d(self.feature_dim['feat1']),
                nn.GELU()),
            'feat1': nn.Sequential(            # 32x32 -> 64x64
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(self.feature_dim['feat1'], self.feature_dim['feat1'], 3, 1, 1),
                nn.BatchNorm2d(self.feature_dim['feat1']),
                nn.GELU(),
                nn.Conv2d(self.feature_dim['feat1'], self.feature_dim['feat1'], 3, 1, 1),
                nn.BatchNorm2d(self.feature_dim['feat1']),
                nn.GELU()),
            })
        self.coord_head = nn.Sequential(nn.Conv2d(self.feature_dim['feat1'], self.coord_dim, 3, 1, 1), nn.Tanh())
        self.mask_head = nn.Sequential(nn.Conv2d(self.feature_dim['feat1'], 1, 3, 1, 1), nn.Sigmoid())
        self.mask_visib_head = nn.Sequential(nn.Conv2d(self.feature_dim['feat1'], 1, 3, 1, 1), nn.Sigmoid())
        #self.error_head = nn.Sequential(nn.Conv2d(self.feature_dim['feat1'], self.coord_dim, 3, 1, 1), nn.Tanh())

    def forward(self, backbone_feature, id_feature=None):
        x = self.m['feat3'](backbone_feature['feat3'])
        x = x + backbone_feature['feat2']
        x = self.m['feat2'](x)
        x = x + backbone_feature['feat1']
        x = self.m['feat1'](x)
        coord = self.coord_head(x)
        mask = self.mask_head(x)
        mask_visib = self.mask_visib_head(x)
        # error = self.error_head(x)
        return coord, mask, mask_visib

class UpsampleHeadProb(nn.Module):
    def __init__(self, feature_dim=None, coord_dim=None):
        super(UpsampleHeadProb, self).__init__()
        self.feature_dim = feature_dim
        self.coord_dim = coord_dim
        self.m = nn.ModuleDict({
            'feat3': nn.Sequential(            # 8x8 -> 16x16
                nn.ConvTranspose2d(self.feature_dim['feat3'], self.feature_dim['feat3'], 3, 2, 1, 1, bias=False),
                nn.BatchNorm2d(self.feature_dim['feat3']),
                nn.GELU(),
                nn.Conv2d(self.feature_dim['feat3'], self.feature_dim['feat3'], 3, 1, 1),
                nn.BatchNorm2d(self.feature_dim['feat3']),
                nn.GELU(),
                nn.Conv2d(self.feature_dim['feat3'], self.feature_dim['feat2'], 3, 1, 1),
                nn.BatchNorm2d(self.feature_dim['feat2']),
                nn.GELU()),
            'feat2': nn.Sequential(            # 16x16 -> 32x32
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(self.feature_dim['feat2'], self.feature_dim['feat2'], 3, 1, 1),
                nn.BatchNorm2d(self.feature_dim['feat2']),
                nn.GELU(),
                nn.Conv2d(self.feature_dim['feat2'], self.feature_dim['feat1'], 3, 1, 1),
                nn.BatchNorm2d(self.feature_dim['feat1']),
                nn.GELU()),
            'feat1': nn.Sequential(                  # 32x32 -> 32x32
                # nn.UpsamplingBilinear2d(scale_factor=2),# 32x32 -> 64x64   
                nn.Conv2d(self.feature_dim['feat1'], self.feature_dim['feat1'], 3, 1, 1),
                nn.BatchNorm2d(self.feature_dim['feat1']),
                nn.GELU(),
                nn.Conv2d(self.feature_dim['feat1'], self.feature_dim['feat1'], 3, 1, 1),
                nn.BatchNorm2d(self.feature_dim['feat1']),
                nn.GELU()),
            })
        self.coord_head = nn.Sequential(nn.Conv2d(self.feature_dim['feat1'], self.coord_dim, 3, 1, 1), nn.Tanh())
        self.mask_head = nn.Sequential(nn.Conv2d(self.feature_dim['feat1'], 1, 3, 1, 1), nn.Sigmoid())
        self.mask_visib_head = nn.Sequential(nn.Conv2d(self.feature_dim['feat1'], 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, backbone_feature, id_feature=None):
        x = self.m['feat3'](backbone_feature['feat3'])
        x = x + backbone_feature['feat2']
        x = self.m['feat2'](x)
        x = x + backbone_feature['feat1']
        x = self.m['feat1'](x)
        coord = self.coord_head(x)
        mask = self.mask_head(x)
        mask_visib = self.mask_visib_head(x)
        return coord, mask, mask_visib


class PoseHead(nn.Module):
    def __init__(self, feature_dim=None, size=None):
        super(PoseHead, self).__init__()
        self.feature_dim = feature_dim
        self.feature_size = [size[0] // 8, size[1] // 8]
        self.model = nn.Sequential(
            nn.Conv2d(self.feature_dim, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(1, 3),
            nn.Linear(128 * self.feature_size[0] * self.feature_size[1], 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.1))
        self.trans_fc = nn.Linear(256, 3, bias=True)
        self.trans_fc.weight.data = nn.Parameter(torch.zeros_like(self.trans_fc.weight.data))
        self.trans_fc.bias.data = nn.Parameter(torch.Tensor([0,0,0]))
        self.rotat_fc = nn.Linear(256, 6, bias=True)
        self.rotat_fc.weight.data = nn.Parameter(torch.zeros_like(self.rotat_fc.weight.data))
        self.rotat_fc.bias.data = nn.Parameter(torch.Tensor([1,0,0,0,1,0]))

    def forward(self, f):
        encoded = self.model(f)
        rotation = self.rotat_fc(encoded)
        translation = self.trans_fc(encoded)
        result = torch.cat([rotation, translation], -1)
        return result
