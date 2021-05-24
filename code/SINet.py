import torch
import torch.nn as nn
import torchvision.models as models
from search_attention import SA
from ResNet_backbone import ResNet_2Branch


# Defining a basic Bconv block having conv_layer, batchnorm and relu
class Bconv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1):
    super(Bconv, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding, dilation = dilation, bias = False)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(out_channels)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    return x

# Receptive field module
# Contains total 5 branches
class R_F(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(R_F, self).__init__()
    self.relu = nn.ReLU(True)
     
    # branch comprising only bconv 
    self.branch_0 = nn.Sequential(Bconv(in_channels, out_channels, 1))

    # branch comprising bconv of 1x1 to reduce size
    # it is followed by bconv of 1x3, 3x1 and dilation of 3
    self.branch_1 = nn.Sequential(
                    Bconv(in_channels, out_channels, 1),
                    Bconv(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
                    Bconv(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
                    Bconv(out_channels, out_channels, 3, padding = 3, dilation = 3)
    )

    # branch comprising bconv of 1x1 to reduce size
    # it is followed by bconv of 1x5, 5x1 and dilation of 5
    self.branch_2 = nn.Sequential(
                    Bconv(in_channels, out_channels, 1),
                    Bconv(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
                    Bconv(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
                    Bconv(out_channels, out_channels, 3, padding = 5, dilation = 5)
    )

    # branch comprising bconv of 1x1 to reduce size
    # it is followed by bconv of 1x7, 7x1 and dilation of 7
    self.branch_3 = nn.Sequential(
                    Bconv(in_channels, out_channels, 1),
                    Bconv(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
                    Bconv(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
                    Bconv(out_channels, out_channels, 3, padding = 7, dilation = 7)
    )

    self.conv_concat = Bconv(4*out_channels, out_channels, 3, padding = 1)
    self.conv_res = Bconv(in_channels, out_channels, 1)

  def forward(self, x):

    x_0 = self.branch_0(x)
    x_1 = self.branch_1(x)
    x_2 = self.branch_2(x)
    x_3 = self.branch_3(x)

    # 4 branches that are concatenated together
    x_concat = self.conv_concat(torch.cat(x_0, x_1, x_2, x_3), dim = 1)

    # output of receptive field module after element 
    x = self.relu(x_concat + self.conv_res(x))

    return x

# Partial Decoder Component_Search Module
class PDC_SM(nn.Module):
  def __init__(self, channel):
    super(PDC_SM, self).__init__()
    self.relu = nn.ReLU(True)

    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    # upsampled layers defined
    self.conv_upsample1 = Bconv(channel, channel, 3, padding=1)
    self.conv_upsample2 = Bconv(channel, channel, 3, padding=1)
    self.conv_upsample3 = Bconv(channel, channel, 3, padding=1)
    self.conv_upsample4 = Bconv(channel, channel, 3, padding=1)
    self.conv_upsample5 = Bconv(2*channel, 2*channel, 3, padding=1)

    # cocatenation layers defined 
    self.conv_concat2 = Bconv(2*channel, 2*channel, 3, padding=1)
    self.conv_concat3 = Bconv(4*channel, 4*channel, 3, padding=1)

    # convolution layers defined
    self.conv4 = Bconv(4*channel, 4*channel, 3, padding=1)
    self.conv5 = nn.Conv2d(4*channel, 1, 1)

  def forward(self, x1, x2, x3, x4):
        
    # pdc serach module implemented
    # 4 branches undergoing different operations according to the defined pdc architecture
    x1_1 = x1
    x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
    x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

    x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
    x2_2 = self.conv_concat2(x2_2)

    x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2)), x4), 1)
    x3_2 = self.conv_concat3(x3_2)

    x = self.conv4(x3_2)
    x = self.conv5(x)

    return x

# Partial Decoder Component_Identification Module
class PDC_IM(nn.Module):
    
    def __init__(self, channel):
        super(PDC_IM, self).__init__()
        self.relu = nn.ReLU(True)

        # upsampled layers defined
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = Bconv(channel, channel, 3, padding=1)
        self.conv_upsample2 = Bconv(channel, channel, 3, padding=1)
        self.conv_upsample3 = Bconv(channel, channel, 3, padding=1)
        self.conv_upsample4 = Bconv(channel, channel, 3, padding=1)
        self.conv_upsample5 = Bconv(2*channel, 2*channel, 3, padding=1)

        # cocatenation layers defined 
        self.conv_concat2 = Bconv(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = Bconv(3*channel, 3*channel, 3, padding=1)

        # convolution layers defined
        self.conv4 = Bconv(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):

        # pdc identification module implemented
        # 3 branches undergoing different operations according to the defined pdc architecture
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)

        x3_2 = self.conv_concat3(x3_2)
        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

#ResNet50 model based encoder decoder
class SINet_ResNet50(nn.Module):
  def __init__(self, channel=32, opt=None):
        super(SINet_ResNet50, self).__init__()

        self.resnet = ResNet_2branch()
        self.downSample = nn.MaxPool2d(2, stride=2)    # downsampled in order to feed into rf

        # search module layers defined
        self.rf_low_sm = R_F(320, channel)
        self.rf2_sm = R_F(3584, channel)
        self.rf3_sm = R_F(3072, channel)
        self.rf4_sm = R_F(2048, channel)
        self.pdc_sm = PDC_SM(channel)

        # identification layers defined
        self.rf2_im = R_F(512, channel)
        self.rf3_im = R_F(1024, channel)
        self.rf4_im = R_F(2048, channel)
        self.pdc_im = PDC_IM(channel)

        # upsampling layers defined
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        self.SA = SA()

        if self.training:
            self.initialize_weights()

  def forward(self, x):

    x0 = self.resnet.conv1(x)
    x0 = self.resnet.bn1(x0)
    x0 = self.resnet.relu(x0)

    # - low-level features
    x0 = self.resnet.maxpool(x0)    # (88 x 88 x 64)
    x1 = self.resnet.layer1(x0)     # (88 x 88 x 256)
    x2 = self.resnet.layer2(x1)     # (44 x 44 x 512)

    # ---- Stage-1: Search Module (SM) ----
    x01 = torch.cat((x0, x1), dim=1)        # (88 x 88 x (64+256)) as concatenated
    x01_down = self.downSample(x01)         # (44 x 44 x 320) as downsampled
    x01_sm_rf = self.rf_low_sm(x01_down)    # (44 x 44 x 32) channel size reduced as fed into receptive field module 

    x2_sm = x2                              # (44 x 44 x 512)
    x3_sm = self.resnet.layer3_1(x2_sm)     # (22 x 22 x 1024)
    x4_sm = self.resnet.layer4_1(x3_sm)     # (11 x 11 x 2048)

    x2_sm_cat = torch.cat((x2_sm, self.upsample_2(x3_sm), self.upsample_2(self.upsample_2(x4_sm))), dim=1)   # 2048 + 1024 +512 = 3584 channels
    x3_sm_cat = torch.cat((x3_sm, self.upsample_2(x4_sm)), dim=1)                                            # 2048 + 1024 = 3072 channels

    x2_sm_rf = self.rf2_sm(x2_sm_cat)
    x3_sm_rf = self.rf3_sm(x3_sm_cat)
    x4_sm_rf = self.rf4_sm(x4_sm)
    camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf, x01_sm_rf) # camouflaged map obtained from search module

    # ---- Switcher: Search Attention (SA) ----
    x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2)    # (44 x 44 x 512)

    # ---- Stage-2: Identification Module (IM) ----
    x3_im = self.resnet.layer3_2(x2_sa)                 # (22 x 22 x 1024)
    x4_im = self.resnet.layer4_2(x3_im)                 # (11 x 11 x 2048)

  
    x2_im_rf = self.rf2_im(x2_sa)
    x3_im_rf = self.rf3_im(x3_im)
    x4_im_rf = self.rf4_im(x4_im)

    # - decoder part
    camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)  #camouflaged map obtained from identification module

    # ---- output ----
    return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)  #upsampled output

  def initialize_weights(self):
        resnet50 = models.resnet50(pretrained=True)
        pretrained_dict = resnet50.state_dict()
        all_params = {}

        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())

        self.resnet.load_state_dict(all_params)
        print('[INFO] initialize weights from resnet50')
