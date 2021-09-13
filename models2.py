from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from inception import InceptionBlock

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, kernel_size=8, padding=3):
        super(BasicBlock1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.conv1 = nn.Conv1d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size, padding=padding)
        self.bn1 = norm_layer(planes, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out

class Se1Block(nn.Module):

    def __init__(self, channel, reduction=16):
        super(Se1Block, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super(ResBlock1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.conv1 = BasicBlock1D(inplanes=inplanes, planes=planes, kernel_size=7, padding=3, norm_layer=norm_layer)

        self.conv2 = BasicBlock1D(inplanes=planes, planes=planes, kernel_size=5, padding=2, norm_layer=norm_layer)

        self.conv3 = nn.Conv1d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.bn3 = norm_layer(planes, momentum=0.01)
        self.downsample = nn.Conv1d(in_channels=inplanes, out_channels=planes, kernel_size=1, padding=0)
        self.bnd = norm_layer(planes, momentum=0.01)
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.bnd(identity)

        out += identity
        out = self.relu(out)

        return out

class Classifier_RESNET(nn.Module):

    def __init__(self, input_shape, D=1):
        super(Classifier_RESNET, self).__init__()
        self.out_shape = int(128 / D)
        self.blk1 = ResBlock1D(input_shape, int(64 / D))
        self.blk2 = ResBlock1D(int(64 / D), int(128 / D))
        self.blk3 = ResBlock1D(int(128 / D), int(128 / D))
        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.AVG(x)
        return x

class FCN(nn.Module):

    def __init__(self, input_shape, reduction=0,D=1):
        super(FCN, self).__init__()

        self.input_shape = input_shape
        self.reduction=reduction
        self.out_shape = int(128 / D)
        # self.conv0 = BasicBlock1D(inplanes=input_shape, planes=32, kernel_size=1, padding=0)
        self.conv1 = BasicBlock1D(inplanes=input_shape, planes=int(128 / D), kernel_size=8, padding=3)
        if not reduction==0:
            self.SE1=Se1Block(channel=int(128 / D),reduction=reduction)


        self.conv2 = BasicBlock1D(inplanes=int(128 / D), planes=int(256 / D), kernel_size=5, padding=2)
        if not reduction==0:
            self.SE2=Se1Block(channel=int(256 / D),reduction=reduction)

        self.conv3 = BasicBlock1D(inplanes=int(256 / D), planes=int(128 / D), kernel_size=3, padding=1)
        if not reduction==0:
            self.SE3=Se1Block(channel=int(128 / D),reduction=reduction)

        self.AVG = nn.AdaptiveAvgPool1d(1)
        self.dp = nn.Dropout(p=0.7)
    def forward(self, x):
        # x = self.conv0(x)
        x = self.conv1(x)
        if not self.reduction==0:
            x=self.SE1(x)
        x = self.conv2(x)
        if not self.reduction==0:
            x=self.SE2(x)
        x = self.conv3(x)
        if not self.reduction==0:
            x=self.SE3(x)
        x = self.AVG(x)
        x = x.reshape(x.size(0), -1)
        x = self.dp(x)
        return x

class ResBlock1D(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super(ResBlock1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.conv1 = BasicBlock1D(inplanes=inplanes, planes=planes, kernel_size=7, padding=3, norm_layer=norm_layer)

        self.conv2 = BasicBlock1D(inplanes=planes, planes=planes, kernel_size=5, padding=2, norm_layer=norm_layer)

        self.conv3 = nn.Conv1d(in_channels=planes, out_channels=planes, kernel_size=3, padding=1)
        self.bn3 = norm_layer(planes, momentum=0.01)
        self.downsample = nn.Conv1d(in_channels=inplanes, out_channels=planes, kernel_size=1, padding=0)
        self.bnd = norm_layer(planes, momentum=0.01)
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.bnd(identity)

        out += identity
        out = self.relu(out)

        return out

class RESNET(nn.Module):

    def __init__(self, input_shape, reduction=0,D=1):
        super(RESNET, self).__init__()
        self.out_shape = int(128 / D)
        self.blk1 = ResBlock1D(input_shape, int(64 / D))
        self.blk2 = ResBlock1D(int(64 / D), int(128 / D))
        self.blk3 = ResBlock1D(int(128 / D), int(128 / D))
        self.AVG = nn.AdaptiveAvgPool1d(1)
        self.dp = nn.Dropout(p=0.7)
    def forward(self, x):
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.AVG(x)
        x = x.reshape(x.size(0), -1)
        x = self.dp(x)
        return x

class MYLSTM(nn.Module):

    def __init__(self, input_size=512, hidden_size=128):
        super(MYLSTM, self).__init__()


        self.RNN=nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.dp=nn.Dropout(p=0.7)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        hiddens, (ht, ct) = self.RNN(x)

        x = hiddens[-1]
        x = x.reshape(x.size(0), -1)
        x=self.dp(x)

        return x

class AttentionAVG(nn.Module):

    def __init__(self, input_size=512, reduction=4):
        super(AttentionAVG, self).__init__()


        self.ATT=nn.Sequential(
                nn.Linear(input_size, input_size//reduction),
                nn.ReLU(),
                nn.Linear(input_size//reduction, 1),
                nn.Softmax(dim=1))
        self.dp=nn.Dropout(p=0.7)
        self.AVG = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x=x.permute(0,2,1)
        tmp=self.ATT(x)
        x=x*tmp.expand_as(x)
        x=self.AVG(x.permute(0,2,1))
        x = x.reshape(x.size(0), -1)
        x=self.dp(x)

        return x


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CNNTemporal(nn.Module):
    def __init__(self,device='cpu', reduction=4,reduction3=0, hids=64, Frz=0,mode=0,n_cl=3):
        super(CNNTemporal, self).__init__()
        if Frz:

            self.features = InceptionResnetV1(pretrained='vggface2').eval()
        else:
            # features = InceptionResnetV1(classify=True, num_classes=2)

            self.features = InceptionResnetV1(pretrained='vggface2')
        self.Frz=Frz
        self.mode=mode
        self.TP= nn.ModuleList([])
        self.TPO=0
        if mode==0:
            self.TP.append(nn.Sequential(nn.AdaptiveAvgPool1d(1),Flatten(),nn.Dropout(p=0.7)).to(device))
            self.TPO+=512
        elif mode==1:
            self.TP.append(AttentionAVG(input_size=512,reduction=reduction).to(device))
            self.TPO += 512
        elif mode==2:
            self.TP.append(MYLSTM(input_size=512, hidden_size=hids).to(device))
            self.TPO += hids
        elif mode == 3:
            self.TP.append(FCN(input_shape=512 ,reduction=reduction3).to(device))
            self.TPO += 128
        elif mode == 4:
            self.TP.append(RESNET(input_shape=512 ,reduction=reduction3).to(device))
            self.TPO += 128
        elif mode == 5:
            self.TP.append(InceptionBlock(in_channels=512 ,n_filters=hids).to(device))
            self.TPO += 4*hids
        elif mode==21:
            self.TP.append(MYLSTM(input_size=512, hidden_size=hids).to(device))
            self.TPO += hids
            self.TP.append(AttentionAVG(input_size=512,reduction=reduction).to(device))
            self.TPO += 512

        elif mode == 23:
            self.TP.append(MYLSTM(input_size=512, hidden_size=hids).to(device))
            self.TPO += hids
            self.TP.append(FCN(input_shape=512 ,reduction=reduction3).to(device))
            self.TPO += 128
        elif mode == 24:
            self.TP.append(MYLSTM(input_size=512, hidden_size=hids).to(device))
            self.TPO += hids
            self.TP.append(RESNET(input_shape=512, reduction=reduction3).to(device))
            self.TPO += 128
        elif mode == 25:
            self.TP.append(MYLSTM(input_size=512, hidden_size=hids).to(device))
            self.TPO += hids
            self.TP.append(InceptionBlock(in_channels=512, n_filters=hids).to(device))
            self.TPO += 4 * hids
        elif mode == 31:
            self.TP.append(AttentionAVG(input_size=512,reduction=reduction).to(device))
            self.TPO += 512
            self.TP.append(FCN(input_shape=512,reduction=reduction3).to(device))
            self.TPO += 128
        elif mode == 41:
            self.TP.append(AttentionAVG(input_size=512,reduction=reduction).to(device))
            self.TPO += 512
            self.TP.append(InceptionBlock(in_channels=512, n_filters=hids).to(device))
            self.TPO += 4 * hids
        elif mode == 51:
            self.TP.append(AttentionAVG(input_size=512,reduction=reduction).to(device))
            self.TPO += 512
            self.TP.append(RESNET(input_shape=512, reduction=reduction3).to(device))
            self.TPO += 128
        elif mode == 312:
            self.TP.append(MYLSTM(input_size=512, hidden_size=hids).to(device))
            self.TPO += hids
            self.TP.append(FCN(input_shape=512,reduction=reduction3).to(device))
            self.TPO += 128
            self.TP.append(AttentionAVG(input_size=512,reduction=reduction).to(device))
            self.TPO += 512
        elif mode == 412:
            self.TP.append(MYLSTM(input_size=512, hidden_size=hids).to(device))
            self.TPO += hids
            self.TP.append(RESNET(input_shape=512, reduction=reduction3).to(device))
            self.TPO += 128
            self.TP.append(AttentionAVG(input_size=512,reduction=reduction).to(device))
            self.TPO += 512
        elif mode == 512:
            self.TP.append(MYLSTM(input_size=512, hidden_size=hids).to(device))
            self.TPO += hids
            self.TP.append(InceptionBlock(in_channels=512, n_filters=hids).to(device))
            self.TPO += 4 * hids
            self.TP.append(AttentionAVG(input_size=512,reduction=reduction).to(device))
            self.TPO += 512


        self.classifier = nn.Linear(in_features=self.TPO, out_features=n_cl)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.zero_()
        self.hids = hids
        if Frz:
            for parameter in self.features.parameters():
                parameter.requires_grad = False
        self.n_cl=n_cl

    def forward(self, images):
        """Extract feature vectors from input images."""
        BS = images.shape
        images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        out=[]
        if not self.Frz:
            x = self.features(images)
        else:
            with torch.no_grad():
                x = self.features(images)
        x = x.reshape(x.size(0), -1)
        x = x.view(BS[0], BS[1], -1).permute(0,2,1)
        for path in self.TP:
            out.append(path(x))
        out=torch.cat(out,dim=1)
        out=self.classifier(out)


        return out.squeeze(1),x