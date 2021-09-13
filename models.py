from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class incLstm(nn.Module):
    def __init__(self, n_cl=2, hids=64, mode=0, bidi=1):
        super(incLstm, self).__init__()
        if hids == n_cl:
            bidi = 0
        if not mode:

            self.features = InceptionResnetV1(pretrained='vggface2').eval()
        else:
            # features = InceptionResnetV1(classify=True, num_classes=2)

            self.features = InceptionResnetV1()
        self.mode=mode
        self.RNN = nn.LSTM(input_size=512, hidden_size=hids,
                           bidirectional=True if bidi else False)
        if hids!=n_cl:
            self.drop = nn.Dropout(p=0.5)
            # self.act = nn.ReLU(inplace=True)
            self.classifier = nn.Linear(in_features=hids * 2 if bidi else hids, out_features=n_cl)
            torch.nn.init.xavier_uniform_(self.classifier.weight)
            self.classifier.bias.data.zero_()
        self.hids = hids
        if not mode:
            for parameter in self.features.parameters():
                parameter.requires_grad = False
        self.n_cl=n_cl

    def forward(self, images):
        """Extract feature vectors from input images."""
        BS = images.shape
        images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        if self.mode:
            x = self.features(images)
            x = F.normalize(x,p=2,dim=1)
        else:
            with torch.no_grad():
                x = self.features(images)
        x = x.reshape(x.size(0), -1)
        unpacked = x.view(BS[0], BS[1], -1).permute(1, 0, 2)


        hiddens, (ht, ct) = self.RNN(unpacked)

        last_seq_items = hiddens[-1]
        if self.hids!=self.n_cl:
            x = last_seq_items
            x = self.drop(x)
            # x = self.act(x)
            x = self.classifier(x)
        else:
            x = last_seq_items

        return x

class cnnFull(nn.Module):
    def __init__(self, n_cl=8, hids=64, mode=0, bidi=1):
        super(cnnFull, self).__init__()
        if hids == n_cl:
            bidi = 0
        if not mode:
            self.features = InceptionResnetV1(pretrained='vggface2').eval()
        else:
            self.features = InceptionResnetV1(pretrained='vggface2')
        self.mode=mode
        self.full1 = nn.Linear(in_features=25,out_features=1)
        self.act=nn.ReLU(inplace=True)
        self.drop=nn.Dropout(p=0.5)
        self.classifier = nn.Linear(in_features=512, out_features=n_cl)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.zero_()

        if not mode:
            for parameter in self.features.parameters():
                parameter.requires_grad = False


    def forward(self, images):
        """Extract feature vectors from input images."""
        BS = images.shape
        images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        if self.mode:
            x = self.features(images)
        else:
            with torch.no_grad():
                x = self.features(images)
        x = x.reshape(x.size(0), -1)
        x = x.view(BS[0],BS[1],-1)
        x = x.permute(0, 2, 1).contiguous().view(-1,BS[1])
        x = self.full1(x)
        x = x.reshape(BS[0], -1)
        x = self.act(x)
        x = self.drop(x)
        x = self.classifier(x)


        return x

class cnnCov1(nn.Module):
    def __init__(self, n_cl=8, hids=64, mode=0, bidi=1):
        super(cnnCov1, self).__init__()
        if hids == n_cl:
            bidi = 0
        if not mode:
            self.features = InceptionResnetV1(pretrained='vggface2').eval()
        else:
            self.features = InceptionResnetV1(pretrained='vggface2')
        self.mode=mode
        self.cov1 = nn.Conv1d(in_channels=512,out_channels=512,kernel_size=25,groups=512)
        self.act1=nn.ReLU(inplace=True)

        self.drop1=nn.Dropout(p=0.5)
        self.classifier = nn.Linear(in_features=512, out_features=n_cl)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.zero_()

        if not mode:
            for parameter in self.features.parameters():
                parameter.requires_grad = False


    def forward(self, images):
        """Extract feature vectors from input images."""
        BS = images.shape
        images = images.contiguous().view(-1, BS[2], BS[3], BS[4])
        if self.mode:
            x = self.features(images)
        else:
            with torch.no_grad():
                x = self.features(images)
        x = x.reshape(x.size(0), -1)
        x = x.view(BS[0],BS[1],-1)
        x = x.permute(0, 2, 1)
        x = self.cov1(x)
        x = x.reshape(BS[0], -1)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.classifier(x)


        return x