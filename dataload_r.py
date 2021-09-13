import glob

import torch
from torch.utils.data import Dataset
import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps, ImageChops, ImageFilter
import numbers
import random
from itertools import islice
from random import shuffle
import math
import pickle
import csv
import torchvision.transforms as transforms
from pathlib import Path
import math
# import pandas as pd
import cv2
import h5py
import time
from pathlib import Path
from facenet_pytorch import MTCNN,InceptionResnetV1

def datPP3_r(root_dir='D:\\Downloads\\Compressed\\angry_cropped\\',Ra=0):
    videos = glob.glob(root_dir + '*/*/*')
    listA = []
    listTr = []
    listVa = []
    dict={'ANGR':0,'DISG':2,'CONF':1}
    x = [x for x in range(5) if x!=Ra]
    #print(videos)
    #print(len(videos))
    for vid in videos:
        name=vid.split(os.path.sep)[-1]
        duration=len(glob.glob(vid+'/*.jpeg'))
        emotion=name.split('_')[-1]
        emotion_no=dict[emotion]
        # cut=int(name.split('_')[-2])
        # MF=int(name.split('_')[-4])
        fold=int(name.split('_')[-6])
        tmpinfo=[vid,emotion_no,fold,duration]
        listA.append(tmpinfo)
        if fold==Ra+1:
            listVa.append(tmpinfo)
        else:
            listTr.append(tmpinfo)
    return listA, listTr,listVa


class MakeH_r(Dataset):
    def __init__(self,
                 clips,
                 transform=None,
                 Len=64,
                 val=0
                 ):
        self.clips = clips
        self.transform = transform
        self.lenD=len(clips)
        self.Len=Len
        self.val=val


    def __getitem__(self, index):
        path,target,fold,duration= self.clips[index]

        images = []
        if duration<=self.Len:
            start=0
        else:
            if self.val==0:
                start=random.randint(0, duration%self.Len)
            else:
                #start=(duration%self.Len)//2
                start=(duration-self.Len)//2
        img_pat=path+'/%d.jpeg'
        seed = np.random.randint(2147483646)
        for i in range(1,self.Len+1):
            mod_i=(i+start)%duration
            mod_i=duration if mod_i==0 else mod_i
            img_path=img_pat % mod_i
            image = Image.open(img_path)
            # ime=ImO.equalize(image)
            # im3 = ImageChops.multiply(image, mask)
            # im3.show()
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if self.transform:
                image = self.transform(image)
            # else:
            #     image = transforms.ToTensor()(image)
            #            image=image.repeat(3,1,1)
            images.append(image)


        images=torch.stack(images)
        return images,target,fold

    def __len__(self):
        return len(self.clips)

class MakeH_ra(Dataset):
    def __init__(self,
                 clips,
                 transform=None,
                 Len=64,
                 val=0
                 ):
        self.clips = clips
        self.transform = transform
        self.lenD=len(clips)
        self.Len=Len
        self.val=val
        self.ALL=[]
        for clip in self.clips:
            path, target, fold, duration = clip

            images = []
            img_pat = path + '/%d.jpeg'

            for i in range(1, duration + 1):

                img_path = img_pat % i
                image = transforms.ToTensor()(Image.open(img_path)).unsqueeze_(0)

                images.append(image)
            self.ALL.append([images,target,fold,duration])

    def __getitem__(self, index):
        images_a,target,fold,duration= self.ALL[index]

        images = []
        if duration<=self.Len:
            start=0
        else:
            if self.val==0:
                start=random.randint(0, duration%self.Len)
            else:
                #start=(duration%self.Len)//2
                start=(duration-self.Len)//2

        seed = np.random.randint(2147483646)
        for i in range(self.Len):
            mod_i=(i+start)%duration
            image=transforms.ToPILImage()(images_a[mod_i].squeeze_(0))

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if self.transform:
                image = self.transform(image)
            #            image=image.repeat(3,1,1)
            images.append(image)


        images=torch.stack(images)
        return images,target,fold
    def __len__(self):
        return len(self.clips)

if __name__ == '__main__':
    a, b, c = datPP3_r()
    path='D:\\Downloads\\Compressed\\angry_f\\'
    # transform = transforms.Compose([
    #     # transforms.CenterCrop(270),
    #     # transforms.Resize((160, 160)),
    #     transforms.RandomChoice([
    #         transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    #         transforms.RandomAffine(5, translate=(0.05, 0.05), scale=None, shear=5, resample=Image.NEAREST,
    #                                 fillcolor=0),
    #         transforms.RandomResizedCrop(160, scale=(0.9, 1.0), ratio=(0.95, 1.15))]),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5], [0.5]),
    # ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    TD = MakeH_ra(a, transform=transform,val=1)
    DataLoader=torch.utils.data.DataLoader(
            TD,
            batch_size=8,
            num_workers=0, pin_memory=True,shuffle=False)
    net=InceptionResnetV1(pretrained='vggface2').eval()
    for parameter in net.parameters():
        parameter.requires_grad = False
    net=net.to(device)
    pattern=path+'features_%d_%d_emo_%d_fold_%d.pth'
    for i,data in enumerate(DataLoader):
        images=data[0]
        target=data[1]
        fold=data[2]
        images=images.to(device)
        BS=images.shape
        with torch.set_grad_enabled(False):
            features=net(images.contiguous().view(-1,BS[2],BS[3],BS[4])).view(BS[0],BS[1],-1)

        for j,f in enumerate(features.cpu()):
            torch.save(f,pattern%(i,j,target[j],fold[j]))



