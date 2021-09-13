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
import time
from facenet_pytorch import MTCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pure_pil_alpha_to_color_v2(image, color=(255, 255, 255)):
    """Alpha composite an RGBA Image with a specified color.

    Simpler, faster version than the solutions above.

    Source: http://stackoverflow.com/a/9459208/284318

    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)

    """
    image.load()  # needed for split()
    background = Image.new('RGB', image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background

def datPP3(root_dir='C:\\Users\\abbas\\OneDrive\\Documents\\makehuman\\v1py3\\grab_cropped\\', iv=1,it=2):
    class_paths = glob.glob(root_dir + '*\\*\\')
    listp=[[] for x in range(17)]
    listPT = []
    listPV = []
    listPTe = []
    foldkey = {'mass0005':0,
                'mass0006':1,
                'mass0011':2,
                'mass0016':3,
                'mass0018':4,
                'mass0019':5,
                'mass0023':6,
                'mod0002':7,
                'mod0003':8,
                'mod0010':9,
                'mod0014':10,
                'mod0017':11,
                'mod0025':12,
                'mod0027':13,
                'sefid0002':14,
                'sefid0004':15,
                'sefid0005':16}
    foldkey2 = {
        'anger01': 0,
        'anger02': 1,
        'angerdefault01': 2,
        'angerdefault02': 3,
        'angerdefault': 4,
        'disgust01': 5,
        'disgust02': 6,
        'disgustdefault02': 7,
        'disgustdefault02': 8,
        'disgustdefault02': 9,
        'disgustdefault02': 10,
        'EyesRLnChinUppernMouthLowernBrowlower': 11,
        'eyesRLnWHAAT': 12,
        'eyesRL': 13,
        'headback': 14,
        'LidtightnBrowLowernChinUpper': 15,
        'LidtightnBrowLowernOpenMouth': 16,
        'LidTightnLipChange': 17,
    }
    for class_path in class_paths:
        paths = len(sorted(glob.glob(os.path.join(class_path, '*.jpeg'))))
        folder=class_path.split(os.path.sep)[-2]
        ya = int(folder.split('_')[7])
        za = int(folder.split('_')[9])
        model = folder.split('_')[10]
        emotion = folder.split('_')[11]
        target=int(class_path.split(os.path.sep)[-3])
        if emotion!='HeadLeftnBrowLower':
            listp[foldkey[model]].append([class_path, target, ya, za, foldkey[model], foldkey2[emotion], paths])
        # if emotion!='HeadLeftnBrowLower' and emotion!='angerdefault01'and emotion!='angerdefault02'  :
        #     if target==0:
        #         if folder.split('_')[12]!='combination22'and  folder.split('_')[12]!='combination122':
        #             listp[foldkey[model]].append([class_path, target, ya, za, foldkey[model], foldkey2[emotion], paths])
        #     else:
        #         listp[foldkey[model]].append([class_path,target,ya,za,foldkey[model],foldkey2[emotion],paths])
    ratio = [0.64, 0.18, 0.18]
    ld = len(listp)
    lengths = [int(ld * ratio[0]), int(ld * ratio[1]), int(ld * ratio[2])]
    if (sum(lengths) != ld):
        # lengths[0] += ld - sum(lengths)
        for ic in range(ld - sum(lengths)):
            lengths[ic] += 1
    random.seed(13722)
    shuffle(listp)
    Inputt = iter(listp)
    S_d1 = [list(islice(Inputt, elem))
            for elem in lengths]
    for x in S_d1[0]:
        listPT.extend(x)
    for x in S_d1[iv]:
        listPV.extend(x)
    for x in S_d1[it]:
        listPTe.extend(x)
    return listPT,listPV,listPTe


class MakeH(Dataset):
    def __init__(self,
                 clips,
                 transform=None,
                 length=25):
        self.clips = clips
        self.transform = transform
        self.length=length
        self.lenD=len(clips)
        self.mtcn=MTCNN()

    def __getitem__(self, index):
        path, target, _,_,_,_,duration= self.clips[index]

        images = []
        seed = np.random.randint(2147483646)
        for i in range(1,self.length+1):
            #            print(i)
            modedindex=i%duration
            modedindex= duration if modedindex == 0 else modedindex

            image_path = os.path.join(path+'{:d}.png'.format(modedindex))
            # image = pure_pil_alpha_to_color_v2(Image.open(image_path))
            image=cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            # try:
            #     image = self.mtcn(image)
            # except Exception as e:
            #     # print("<p>Error: %s</p>" % str(e))
            #     print(path)
            #     image = Image.fromarray(image)
            #     image = self.transform(image)



            # cropped=self.mtcn(image)
            # image=Image.fromarray(image)
            #
            # torch.manual_seed(seed)
            # torch.cuda.manual_seed(seed)
            # torch.cuda.manual_seed_all(seed)
            # np.random.seed(seed)
            # random.seed(seed)
            #
            # torch.backends.cudnn.enabled = False
            # torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.deterministic = True

            # if self.transform:
            #     image = self.transform(image)

            images.append(torch.tensor(image))
            # images.append(image)


        x = torch.stack(images)
        images = []
        try:
            x = torch.stack(self.mtcn(x))
        except Exception as e:
            # print("<p>Error: %s</p>" % str(e))
            print(path)
            for image in x:
                image = Image.fromarray(image.numpy())
                image = self.transform(image)
                images.append(image)
            x=torch.stack(images)

        return x, target

    def __len__(self):
        return self.lenD

class MakeH2_(Dataset):
    def __init__(self,
                 clips,
                 transform=None,
                 length=25):
        self.clips = clips
        self.transform = transform
        self.length=length
        self.lenD=len(clips)
        # self.mtcn=MTCNN()

    def __getitem__(self, index):
        path, target,ya,za,model,emotion,duration= self.clips[index]

        images = []
        seed = np.random.randint(2147483646)
        for i in range(1,self.length+1):
            #            print(i)
            modedindex=i%duration
            modedindex= duration if modedindex == 0 else modedindex

            image_path = os.path.join(path+'{:d}.jpeg'.format(modedindex))
            # image = pure_pil_alpha_to_color_v2(Image.open(image_path))
            image=cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            # try:
            #     image = self.mtcn(image)
            # except Exception as e:
            #     # print("<p>Error: %s</p>" % str(e))
            #     print(path)
            #     image = Image.fromarray(image)
            #     image = self.transform(image)



            # cropped=self.mtcn(image)
            image=Image.fromarray(image)
            #
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

            # images.append(torch.tensor(image))
            images.append(image)


        x = torch.stack(images)

        return x, target,ya,za,model,emotion

    def __len__(self):
        return self.lenD
#####################
###################################
class MakeH2(Dataset):
    def __init__(self,
                 clips,
                 transform=None,
                 length=25):
        self.clips = clips
        self.transform = transform
        self.length=length
        self.lenD=len(clips)
        self.imagelist=[]
        # self.mtcn=MTCNN()
        imgs=[]
        for folder,_,_,_,_,_,duration in clips:
            for i in range(1,self.length+1):
                modedindex = i % duration
                modedindex = duration if modedindex == 0 else modedindex
                image_path = os.path.join(folder + '{:d}.jpeg'.format(modedindex))
                image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                imgs.append(image)
            self.imagelist.append(imgs)
            imgs=[]



    def __getitem__(self, index):
        path, target,ya,za,model,emotion,duration= self.clips[index]

        images = []
        seed = np.random.randint(2147483646)
        imgs=self.imagelist[index]
        for i in range(0,self.length):
            #            print(i)
            image=imgs[i]
            #
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

            # images.append(torch.tensor(image))
            images.append(image)


        x = torch.stack(images)

        return x, target,ya,za,model,emotion

    def __len__(self):
        return self.lenD
    ############################

# if __name__ == '__main__':
#     a, _, _ = datPP3()
#     transform = transforms.Compose([
#
#         # transforms.RandomRotation(args.An * 45, resample=Image.BICUBIC),
#         # transforms.RandomAffine( 45, translate=(0.05, 0.05), scale=None, shear=2, resample=Image.NEAREST,
#         #                         fillcolor= 0),
#         # transforms.ColorJitter(brightness=0.05, contrast=0.05),
#         # transforms.RandomPerspective(distortion_scale=0.1, p=0.5, interpolation=3),
#         # Crp(c=args.crp),
#         # transforms.Resize((224, 224)),
#         # transforms.RandomResizedCrop(224, scale=(0.85, 1.05), ratio=(1.0, 1.0)),
#         # transforms.RandomHorizontalFlip(p=0.5),
#         # transforms.RandomVerticalFlip(p=0.3),
#         transforms.ToTensor(),
#         # transforms.Normalize([0.4], [0.2]),
#     ])
#     TD = MakeH(a, transform=transform)
#     train_loader = torch.utils.data.DataLoader(
#         TD,
#         batch_size=4,
#         num_workers=1, pin_memory=True, shuffle=True)
#     start = time.time()
#     mtcn=MTCNN()
#     # mtcn2 = MTCNN(device=device)
#     input,target=next(iter(train_loader))
#     start = time.time()
#     cropped=mtcn(input.contiguous().view(-1,500,500,3))
#     print(time.time() - start)
#
#
#     start=time.time()
#     mtcn2 = MTCNN(device=device)
#     input = mtcn2(input.contiguous().view(-1, 500, 500, 3))
#     del mtcn2
#     torch.cuda.empty_cache()
#     print(time.time() - start)
#     print('fonr')


