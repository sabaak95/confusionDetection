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

def datPPI3(root_dir='C:\\Users\\abbas\\OneDrive\\Documents\\makehuman\\v1py3\\anger_complete9\\', iv=1,it=2):
    class_paths = glob.glob(root_dir + '/*/*/')
    listp=[[] for x in range(23)]
    listPT = []
    listPV = []
    listPTe = []
    foldkey = {'man1_Afr':0,
                'Man1_Asi':1,
                'Man1_Mix':2,
                'man1_whi':3,
                'man2_Afr':4,
                'Man2_Asi':5,
                'man2_mix':6,
                'man2_whi':7,
                'man3_Afr':8,
                'Man3_Asi':9,
                'man3_mix':10,
                'man3_whi':11,
                'woman2_mix':12,
                'Women1_Afr':13,
                'Women1_Asi':14,
                'women1_Mix':15,
                'Women1_Whi':16,
                'Women2_afr':17,
                'Women2_Asi':18,
                'Women2_Whi':19,
                'Women3_Asi':20,
                'women3_mix':21,
                'women3_whi':22,
                #'women3_Afr': 23
        }
    dict = {'ANGR': 0, 'DISG': 2, 'CONF': 1}
    for class_path in class_paths:
        paths = len(sorted(glob.glob(os.path.join(class_path, '*.jpeg'))))
        folder=class_path.split(os.path.sep)[-2]
        # ya=int(folder.split('_')[7])
        # za = int(folder.split('_')[9])
        model = folder.split('_')[-4]+'_'+folder.split('_')[-3]
        # emotion = folder.split('_')[11]
        target=dict[folder.split('_')[-1]]

        if model!='Women3_afr':
            listp[foldkey[model]].append([class_path,target,paths,folder])
    ratio = [0.78, 0.22,0]
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
    return listPT+listPV,listPT,listPV

def datPPI32(root_dir='C:\\Users\\abbas\\OneDrive\\Documents\\makehuman\\v1py3\\anger_complete9\\'):
    class_paths = glob.glob(root_dir +'/*/*/')
    listp=[[[] for x in range(21)] for x in range(23)]
    listPT = []
    listPV = []
    listPTe = []
    foldkey = {'man1_Afr':0,
                'Man1_Asi':1,
                'Man1_Mix':2,
                'man1_whi':3,
                'man2_Afr':4,
                'Man2_Asi':5,
                'man2_mix':6,
                'man2_whi':7,
                'man3_Afr':8,
                'Man3_Asi':9,
                'man3_mix':10,
                'man3_whi':11,
                'woman2_mix':12,
                'Women1_Afr':13,
                'Women1_Asi':14,
                'women1_Mix':15,
                'Women1_Whi':16,
                'Women2_afr':17,
                'Women2_Asi':18,
                'Women2_Whi':19,
                'Women3_Asi':20,
                'women3_mix':21,
                'women3_whi':22,
                #'women3_Afr': 23
        }
    foldkey2={'eyew2':0,'100000eyew2':0,
                '11112':1,
                '1':2,
                '5MEH':3,
                '66':4,
                '7MEH':5,
                '8MEH':6,
                'cheeckraise':7,
                'eyebrowraiseJaw':8,
                'eyesRL':9,
                'lLidtightnBrowLowernChinUpper':10,
                'lLidtightnBrowLowernOpenMouth':11,
                'openMouth':12,
                'still3':13,
                'Z5changes':14,
                'ZCheekRaiseLipstrecher1':15,
                'Zeyeclosed':16,
                'ZLipCornerDepress2':17,
                'ZLipsPartDimpler3':18,
                'ZUpperLipJawopen6':19,
                'zuznew7':20
            }
    dict = {'ANGR': 0, 'DISG': 2, 'CONF': 1}
    for class_path in class_paths:
        paths = len(sorted(glob.glob(os.path.join(class_path, '*.jpeg'))))
        folder=class_path.split(os.path.sep)[-2]
        # ya=int(folder.split('_')[7])
        # za = int(folder.split('_')[9])
        model = folder.split('_')[-4]+'_'+folder.split('_')[-3]
        emotion = folder.split('_')[-2]
        target=dict[folder.split('_')[-1]]

        if model!='Women3_afr':
            listp[foldkey[model]][foldkey2[emotion]].append([class_path,target,paths,folder])
    ratio = [0.78, 0.22,0]
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
    for x in S_d1[1]:
        listPV.extend(x)
    for x in S_d1[2]:
        listPTe.extend(x)
    return listp,listPT,listPV

def splitratio(listp,ratio,val,val2=0):
    if val:
        random.seed(543)
    shuffle(listp)
    Inputt = iter(listp)
    ld = len(listp)
    listPT=[]
    listPV=[]
    lengths = [int(ld * ratio), int(ld * (1-ratio))]
    S_d1 = [list(islice(Inputt, elem))
            for elem in lengths]
    
    for x in S_d1[val2]:
        listPT.extend(x)



    
    return listPT

class MySamplerV(torch.utils.data.sampler.Sampler):
    def __init__(self, lenght):
        indices = []
        for i in range(lenght):

            indices += 9 * [i]

        indices = torch.tensor(indices)
        self.indices = indices

    def __iter__(self):
        indices = self.indices[torch.randperm(len(self.indices))]
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)

class MakeHIratio(Dataset):
    def __init__(self,
                 clips,
                 transform=None,
                 Len=25,
                 folder=0,
                 ratio=0.79,
                 val=0,
                 val2=0):
        self.listclips = clips
        self.ratio=ratio
        self.clips=splitratio(clips,ratio,val,val2)
        self.transform = transform
        self.length=Len
        self.lenD=len(self.clips) if not val2 else len(self.clips)*9
        # self.mtcn=MTCNN().to(device)
        self.FE=folder
        self.val=val
        self.val2=val2

    def __getitem__(self, index):
        if self.val2:
            self.clips[index].append(self.clips[index].pop(0))
            path, target, duration, folder = self.clips[index][0]
        else:
            #print(len(self.clips[index]))
            #print(self.clips[index])
            path, target,duration,folder= self.clips[index][random.randrange(9)]

        images = []
        seed = np.random.randint(2147483646)
        for i in range(1,self.length+1):
            #            print(i)
            modedindex=i%duration
            modedindex= duration if modedindex == 0 else modedindex

            image_path = os.path.join(path+'{:d}.jpeg'.format(modedindex))
            # image = pure_pil_alpha_to_color_v2(Image.open(image_path))
            if self.FE:
                image=cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            else:
                image = Image.open(image_path)
            # cropped=self.mtcn(image)
            # image=Image.fromarray(image)
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
            else:
                image=torch.tensor(image)
            images.append(image)


        x = torch.stack(images)
        if self.FE:
            return x, target, folder
        else:
            return x, target,target
    def update(self):
        self.clips = splitratio(self.listclips, self.ratio,self.val)
    def __len__(self):
        return self.lenD
        
class MakeHIratio2(Dataset):
    def __init__(self,
                 clips,
                 transform=None,
                 Len=25,
                 folder=0,
                 ratio=0.79,
                 val=0,
                 val2=0,
                 loopD=5):
        self.listclips = clips
        self.ratio=ratio
        self.clips=splitratio(clips,ratio,val,val2)
        self.transform = transform
        self.length=Len
        self.lenD=len(self.clips) if not val2 else len(self.clips)*9
        # self.mtcn=MTCNN().to(device)
        self.FE=folder
        self.val=val
        self.val2=val2
        self.loopD=loopD
    def __getitem__(self, index):
        if self.val2:
            self.clips[index].append(self.clips[index].pop(0))
            path, target, duration, folder = self.clips[index][0]
        else:
            path, target,duration,folder= self.clips[index][random.randrange(9)]

        images = []
        # start=random.randrange(duration)
        start=duration-self.loopD
        seed = np.random.randint(2147483646)
        inv=0
        for i in range(1,self.length+1):
            #            print(i)
            modedindex=i%self.loopD
            if modedindex==0:
                inv=1-inv
            if inv == 0:
                modedindex+=start
            else:
                modedindex =duration- modedindex


            image_path = os.path.join(path+'{:d}.jpeg'.format(modedindex))
            # image = pure_pil_alpha_to_color_v2(Image.open(image_path))
            if self.FE:
                image=cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            else:
                image = Image.open(image_path)
            # cropped=self.mtcn(image)
            # image=Image.fromarray(image)
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
            else:
                image=torch.tensor(image)
            images.append(image)


        x = torch.stack(images)
        if self.FE:
            return x, target, folder
        else:
            return x, target,target
    def update(self):
        self.clips = splitratio(self.listclips, self.ratio,self.val)
    def __len__(self):
        return self.lenD

class MakeHI(Dataset):
    def __init__(self,
                 clips,
                 transform=None,
                 Len=25,
                 folder=0):
        self.clips = clips
        self.transform = transform
        self.length=Len
        self.lenD=len(clips)
        # self.mtcn=MTCNN().to(device)
        self.FE=folder

    def __getitem__(self, index):
        path, target,duration,folder= self.clips[index]

        images = []
        seed = np.random.randint(2147483646)
        for i in range(1,self.length+1):
            #            print(i)
            modedindex=i%duration
            modedindex= duration if modedindex == 0 else modedindex

            image_path = os.path.join(path+'{:d}.jpeg'.format(modedindex))
            # image = pure_pil_alpha_to_color_v2(Image.open(image_path))
            if self.FE:
                image=cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            else:
                image = Image.open(image_path)
            # cropped=self.mtcn(image)
            # image=Image.fromarray(image)
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
            else:
                image=torch.tensor(image)
            images.append(image)


        x = torch.stack(images)
        if self.FE:
            return x, target, folder
        else:
            return x, target,target

    def __len__(self):
        return self.lenD

class MakeHIa(Dataset):
    def __init__(self,
                 clips,
                 transform=None,
                 Len=25,
                 folder=0):
        self.clips = clips
        self.transform = transform
        self.length=Len
        self.lenD=len(clips)
        # self.mtcn=MTCNN().to(device)
        self.FE=folder
        self.ALL = []
        for clip in self.clips:
            path, target,duration,folder = clip

            images = []
            img_pat = path + '%d.jpeg'

            for i in range(1, duration + 1):
                img_path = img_pat % i
                image = transforms.ToTensor()(Image.open(img_path)).unsqueeze_(0)

                images.append(image)
            self.ALL.append([images, target, duration,folder])
    def __getitem__(self, index):
        images_a, target,duration,folder= self.ALL[index]

        images = []
        seed = np.random.randint(2147483646)
        for i in range(self.length):
            #            print(i)
            modedindex=i%duration

            image = transforms.ToPILImage()(images_a[modedindex].squeeze_(0))

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
            else:
                image=torch.tensor(image)
            images.append(image)


        x = torch.stack(images)
        if self.FE:
            return x, target, folder
        else:
            return x, target,int(target)

    def __len__(self):
        return self.lenD
#####################

if __name__ == '__main__':
    a, b, c = datPPI3(root_dir='D:\\Mehryar\\MLN\\grab\\*\\')
    mtcn = MTCNN(device=device,thresholds=[0.4,0.5,0.5],selection_method='probability')
    TD=MakeHI(a,folder=1)
    dict = {0:'ANGR', 2:'DISG', 1:'CONF'}
    DataLoader=torch.utils.data.DataLoader(
            TD,
            batch_size=1,
            num_workers=0, pin_memory=True)
    transform = transforms.Compose([transforms.ToPILImage(),
        transforms.CenterCrop(250),
                                    transforms.Resize(160),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    path='D:\\Mehryar\\MLN\\anger+conf\\'
    for data in DataLoader:
        images=data[0].squeeze(0)
        # path=data[-1][0]
        folder=data[-1][0]
        target=data[1][0]
        try:
            imgs_cut=mtcn(images)
        except Exception as e:

            print(folder)
            imgs_cut=[]
            for img_o in images:

                imgs_cut.append(transform(img_o.permute(2,0,1)))
            # imgs_cut=torch.stack(imgs_cut)



        new_path=path.split(os.sep)
        new_path[-2] += '_cropped'
        new_path[0]+='\\'
        newpath=os.path.join(*new_path[:-1])+'\\'+dict[int(target)]+'\\'+ folder+'\\'
        # name=new_path[-1].split('.')[0]
        # imgs_s=[]
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        for i in range(len(imgs_cut)):
            image=imgs_cut[i].permute(1,2,0).cpu().numpy()
            image=cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.imwrite(os.path.join(newpath,'{:d}.jpeg'.format(i+1)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

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


