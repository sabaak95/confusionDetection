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

def datPP3(root_dir='C:\\Users\\abbas\\OneDrive\\Documents\\makehuman\\v1py3\\grab\\', iv=1,it=2):
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
    for class_path in class_paths:
        paths = len(sorted(glob.glob(os.path.join(class_path, '*.png'))))
        folder=class_path.split(os.path.sep)[-2]
        ya=folder.split('_')[7]
        za = folder.split('_')[9]
        model = folder.split('_')[10]
        emotion = folder.split('_')[11]
        target=int(class_path.split(os.path.sep)[-3])
        listp[foldkey[model]].append([class_path,target,ya,za,model,emotion,paths])
    ratio = [0.70, 0.15, 0.15]
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
        nop=0
        try:
            x = self.mtcn(x)
        except Exception as e:
            # print("<p>Error: %s</p>" % str(e))
            print(path)
            nop=1
        if not nop:
            newpath=path.split(os.path.sep)
            newpath[-4]='grab_cropped2'
            newpath[0]+='\\'
            newpath=os.path.join(*newpath)
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            for i in range(len(x)):
                image=x[i].permute(1,2,0).numpy()
                image=cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                cv2.imwrite(os.path.join(newpath,'{:d}.jpeg'.format(i+1)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return target, target

    def __len__(self):
        return self.lenD

#####################


if __name__ == '__main__':
    a, b, c = datPP3()
    TD1 = MakeH(a)
    TD2 = MakeH(b)
    TD3 = MakeH(c)
    concat=torch.utils.data.ConcatDataset([TD1,TD2,TD3])
    train_loader = torch.utils.data.DataLoader(
            concat,
            batch_size=4,
            num_workers=4, pin_memory=True)
    for i,(input, target) in enumerate(train_loader):
        print(i)
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


