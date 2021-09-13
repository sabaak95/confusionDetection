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
from facenet_pytorch import MTCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class MakeH_gifi(Dataset):
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
        path,target,duration= self.clips[index]

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
        return images,target,int(path.split(os.sep)[-1])

    def __len__(self):
        return len(self.clips)

def datPP3_gifi(root_dir='D:/Mehryar/MLN/gifigifi_cropped'):
    videos = glob.glob(root_dir + '/*/*')
    listA = []
    dictT = {'anger': 0, 'disgust': 1}
    for vid in videos:
        name=vid.split(os.path.sep)[-1]
        duration=len(glob.glob(vid+'/*.jpeg'))

        target=dictT[vid.split(os.sep)[-2]]

        tmpinfo=[vid,target,duration]
        listA.append(tmpinfo)
    return listA

class MakeH(Dataset):
    def __init__(self,
                 clips,
                 transform=None
                 ):
        self.clips = glob.glob(clips+'/*/*.gif')
        self.transform = transform
        self.lenD=len(glob.glob(clips+'/*/*.gif'))
        self.dict={'anger':0,'disgust':1}

    def __getitem__(self, index):
        path=self.clips[index]
        target=self.dict[path.split(os.sep)[-2]]
        images = Image.open(path)
        # data = cv2.VideoCapture(path)
        iml=[]
        # images=[]
        # count the number of frames
        # frames = int(data.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps = int(data.get(cv2.CAP_PROP_FPS))
        # frames=frames if cut==0 else frames-int(fps*5)
        # for i in range(frames):
        #     ret, frame = data.read()
        #     if ret:
        #         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #         images.append(torch.tensor(img))
        # images=torch.stack(images)
        # return images,target,path
        for i in range(images.n_frames):
            images.seek(i)
            frame=Image.new("RGB", images.size)
            frame.paste(images)
            # frame=frame.resize((frame.size[0]*3,frame.size[1]*3))
            iml.append(torch.tensor(np.array(frame)))
        iml=torch.stack(iml)
        return iml,target,path

    def __len__(self):
        return self.lenD

if __name__ == '__main__':




    mtcn = MTCNN(device=device, thresholds=[0.5, 0.6, 0.6], selection_method='probability')
    TD = MakeH(clips='D:/Mehryar/MLN/gifigifi/')

    DataLoader = torch.utils.data.DataLoader(
        TD,
        batch_size=1,
        num_workers=0, pin_memory=True)
    for data in DataLoader:
        images = data[0].squeeze(0)
        target=data[1][0]
        path = data[-1][0]
        try:
            imgs_cut = mtcn(images)
        except Exception as e:
            imgs_cut = []
            for no, img in enumerate(images):

                try:
                    img_cut = mtcn(img)
                    imgs_cut.append(img_cut)

                except Exception as e:

                    print(path + 'Fno{}'.format(no))
                    # transform = transforms.Compose([transforms.ToPILImage(),
                    #     transforms.CenterCrop((int(img.shape[0]/2),int(img.shape[1]/2))),
                    #     transforms.Resize((160, 160)),
                    #     transforms.ToTensor(),
                    #     transforms.Normalize([0.5], [0.5]),
                    # ])
                    # img_tmp=transform(img.transpose(0,2)).transpose(0,2)
                    # imgs_cut.append(img_tmp)
                    # img_cut=
            if not len(imgs_cut)==0:
                imgs_cut = torch.stack(imgs_cut)
        if not len(imgs_cut) == 0:
            new_path = path.split(os.sep)
            new_path[-3] += '_cropped'
            new_path[-1] = new_path[-1].split('.')[0]
            # new_path.insert(-1,target)
            new_path[0] += '\\'
            newpath = os.path.join(*new_path) + '\\'
            # name=new_path[-1].split('.')[0]
            # imgs_s=[]
            if not os.path.exists(newpath):
                os.makedirs(newpath)
                print('newpath+------------'+newpath)

            for i in range(len(imgs_cut)):
                image = imgs_cut[i].permute(1, 2, 0).cpu().numpy()
                image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                cv2.imwrite(os.path.join(newpath, '{:d}.jpeg'.format(i + 1)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))