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

def datPP3(root_dir='D:\\Work\\ml\\fold bandi\\',Ra=0):
    videos = glob.glob(root_dir + '*\\*\\*.mp4')
    listA = []
    listTr = []
    listVa = []
    dict={'ANGR':0,'DISG':1,'CONF':2}
    x = [x for x in range(5) if x!=Ra]
    for vid in videos:
        name=vid.split(os.path.sep)[-1]
        emotion=name.split('_')[-1].split('.')[-2]
        emotion_no=dict[emotion]
        cut=int(name.split('_')[-2])
        MF=int(name.split('_')[-4])
        fold=int(name.split('_')[-6])
        tmpinfo=[vid,emotion,emotion_no,cut,MF,fold]
        listA.append(tmpinfo)
        if fold==Ra-1:
            listVa.append(tmpinfo)
        else:
            listTr.append(tmpinfo)
    return listA, listTr,listVa


class MakeH(Dataset):
    def __init__(self,
                 clips,
                 transform=None
                 ):
        self.clips = clips
        self.transform = transform
        self.lenD=len(clips)


    def __getitem__(self, index):
        path, _,target,cut,MF,_= self.clips[index]

        images = []
        data = cv2.VideoCapture(path)

        # count the number of frames
        frames = int(data.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(data.get(cv2.CAP_PROP_FPS))
        frames=frames if cut==0 else frames-int(fps*5)
        for i in range(frames):
            ret, frame = data.read()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(torch.tensor(img))
        images=torch.stack(images)
        return images,target,MF,path


        # seed = np.random.randint(2147483646)
        # for i in range(1,self.length+1):
            #            print(i)
            # modedindex=i%duration
            # modedindex= duration if modedindex == 0 else modedindex

            # image_path = os.path.join(path+'{:d}.png'.format(modedindex))
            # image = pure_pil_alpha_to_color_v2(Image.open(image_path))
            # image=cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

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

            # images.append(torch.tensor(image))
            # images.append(image)


        # x = torch.stack(images)
        # images = []
        # nop=0
        # try:
        #     x = self.mtcn(x)
        # except Exception as e:
        #     # print("<p>Error: %s</p>" % str(e))
        #     print(path)
        #     nop=1
        # if not nop:
        #     newpath=path.split(os.path.sep)
        #     newpath[-4]='grab_cropped2'
        #     newpath[0]+='\\'
        #     newpath=os.path.join(*newpath)
        #     if not os.path.exists(newpath):
        #         os.makedirs(newpath)
        #     for i in range(len(x)):
        #         image=x[i].permute(1,2,0).numpy()
        #         image=cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #         cv2.imwrite(os.path.join(newpath,'{:d}.jpeg'.format(i+1)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # return target, target

    def __len__(self):
        return self.lenD

#####################


if __name__ == '__main__':
    a, b, c = datPP3(root_dir='D:\\Work\\ml\\fold bandi\\')
    mtcn = MTCNN(device=device,thresholds=[0.5,0.6,0.6],selection_method='probability')
    TD=MakeH(a)

    DataLoader=torch.utils.data.DataLoader(
            TD,
            batch_size=1,
            num_workers=0, pin_memory=True)
    for data in DataLoader:
        images=data[0].squeeze(0)
        path=data[-1][0]
        try:
            imgs_cut=mtcn(images)
        except Exception as e:
            imgs_cut=[]
            for no,img in enumerate(images):

                try:
                    img_cut = mtcn(img)
                    imgs_cut.append(img_cut)

                except Exception as e:

                    print(path+'Fno{}'.format(no))
                    # transform = transforms.Compose([transforms.ToPILImage(),
                    #     transforms.CenterCrop((int(img.shape[0]/2),int(img.shape[1]/2))),
                    #     transforms.Resize((160, 160)),
                    #     transforms.ToTensor(),
                    #     transforms.Normalize([0.5], [0.5]),
                    # ])
                    # img_tmp=transform(img.transpose(0,2)).transpose(0,2)
                    # imgs_cut.append(img_tmp)
                    # img_cut=
            if imgs_cut:
                imgs_cut=torch.stack(imgs_cut)

        new_path=path.split(os.sep)
        new_path[3] += '_cropped'
        new_path[-1]=new_path[-1].split('.')[0]
        new_path[0]+='\\'
        newpath=os.path.join(*new_path)+'\\'
        # name=new_path[-1].split('.')[0]
        # imgs_s=[]
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        for i in range(len(imgs_cut)):
            image=imgs_cut[i].permute(1,2,0).cpu().numpy()
            image=cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.imwrite(os.path.join(newpath,'{:d}.jpeg'.format(i+1)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # imgs_s.append(image)


        # imgs_s=np.array(imgs_s)
        # file = h5py.File(newpath + name+'.h5', "w")
        # dataset = file.create_dataset(
        #     "image", np.shape(imgs_s), h5py.h5t.STD_U8BE, data=imgs_s
        # )

    # TD1 = MakeH(a)
    # TD2 = MakeH(b)
    # TD3 = MakeH(c)
    # concat=torch.utils.data.ConcatDataset([TD1,TD2,TD3])
    # train_loader = torch.utils.data.DataLoader(
    #         concat,
    #         batch_size=4,
    #         num_workers=4, pin_memory=True)
    # for i,(input, target) in enumerate(train_loader):
    #     print(i)
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


