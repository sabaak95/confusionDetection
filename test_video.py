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

def video_loader(path,transform,mtcn,device):
    cap = cv2.VideoCapture(path)
    images=[]
    while (True):
        ret, frame = cap.read()
        if ret==True:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(torch.tensor(image))
        else:
            break

    cap.release()
    x = torch.stack(images)
    images = []
    try:
        x = torch.stack(mtcn(x))
    except Exception as e:
        # print("<p>Error: %s</p>" % str(e))
        print(path)
        for image in x:
            image = Image.fromarray(image.numpy())
            image = transform(image)
            images.append(image)
        x = torch.stack(images)

    return x
# torch.nn.Module.dump_patches = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
mtcn=MTCNN(device=device)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

path='D:\\Work\\ml\\test.mp4'

# model_path="D:\\Work\\ml\\log2--BS12--modeR1--patience5--lr0.0001.pth"
model_path="D:\\Work\\ml\\log2--BS72--modeR0--patience5--lr0.001.pth"

model=torch.load(model_path, map_location=torch.device('cpu'))
# model=model.to(device)
model.train(False)

input=video_loader(path,transform,mtcn,device)
input=torch.nn.functional.interpolate(input=input.permute(1,2,0,3),size=[25,160], mode='bilinear').permute(2,0,1,3)
print(input.shape)
with torch.set_grad_enabled(False):
    output= model(input.unsqueeze(0).to(device))
print(torch.sigmoid(output))
