import numpy as np
from sklearn.metrics import jaccard_score
import glob
import torch
from torch.utils.data import Dataset
import os
import sys
import numpy as np

path='D:\\Work\\paper video read\\scrolist\\1\\'

paths=glob.glob(path+'*.pt')
j_i=[]
for pt in paths:
    scor=torch.load(pt)
    y_true=scor[:,0]
    y_pred=torch.max(scor[:,1:],1)[1]
    j_i.append(jaccard_score(y_true, y_pred, average='macro'))

J=np.mean(np.array(j_i))
print(J)