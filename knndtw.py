import video_transforms
import numpy as np
from dataload_r import *
import argparse
# from utility import *
from models2 import *
from facenet_pytorch import MTCNN, fixed_image_standardization
import logging
import torch.optim.lr_scheduler as lrs
from torch.optim import Adam
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import jaccard_score, roc_curve, auc
import pandas as pd
from collections import OrderedDict
from dtaidistance import dtw_ndim
from tslearn.clustering import TimeSeriesKMeans
from tslearn.neighbors import KNeighborsTimeSeries,KNeighborsTimeSeriesClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import zero_one_loss,accuracy_score
from tslearn.metrics import dtw, soft_dtw

#############################################
parser = argparse.ArgumentParser()

parser.add_argument("--BS", type=int, default=8)  # 3 modeR==1 9 if modeR==0
parser.add_argument("--JobId", type=int, default=0)
parser.add_argument("--NN", type=int, default=1)
parser.add_argument("--hids", type=int, default=128)
parser.add_argument("--mode", type=int, default=0)
parser.add_argument("--Frz", type=int, default=1)
parser.add_argument("--Ra", type=int, default=0)
# parser.add_argument("--bidi", type=int, default=0)
parser.add_argument("--cont", type=int, default=0)
parser.add_argument("--reduction", type=int, default=4)
parser.add_argument("--reduction3", type=int, default=0)

parser.add_argument("--numepc", type=int, default=1)

parser.add_argument("--patience", type=int, default=3)  ####8,5,10
parser.add_argument("--Sc", type=int, default=1)  # 0,1

parser.add_argument("--wd", type=float, default=0.001)
parser.add_argument("--lr", type=float, default=0.0001)

args = parser.parse_args()
# width = 340
# height = 256


# width = 224
# height =224
###############################

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.shutdown()

name = 'D:\Work\ml\logs\logKNN' + (''.join(sys.argv[1:]))
if args.cont:
    logF = name + 'con' + '.csv'
    netS2 = name + '.pth'
    netS = name + 'con' + '.pth'
else:
    logF = name + '.csv'
    netS = name + '.pth'

if (os.path.isfile(logF)):
    os.remove(logF)
logging.basicConfig(filename=logF,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('myloger')

logger.info(sys.argv)
print(args)
logger.info(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    path='D:\\Downloads\\Compressed\\angry_f\\'
    tloss = 0
    vloss = 0
    teloss = 0
    tac = 0
    vac = 0
    teac = 0
    lgloss = {"train": tloss, "val": vloss, 'test': teloss}
    lgacc = {"train": tac, "val": vac, 'test': teac}
    SV = []
    ST= []
    Scorelist = {"train": ST, "val": SV, 'test': SV}
    L_t=[]
    L_v=[]
    Labellist = {"train": L_t, "val": L_v, 'test': L_v}
    best_acc = 0
    best_accV = 0
    b_loss = 0
    cc = 0
    bep = 0
    for file_path in glob.glob(path+'*.pth'):
        F=torch.load(file_path)
        target=file_path.split(os.sep)[-1].split('_')[-3]
        fold = file_path.split(os.sep)[-1].split('_')[-1].split('.')[0]
        if int(fold)==args.Ra+1:
            phase='val'
        else:
            phase = 'train'


        Scorelist[phase].append( F.numpy())
        Labellist[phase].append(int(target))
    for phase in ['train', 'val']:
        Scorelist[phase]=np.stack(Scorelist[phase])
        Labellist[phase]=np.stack(Labellist[phase])


sse = 1

K_MAX = 9
logger.info(',k,acc,pre,rec,fs,jc')
for k in range(1, K_MAX + 1):
    knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=k, metric="dtw")
    knn_clf.fit(Scorelist['train'], Labellist['train'])
    predicted_labels = knn_clf.predict(Scorelist['val'])
    acc = accuracy_score(Labellist['val'], predicted_labels)
    pre, rec, fs, _ = precision_recall_fscore_support(Labellist['val'],predicted_labels,average='macro')
    J_s = jaccard_score(Labellist['val'],predicted_labels,average='macro')
    logger.info(',{},{},{},{},{},{}'.format(k,acc,pre,rec,fs,J_s))
    if zero_one_loss(Labellist['val'],predicted_labels)<=sse:
        sse=zero_one_loss(Labellist['val'],predicted_labels)
        k_b = k
        preM=pre
        recM=rec
        fsM=fs
        J_sM=J_s
        accM=acc
    # sse.append(zero_one_loss(Labellist['val'],predicted_labels))
    # ssa.append(accuracy_score(Labellist['val'],predicted_labels))

logger.info(',{},{},{},{},{},{}'.format(k_b,accM,preM,recM,fsM,J_sM))


