# import video_transforms
import numpy as np
# from dataload_r import *
# from dataload_2 import *
import os
import argparse
# from utility import *
from models3 import *
from cut_gifi import  datPP3_gifi,MakeH_gifi
from facenet_pytorch import MTCNN, fixed_image_standardization
import logging
import torch.optim.lr_scheduler as lrs
from torch.optim import Adam
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from sklearn.metrics import jaccard_score, roc_curve, auc
import pandas as pd
from collections import OrderedDict
import csv
# from pytorch_i3d import rgb_I3D64f
import sys
import torchvision.transforms as transforms
from PIL import Image
#############################################
parser = argparse.ArgumentParser()

parser.add_argument("--BS", type=int, default=64)  # 3 modeR==1 9 if modeR==0
parser.add_argument("--JobId", type=int, default=0)
parser.add_argument("--NN", type=int, default=1)
parser.add_argument("--hids", type=int, default=128)
parser.add_argument("--mode", type=int, default=5)
parser.add_argument("--Frz", type=int, default=0)
# parser.add_argument("--bidi", type=int, default=0)

parser.add_argument("--reduction", type=int, default=4)
parser.add_argument("--reduction3", type=int, default=0)


parser.add_argument("--numepc", type=int, default=45)

parser.add_argument("--patience", type=int, default=5)  ####8,5,10
parser.add_argument("--Sc", type=int, default=1)  # 0,1


parser.add_argument("--wd", type=float, default=0.001)
parser.add_argument("--lr", type=float, default=0.0001)

parser.add_argument("--ratio", type=float, default=1)
parser.add_argument("--Ra", type=int, default=1)
parser.add_argument("--JI0", type=int, default=222)
parser.add_argument("--frz0", type=int, default=1)
parser.add_argument("--cont", type=int, default=1)
parser.add_argument("--Len", type=int, default=25)

# width = 340
# height = 256


# width = 224
# height =224
###############################
# D:\Work\Anaconda3\envs\py362\python.exe D:/Work/ml/eval_elder.py --cont 0 --Ra 1
# D:\Work\Anaconda3\envs\py362\python.exe D:/Work/ml/eval_elder.py --cont 1 --Ra 1
# D:\Work\Anaconda3\envs\py362\python.exe D:/Work/ml/eval_elder.py --cont 0 --Ra 2
# D:\Work\Anaconda3\envs\py362\python.exe D:/Work/ml/eval_elder.py --cont 1 --Ra 2
# D:\Work\Anaconda3\envs\py362\python.exe D:/Work/ml/eval_elder.py --cont 0 --Ra 3
# D:\Work\Anaconda3\envs\py362\python.exe D:/Work/ml/eval_elder.py --cont 1 --Ra 3
# D:\Work\Anaconda3\envs\py362\python.exe D:/Work/ml/eval_elder.py --cont 0 --Ra 4
# D:\Work\Anaconda3\envs\py362\python.exe D:/Work/ml/eval_elder.py --cont 1 --Ra 4
# D:\Work\Anaconda3\envs\py362\python.exe D:/Work/ml/eval_elder.py --cont 0 --Ra 5
# D:\Work\Anaconda3\envs\py362\python.exe D:/Work/ml/eval_elder.py --cont 1 --Ra 5


def reslog(line):
    # with open(r'/home/mabbasib/MLN/logs/acc.csv', 'a', newline='') as f:
    with open(r'/home/mabbasib/MLN/logs/logg_gifi_clean.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(line)
        f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.shutdown()

    # name = '/scratch/mabbasib/FCLOGSRI_F_9/RealSynth1' + (''.join(sys.argv[1:]))
    if args.cont:
        # name = 'D:/Mehryar/MLN/models/RealSynth1--Ra{}--mode5--Frz{}--JobId{}--Len64--ratio0.23'.format(args.Ra,args.frz0,args.JI0)
        if args.ratio:
            name="/scratch/mabbasib/FCLOGSRI_F_9/RealSynth1--Ra{}--mode5--Frz0--JobId{}--Len{}--ratio{}".format(args.Ra,args.JI0,args.Len,args.ratio)
        

        else:
            if args.frz0:
                name="/scratch/mabbasib/FCLOGS_F_RC3_9/finetune2--Ra{}--FrzN1--cont1--Frz1--mode5--JobId{}--Len{}".format(args.Ra,args.JI0,args.Len)
            else:
                name="/scratch/mabbasib/FCLOGS_F_RC2_9/finetune2--Ra{}--FrzN0--cont1--Frz0--mode5--JobId{}--Len{}".format(args.Ra,args.JI0,args.Len)
                
    else:
        if args.frz0:
            name="/scratch/mabbasib/FCLOGS_F_RC3_9/finetune2--Ra{}--FrzN1--cont0--Frz1--mode5--JobId{}--Len{}".format(args.Ra,args.JI0,args.Len)
        else:
            name="/scratch/mabbasib/FCLOGS_F_RC2_9/finetune2--Ra{}--FrzN0--cont0--Frz0--mode5--JobId{}--Len{}".format(args.Ra,args.JI0,args.Len)
    # name = '.\\logs\\RealSynth' + (''.join(sys.argv[1:]))
    netS =name +'.pth'
    name1='/scratch/mabbasib/eval_gifi_clean/eval_gifi' + (''.join(sys.argv[1:]))
    # if args.cont:
    #     logF = name + 'con' + '.csv'
    #     netS2 = name + '.pth'
    #     netS = name + 'con' + '.pth'
    # else:
    #     logF = name + '.csv'
    #     netS = name + '.pth'


    print(args)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if args.cont:
    #     net = torch.load(netS2)
    # else:
    #     if args.NN == 33:
    #         net = rgb_I3D64f(num_classes=8, modelPath='./weights/rgb_imagenet.pth')
    #     else:
    net = CNNTemporal(device=device, n_cl=3, hids=args.hids, mode=args.mode, reduction3=args.reduction3,
                      reduction=args.reduction, Frz=args.Frz)
    state_dict = torch.load(netS, map_location="cpu")
    net.load_state_dict(state_dict)

    batchsize = args.BS if args.Frz else args.BS // 8
    if args.NN == 33:
        batchsize * 3 // 2
    net = net.to(device)
    ############################
    if args.NN == 33:
        transform = transforms.Compose([
            # transforms.CenterCrop(270),
            transforms.Resize((224, 224)),
            transforms.RandomChoice([
                transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                transforms.RandomAffine(5, translate=(0.05, 0.05), scale=None, shear=5, resample=Image.NEAREST,
                                        fillcolor=0),
                transforms.RandomResizedCrop((224, 224), scale=(0.9, 1.0), ratio=(0.95, 1.15))]),
            # ]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        transformV = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5], [0.5]),
                                         ])
    else:
        transform = transforms.Compose([
            # transforms.CenterCrop(270),
            # transforms.Resize((160, 160)),
            transforms.RandomChoice([
                transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                transforms.RandomAffine(5, translate=(0.05, 0.05), scale=None, shear=5, resample=Image.NEAREST,
                                        fillcolor=0),
                transforms.RandomResizedCrop(160, scale=(0.9, 1.0), ratio=(0.95, 1.15))]),
            # ]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        transformV = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    listA=datPP3_gifi('./gifigifi_cropped_clean/')
    DT_va = MakeH_gifi(listA, transform=transformV, val=1, Len=args.Len)



    val_loader = torch.utils.data.DataLoader(
        DT_va,
        batch_size=batchsize,
        num_workers=0, pin_memory=True)

    ########################

    optimizer = Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

    loss_fn = nn.CrossEntropyLoss()

    scheduler = lrs.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=args.patience, min_lr=0.000001)
    ######################################
    data_loaders = {"val": val_loader}
    data_lengths = {"val": len(DT_va)}

    tloss = 0
    vloss = 0
    teloss = 0
    tac = 0
    vac = 0
    teac = 0
    lgloss = {"train": tloss, "val": vloss, 'test': teloss}
    lgacc = {"train": tac, "val": vac, 'test': teac}
    best_acc = 0
    best_accV = 0
    b_loss = 0
    cc = 0
    bep = 0


    for epoch in range(1):
        # print('Epoch{}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['val']:
            if phase == 'train':
                net.train(True)  # Set model to training mode
            else:
                net.train(False)  # Set model to evaluate mode
                Scorelist = []

            running_loss = 0.0
            train_acc = 0
            train_acc2 = 0
            t_a_e = 0
            for data in data_loaders[phase]:

                images = data[0]
                labels = data[1]
                indexs= data[2]
                IS = images.shape
                # images = images.contiguous().view(-1, IS[-3], IS[-2], IS[-1])

                # images = torch.stack(mtcn2(images)).view(IS[0],IS[1],IS[-1],160,160)
                images = images.to(device)
                # del mtcn2
                # torch.cuda.empty_cache()

                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    if args.NN == 33:
                        outputs = net(images.transpose(1, 2))
                    else:
                        outputs, _ = net(images)
                    torch.index_select(outputs, 1, torch.LongTensor([0, 2, 1]).to(device))
                    loss = loss_fn(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    elif phase == 'val':
                        Scorelist.append(torch.cat([indexs.float().cpu().unsqueeze(dim=1),labels.float().cpu().unsqueeze(dim=1), outputs.cpu(),
                                                    torch.max(outputs.data[:,:], 1)[1].cpu().unsqueeze(1).to(
                                                        dtype=torch.float32),torch.max(outputs.data[:,:2], 1)[1].cpu().unsqueeze(1).to(
                                                        dtype=torch.float32)], dim=1))

                running_loss += loss.data.item() * float(len(data[1]))

                _, prediction = torch.max(outputs.data[:,:], 1)
                _, prediction2 = torch.max(outputs.data[:,:2], 1)

                train_acc += torch.sum(prediction == labels)
                train_acc2 += torch.sum(prediction2 == labels)

                # print(float(torch.sum(prediction == labels)) / float(len(data[1])))
                print(loss.data.item() * float(len(data[1])))
            epoch_loss = float(running_loss) / float(data_lengths[phase])
            if ((args.Sc == 1) & (phase == 'val')):
                scheduler.step(epoch_loss)

            t_a = float(train_acc) / float(data_lengths[phase])
            t_a2 = float(train_acc2) / float(data_lengths[phase])
            print('{} Loss: {:} Acc:{}'.format(phase, epoch_loss, t_a))

            lgloss[phase] = epoch_loss
            lgacc[phase] = t_a
            if phase == 'val':

                for param_group in optimizer.param_groups:
                    print(param_group['lr'])

                if (lgacc["val"] > best_accV) or (
                        (lgacc["val"] == best_accV) and (b_loss >= lgloss['val'])) and epoch >= 0:
                    # torch.save(net.state_dict(), netS)
                    bep = epoch
                    print((t_a - best_acc))
                    ScorelistA = torch.cat(Scorelist)
                    x_df = pd.DataFrame(ScorelistA.numpy())
                    x_df.to_csv(name1 + 'scorlist.csv')
                    y_true = ScorelistA[:, 1]
                    y_pred = ScorelistA[:, 1:]
                    # _, c_pred = torch.max(y_pred, 1)
                    c_pred = ScorelistA[:, -2]
                    c_pred2 = ScorelistA[:, -1]
                    preM, recM, fsM, _ = precision_recall_fscore_support(np.array(y_true),
                                                                         np.array(c_pred),
                                                                         labels=[0,1,2],average=None)
                    J_s = jaccard_score(np.array(y_true),
                                        np.array(c_pred),
                                        labels=[0,1,2],average=None)
                                        
                    preM2, recM2, fsM2, _ = precision_recall_fscore_support(np.array(y_true),
                                                                         np.array(c_pred2),
                                                                         average=None)
                    J_s2 = jaccard_score(np.array(y_true),
                                        np.array(c_pred2),
                                        average=None)

                    best_acc = t_a
                    best_accV = lgacc["val"]
                    b_loss = lgloss['val']







        # DT_trS.update()

    reslog([args.Ra,args.cont,args.JI0,args.Len,args.ratio,args.frz0,111,  preM[0], recM[0],
            fsM[0], J_s[0],222,preM[1], recM[1],
            fsM[1], J_s[1], best_acc,333,preM2[0], recM2[0],
            fsM2[0], J_s2[0],444,preM2[1], recM2[1],
            fsM2[1], J_s2[1], t_a2,555])