import video_transforms
import numpy as np
from dataload_2 import *
import argparse
# from utility import *
from models import *
from facenet_pytorch import MTCNN, fixed_image_standardization
import logging
import torch.optim.lr_scheduler as lrs
from torch.optim import Adam
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import jaccard_score, roc_curve, auc
# import pandas as pd
from collections import OrderedDict

#############################################
parser = argparse.ArgumentParser()


parser.add_argument("--BS", type=int, default=9)  # 3 modeR==1 9 if modeR==0
parser.add_argument("--JobId", type=int, default=0)
parser.add_argument("--NN", type=int, default=3)
parser.add_argument("--hids", type=int, default=128)
parser.add_argument("--modeR", type=int, default=0)
parser.add_argument("--bidi", type=int, default=0)


parser.add_argument("--numepc", type=int, default=100)

parser.add_argument("--patience", type=int, default=8)  ####8,5,10
parser.add_argument("--Sc", type=int, default=0)  # 0,1


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

name = 'D:\Work\ml\log_' + (''.join(sys.argv[1:]))
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
    if args.NN == 1:
        net = incLstm(n_cl=2, hids=args.hids, mode=args.modeR, bidi=args.bidi)
    elif args.NN == 2:
        net = cnnFull(n_cl=2, hids=args.hids, mode=args.modeR, bidi=args.bidi)
    elif args.NN == 3:
        net = cnnCov1(n_cl=2, hids=args.hids, mode=args.modeR, bidi=args.bidi)

    net = net.to(device)
    ############################
    D_T, D_V, D_Te = datPP3(root_dir='C:\\Users\\abbas\\OneDrive\\Documents\\makehuman\\v1py3\\grab\\')
    DT_loader = MakeH2(D_T)
    DV_loader = MakeH2(D_V)
    DTe_loader = MakeH(D_Te)
    train_loader = torch.utils.data.DataLoader(
            DT_loader,
            batch_size=args.BS,
            num_workers=0, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        DV_loader,
        batch_size=args.BS,
        num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        DTe_loader,
        batch_size=args.BS,
        num_workers=0, pin_memory=True)
    ########################

    optimizer = Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

    loss_fn = nn.CrossEntropyLoss()

    scheduler = lrs.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=args.patience, min_lr=0.000001)
    ######################################
    data_loaders = {"train": train_loader, "val": val_loader, 'test': test_loader}
    data_lengths = {"train": len(DT_loader), "val": len(DV_loader), 'test': len(DTe_loader)}

    tloss = 0
    vloss = 0
    teloss = 0
    tac = 0
    vac = 0
    teac=0
    lgloss = {"train": tloss, "val": vloss ,'test': teloss}
    lgacc = {"train": tac, "val": vac, 'test':teac}
    best_acc = 0
    best_accV = 0
    b_loss = 0
    cc = 0
    bep = 0
    mtcn2 = MTCNN(device=device)
    ######################################################################
    num_epochs=args.numepc
    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val','test']:
            if phase == 'train':
                net.train(True)  # Set model to training mode
            else:
                net.train(False)  # Set model to evaluate mode
                Scorelist = []

            running_loss = 0.0
            train_acc = 0
            t_a_e = 0
            for data in data_loaders[phase]:

                images = data[0]
                labels = data[1]
                IS = images.shape
                images = images.contiguous().view(-1, IS[-3], IS[-2], IS[-1])


                images = torch.stack(mtcn2(images)).view(IS[0],IS[1],IS[-1],160,160)
                images=images.to(device)
                # del mtcn2
                # torch.cuda.empty_cache()

                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs = net(images)


                    loss = loss_fn(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        Scorelist.append(torch.cat([labels.float().cpu().unsqueeze(dim=1), outputs.cpu()], dim=1))

                running_loss += loss.data.item() * float(len(data[1]))

                _, prediction = torch.max(outputs.data, 1)

                train_acc += torch.sum(prediction == labels)

                # print(float(torch.sum(prediction == labels)) / float(len(data[1])))
                print(loss.data.item() * float(len(data[1])))
            epoch_loss = float(running_loss) / float(len(data_lengths[phase]))
            if ((args.Sc == 1) & (phase == 'val')):

                scheduler.step(epoch_loss)

            t_a = float(train_acc) / float(len(data_lengths[phase]))
            print('{} Loss: {:} Acc:{}'.format(phase, epoch_loss, t_a))

            lgloss[phase] = epoch_loss
            lgacc[phase] = t_a
            if phase == 'test':

                for param_group in optimizer.param_groups:
                    print(param_group['lr'])

                if (lgacc["val"] > best_accV) or ((lgacc["val"] == best_accV) and (b_loss >= lgloss['val'])) and epoch >= 0:
                    torch.save(net, netS)
                    bep = epoch
                    print((t_a - best_acc))
                    ScorelistA = torch.cat(Scorelist)
                    y_true = ScorelistA[:, 0]
                    y_pred = ScorelistA[:, 1:]
                    _, c_pred = torch.max(y_pred, 1)
                    preM, recM, fsM, _ = precision_recall_fscore_support(np.array(y_true),
                                                                         np.array(c_pred),
                                                                         average='macro')
                    J_s = jaccard_score(np.array(y_true),
                                        np.array(c_pred),
                                        average='macro')
                    logger.info(
                        ',{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(lgloss["train"], lgloss["val"], lgloss["test"], lgacc["train"], lgacc["val"], lgacc["test"],
                                                        param_group['lr'],t_a, (t_a - best_acc),preM, recM, fsM,J_s))
                    best_acc = t_a
                    best_accV = lgacc["val"]
                    b_loss = lgloss['val']





                else:
                    logger.info(
                        ',{},{},{},{},{},{},{},{}'.format(lgloss["train"], lgloss["val"], lgloss["test"],
                                                             lgacc["train"], lgacc["val"], lgacc["test"],
                                                             param_group['lr'], best_acc))
                    if epoch > 10:
                        cc = cc + 1
                    if cc > num_epochs:
                        break