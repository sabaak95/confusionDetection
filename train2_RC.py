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
import csv

#############################################
parser = argparse.ArgumentParser()


parser.add_argument("--BS", type=int, default=64)  # 3 modeR==1 9 if modeR==0
parser.add_argument("--JobId", type=int, default=0)
parser.add_argument("--NN", type=int, default=1)
parser.add_argument("--hids", type=int, default=128)
parser.add_argument("--mode", type=int, default=0)
parser.add_argument("--Frz", type=int, default=1)
parser.add_argument("--FrzN", type=int, default=1)
# parser.add_argument("--bidi", type=int, default=0)
parser.add_argument("--cont", type=int, default=1)
parser.add_argument("--reduction", type=int, default=4)
parser.add_argument("--reduction3", type=int, default=0)
parser.add_argument("--Ra", type=int, default=0)
parser.add_argument("--Len", type=int, default=64)
parser.add_argument("--numepc", type=int, default=100)

parser.add_argument("--patience", type=int, default=5)  ####8,5,10
parser.add_argument("--Sc", type=int, default=1)  # 0,1


parser.add_argument("--wd", type=float, default=0.001)
parser.add_argument("--lr", type=float, default=0.0001)



# width = 340
# height = 256


# width = 224
# height =224
###############################

def reslog(line):
    # with open(r'/home/mabbasib/MLN/logs/acc.csv', 'a', newline='') as f:
    with open(r'/home/mabbasib/MLN/logs/accRC.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(line)
        f.close()




if __name__ == '__main__':
    args = parser.parse_args()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.shutdown()
    
    name = '/scratch/mabbasib/FCLOGSRC/finetune-FrzN1' + (''.join(sys.argv[1:]))
    #name = '.\\logs\\finetune' + (''.join(sys.argv[1:]))
    logF = name + '.csv'
    netS = name + '.pth'
    if args.cont:
    
        #netS2 = name = '/scratch/mabbasib/FCLOGSI/Synth--Ra0' + (''.join(sys.argv[5:])) +'.pth'
        netS2 = name = '/scratch/mabbasib/FCLOGSI/Synth--Ra0' + (''.join(sys.argv[3:])) +'.pth'
    
    
    
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
    net = CNNTemporal(device=device,n_cl=3, hids=args.hids, mode=args.mode, reduction3=args.reduction3,reduction=args.reduction,Frz=args.Frz if not args.cont else args.FrzN)
    net = net.to(device)
    if args.cont:
        state_dict=torch.load(netS2, map_location="cpu")
        net.load_state_dict(state_dict)
        #net = torch.load(netS2)

    ff=args.Frz if not args.cont else args.FrzN
    batchsize= args.BS if ff else args.BS//8
    #net = net.to(device)
    ############################
    transform = transforms.Compose([
                # transforms.CenterCrop(270),
                # transforms.Resize((160, 160)),
                transforms.RandomChoice([
                transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                #transforms.RandomAffine(5, translate=(0.05, 0.05), scale=None, shear=5, resample=Image.NEAREST,
                #                                               fillcolor= 0),
                #transforms.RandomResizedCrop(160, scale=(0.9, 1.0), ratio=(0.95, 1.15))]),
                ]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
    transformV = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    a, b, c = datPP3_r(root_dir='/scratch/mabbasib/MLD/',Ra=args.Ra)

    DT_tr = MakeH_r(b, transform=transform, val=0,Len=args.Len)
    DT_va = MakeH_r(c, transform=transformV, val=1,Len=args.Len)
   
    #DT_tr = MakeH_r(b, transform=transform, val=0,Len=args.Len)
    #DT_va = MakeH_r(c, transform=transformV, val=1,Len=args.Len)
    print(len(DT_tr))
    print(len(DT_va))
    train_loader = torch.utils.data.DataLoader(
            DT_tr,
            batch_size=batchsize,
            num_workers=6, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        DT_va,
        batch_size=batchsize,
        num_workers=6, pin_memory=True)

    ########################

    optimizer = Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)

    loss_fn = nn.CrossEntropyLoss()

    scheduler = lrs.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=args.patience, min_lr=0.000001)
    ######################################
    data_loaders = {"train": train_loader, "val": val_loader, 'test': DT_tr}
    data_lengths = {"train": len(DT_tr), "val": len(DT_va), 'test': len(DT_tr)}

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
    # mtcn2 = MTCNN(device=device)
    ######################################################################
    num_epochs=args.numepc
    logger.info(',trainloss,validation loss,test loss,train acc,vali acc,test acc,lr,highest,diff,percision,recall,fscore,jaccardIn')

    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
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
                # images = images.contiguous().view(-1, IS[-3], IS[-2], IS[-1])


                # images = torch.stack(mtcn2(images)).view(IS[0],IS[1],IS[-1],160,160)
                images=images.to(device)
                # del mtcn2
                # torch.cuda.empty_cache()

                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    outputs,_ = net(images)


                    loss = loss_fn(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    elif phase == 'val':
                        Scorelist.append(torch.cat([labels.float().cpu().unsqueeze(dim=1), outputs.cpu(),torch.max(outputs.data, 1)[1].cpu().unsqueeze(1).to(dtype=torch.float32)], dim=1))

                running_loss += loss.data.item() * float(len(data[1]))

                _, prediction = torch.max(outputs.data, 1)

                train_acc += torch.sum(prediction == labels)

                # print(float(torch.sum(prediction == labels)) / float(len(data[1])))
                print(loss.data.item() * float(len(data[1])))
            epoch_loss = float(running_loss) / float(data_lengths[phase])
            if ((args.Sc == 1) & (phase == 'val')):

                scheduler.step(epoch_loss)

            t_a = float(train_acc) / float(data_lengths[phase])
            print('{} Loss: {:} Acc:{}'.format(phase, epoch_loss, t_a))

            lgloss[phase] = epoch_loss
            lgacc[phase] = t_a
            if phase == 'val':

                for param_group in optimizer.param_groups:
                    print(param_group['lr'])

                if (lgacc["val"] > best_accV) or ((lgacc["val"] == best_accV) and (b_loss >= lgloss['val'])) and epoch >= 0:
                    torch.save(net.state_dict(), netS)
                    bep = epoch
                    print((t_a - best_acc))
                    ScorelistA = torch.cat(Scorelist)
                    x_df = pd.DataFrame(ScorelistA.numpy())
                    x_df.to_csv(name+'scorlist.csv')
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


                       
    reslog([args.BS,args.JobId,args.NN,args.hids,args.mode,args.Frz,args.FrzN,args.cont,args.reduction,args.reduction3,args.Ra,args.Len,args.numepc,args.patience,args.Sc,args.wd,args.lr,111,preM, recM, fsM,J_s,best_acc])