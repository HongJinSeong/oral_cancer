from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import random
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import StratifiedKFold,KFold
from torchvision import  transforms
from torch.utils.data import DataLoader
import gc
import argparse
import torch
from torch.utils.data import Dataset
import glob as _glob
import os
import skimage.io as iio
import csv
import numpy as np
from PIL import Image
import os
import time
from utility.datasetutils import *
import matplotlib.pyplot as plt
print(os.getpid())

### device and seed
device ='cpu'
#device = 'cuda' if torch.cuda.is_available else 'cpu'

model = EfficientNet.from_pretrained('efficientnet-b7',num_classes=2)

parser = argparse.ArgumentParser()

# model arguments
parser.add_argument('--img_size', type=int, default=500,
                    help='Image resolution')
parser.add_argument('--datapaths', type=str, default='datasets/oralsamples/',
                    help='data path')
parser.add_argument('--pretrain_path', type=str, default='pretrained_model/efficient-b7-NC2.pt',
                    help='pretrain_path')
#parser.add_argument('--pretrain_path', type=str, default='outputs/checkpoints/0fold_best_validation.pt',
#                    help='pretrain_path')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.00001,
                    help='learning rate')
parser.add_argument('--EPOCH', type=int, default=100,
                    help='EPOCH')

args = parser.parse_args()


##### hyper parameters and paths
datapaths=args.datapaths
pretrain_path=args.pretrain_path
img_size=args.img_size          ### efficientnet B7 기준 600 * 600
learning_rate=args.lr             ## batchsize 128 기준일 때 0.256 (batchsize 줄이는만큼 learning rate 비율 줄이기)
learning_rate_decay_rate=0.97   ## batchsize 128 기준일 때
batch_size=args.batch_size
output_checkpoint_path='outputs/checkpoints/'
EPOCH=args.EPOCH
logpath='outputs/logs/'

# 데이터셋을 tensor type으로 변경 하고 normalize (==> pretrain시에 사용했던 normalize 값 그대로 사용)
tform_train=transforms.Compose([
    transforms.Resize(size=(525,525)),
    transforms.RandomResizedCrop(size=(img_size,img_size), scale=(0.8, 1.0), ratio=(0.8, 1.2), interpolation=2),
    transforms.RandomHorizontalFlip(),           ### 좌우(mirroring)
    transforms.ToTensor()
])

tform_test=transforms.Compose([
    transforms.Resize(size=(img_size,img_size)),
    transforms.ToTensor()
])

#model = EfficientNet.from_pretrained('efficientnet-b7',num_classes=2)
#torch.save(model.state_dict(),pretrain_path)


#dataset 불러오기 ===> train
clslist=[]    ## class list
Tds=[]        ## train 전체 data array
Tlabels=[]
ds_by_cls=[]
cls_LEN=[]
labels=[]
cls_by_folder=[]

# 암
cancer_cls_by_folder = glob(datapaths, '*_C.*', True)
ds_by_cls.append(cancer_cls_by_folder)
labels.append((np.ones(len(cancer_cls_by_folder))*0).tolist())

# 양성
Benign_cls_by_folder = glob(datapaths, '*_B.*', True)
ds_by_cls.append(Benign_cls_by_folder)
labels.append((np.ones(len(Benign_cls_by_folder))*1).tolist())

# 정상
normal_cls_by_folder = glob(datapaths, '*_N.*', True)
ds_by_cls[1]+=normal_cls_by_folder
labels[1]+=(np.ones(len(normal_cls_by_folder))*1).tolist()
for ds in ds_by_cls:
    cls_LEN.append(len(ds))



'''
cat_folder = glob('datasets/afhq/train/cat','*',True)
Tds+=cat_folder[0:1000]
Tlabels+=np.zeros(shape=1000).tolist()
dog_folder = glob('datasets/afhq/train/dog','*',True)
Tds+=dog_folder[0:1000]
Tlabels+=np.ones(shape=1000).tolist()
'''
## class 별 oversampling

for i,ds in enumerate(ds_by_cls):
    if len(ds) < max(cls_LEN):
        label_add = (np.ones(max(cls_LEN) - len(ds))*i).tolist()
        if len(ds)> max(cls_LEN)-len(ds):
            ds_add=np.random.choice(np.array(ds), max(cls_LEN) - len(ds), replace=False).tolist()

        else:
            ds_add = np.random.choice(np.array(ds), max(cls_LEN)  - len(ds), replace=True).tolist()
        ds_by_cls[i]+=ds_add
        labels[i]+=label_add
    Tlabels+=labels[i]
    Tds+=ds_by_cls[i]



TD_by_person=np.array(Tds)
TD_by_person=np.array(np.char.split(TD_by_person,'/'))

key=[]
for person in TD_by_person:
    key.append(person[-1].split('_')[0])

key = np.unique(key, return_counts=True, axis=0)


skf=KFold(n_splits=2,shuffle=True)


for idx,(train,test) in enumerate(skf.split(key[0])):
    train_ds = []
    test_ds = []
    if idx>0:
        del model
    ##### model load / pretrained load
    Tds=np.array(Tds)
    for train_idx in train:
        train_idxs=np.char.find(Tds,key[0][train_idx])
        train_idxs=np.where(train_idxs!=-1)
        train_ds+=Tds[train_idxs].tolist()
    for test_idx in test:
        test_idxs = np.char.find(Tds, key[0][test_idx])
        test_idxs = np.where(test_idxs != -1)
        test_ds += Tds[test_idxs].tolist()

    model = EfficientNet.from_name('efficientnet-b7', num_classes=2,dropout_rate=0.5)
    model = model.to(device)
    model.load_state_dict(torch.load(pretrain_path))
    #criterion=nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    #optimizer=optim.Adam(model.parameters(),lr=learning_rate,betas=[0.9,0.999],weight_decay=0.9)
    #optimizer=optim.RMSprop(model.parameters(),lr=learning_rate,weight_decay=0.9,momentum=0.9)
    base_optimizer=optim.Adam
    optimizer=SAM(model.parameters(),base_optimizer,lr=learning_rate,betas=[0.9,0.999],weight_decay=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.97)



    train_ds=oralDataset_NEW(train_ds,tform_train)
    test_ds=oralDataset_NEW(test_ds,tform_test)

    print(len(test_ds))
    print(len(train_ds))

    traindataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    testdataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    early_stopping = EarlyStopping(patience=7, verbose=True,
                                   path=output_checkpoint_path + str(idx) + 'fold_' + 'best_validation.pt')

    for ep in range(EPOCH):
        since = time.time()
        model.train()
        running_loss = 0
        running_acc = 0

        for i,datas in enumerate(traindataloader):
            names,imgs,labels=datas
            '''
            d1 = imgs[0].permute(1, 2, 0).numpy()
            fig = plt.figure(figsize=(30, 30))
            plt.subplot(1, 1, 1)
            plt.imshow(d1)
            d1 = imgs[1].permute(1, 2, 0).numpy()
            fig = plt.figure(figsize=(30, 30))
            plt.subplot(1, 1, 1)
            plt.imshow(d1)

            plt.show()
            '''
            imgs=imgs.to(device)
            labels=labels.to(device)

            # paramter중에 alpha값은 계수의 의미가 아닌 확률분포의 범위를 의미
            mixed_inputs, label1, label2, lam = mixup_data(imgs, labels, alpha=1)

            outputs=model(imgs)

            #loss=criterion(outputs,labels.long())

            loss = mixup_criterion(criterion, outputs, label1.long(), label2.long(),
                                   lam)
            loss.backward()
            #optimizer.step()
            optimizer.first_step(zero_grad=True)
            outputs = model(mixed_inputs)
            loss = mixup_criterion(criterion, outputs, label1.long(), label2.long(),
                                   lam)

            loss.backward()
            optimizer.second_step(zero_grad=True)

            _, preds = torch.max(outputs, 1)

            prediction = torch.nn.functional.softmax(outputs, dim=1)
            print(str(labels.tolist()[0])+'   vs   '+str(labels.tolist()[1]))
            print(prediction)

            # 현재 loss & accuracy
            now_loss = loss.item()
            now_acc = (lam * preds.eq(label1.data).cpu().sum().float() + (1 - lam) * preds.eq(
                    label2.data).sum().float())
            print(i)
            print('LOSS: ' + str(now_loss))
            if i!=0 and i%20==0:
                print('Accuracy: '+str(now_acc)+'    LOSS: '+str(now_loss))
                #print('LOSS: ' + str(now_loss))
                time_elapsed = time.time() - since
                print(str(i)+' iter complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))

            running_acc+=now_acc
            running_loss+=now_loss
            del imgs,labels,outputs,loss, mixed_inputs, label1, label2, lam, _, preds, now_loss, now_acc
        exp_lr_scheduler.step()
        writecsv(logpath + str(idx) +'fold_train_accloss.csv', [running_acc.double() / len(train_ds),running_loss/len(train_ds)])
        print(str(idx)+' fold '+str(ep)+' epoch train  '+'{} Loss: {:.4f} Acc: {:.4f}'.format(
            'training', running_loss/len(train_ds), running_acc.double() / len(train_ds)))

        running_validation_loss = 0.0
        running_validation_acc = 0.0
        model.eval()
        with torch.no_grad():
            for i,datas in enumerate(testdataloader):

                names, imgs, labels = datas
                '''
                if i > 50:
                    d1 = imgs[0].permute(1, 2, 0).numpy()
                    fig = plt.figure(figsize=(30, 30))
                    plt.subplot(1, 1, 1)
                    plt.imshow(d1)
                    d1 = imgs[1].permute(1, 2, 0).numpy()
                    fig = plt.figure(figsize=(30, 30))
                    plt.subplot(1, 1, 1)
                    plt.imshow(d1)

                    plt.show()
                '''
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs=model(imgs)

                loss = criterion(outputs,
                                 labels.long())
                print('valid loss: ' + str(loss))
                running_validation_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                running_validation_acc += torch.sum(preds == labels.data)
                del imgs,labels,outputs,loss
            print(str(running_validation_loss)+'   '+str(len(test_ds)))
            print('valid loss: '+str((running_validation_loss / len(test_ds))))
            writecsv(logpath + str(idx) + 'fold_valid_accloss.csv',
                     [running_validation_acc.double() / len(test_ds), running_validation_loss / len(test_ds)])
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        early_stopping(running_validation_loss / len(test_ds), model)
        gc.collect()
        if early_stopping.early_stop:
            print(str(idx) +'model early stopping')
            break