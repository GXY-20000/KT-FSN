import argparse
from model import *
from utility.iofile import *
from utility.selfdefine import *
from utility.preprocessing import sparse_to_tensor
from utility.collate import mycollate
from torchvision import transforms
from torch import  optim
import time
from sklearn.metrics import roc_auc_score

import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser(description='GCN for chestXray')
parser.add_argument('--batch-size', type=int, default=8, metavar='N', help='input batch size for training (default: 16)')
#parser.add_argument('--path', default='/media/ChestXray14/', type=str, help='data path')           #ChestXray14
parser.add_argument('--path', default='/media/CheXpert-v1.0-small', type=str, help='data path')     #CheXpert
parser.add_argument('--epochs', type=int, default=15, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='N', help='random seed (default: 1)')
parser.add_argument('--gpu', type=int, default=0, metavar='N', help='the GPU number (default auto schedule)')
parser.add_argument('-e','--encoder', default='vgg16bn', type=str, help='the encoder')
parser.add_argument('-r','--relations', default='all', type=str, help='the considered relations, pid, age, gender, view')
parser.add_argument('--use', default='train', type=str, help='train or test (default train)')
parser.add_argument('-m','--mode', default='RGB', type=str, help='the mode of the image')
parser.add_argument('-s','--neibor', default='relation', type=str, help='the neighbor sampling method (default: relation)')
parser.add_argument('--k',type=int, default=16,  metavar='N', help='the number of neighbors sampling (default: 16)')
parser.add_argument('-p','--train-percent',type=float, default=0.7,  metavar='N', help='the percentage of training data (default: 0.7)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',  help='manual epoch number (useful on restarts)')
parser.add_argument('-d','--weight_decay',type=float, default=0,  metavar='N', help='the percentage of training data (default: 0)')

args = parser.parse_args()
seed = args.seed
torch.manual_seed(seed)
batch_size = args.batch_size
enc = 'single'+args.encoder
neib = args.neibor
k = args.k
inchannel = 3 if args.mode=='RGB' else 1
mode = args.mode
tr_pct = args.train_percent
use = args.use
wd=args.weight_decay
relations = ['pid', 'age', 'gender', 'view'] if args.relations=='all' else [args.relations]

transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
transforms.RandomResizedCrop(224, scale=(0.8, 1.0)) ])


#dataset = ChestXray_Dataset(path=args.path, mode=mode, neib_samp=neib, relations=relations, k=k, transform=transform)  #ChestXray14
dataset = Chexpert_Dataset(path=args.path, mode=mode, neib_samp=neib, relations=relations, k=k, transform=transform)    #CheXpert
train_set, validation_set, test_set = dataset.tr_val_te_split(tr_pct=tr_pct)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn= mycollate, pin_memory=True, num_workers=20)
validation_loader = DataLoader(validation_set, batch_size=batch_size, collate_fn= mycollate, shuffle=True, pin_memory=True, num_workers=20)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, collate_fn= mycollate, pin_memory=True, num_workers=20)


if args.gpu>=0:
    torch.cuda.set_device(args.gpu)
folder = '/media'

model = SingleLayerImageGCN(relations, encoder=enc, out_dim=14, inchannel=inchannel).cuda()#ChestXray14 #CheXpert13

#criterion = W_BCEWithLogitsLoss()  #ChestXray14
criterion = W_BCELossWithNA()       #CheXpert
optimizer = optim.Adam(model.parameters(),  lr=1e-5, amsgrad =False, weight_decay=wd,)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=4)

def train(train_loader, validation_loader, test_loader, model, criterion, optimizer, iter_size=100):
    print('training.....')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start_time = time.time()    

    best_avgroc = 0.0    
    best_roc=[]
    best_epoch=0
    best_val=0.0
    not_renew=0              

    best_test=0.0

    # switch to train mode
    end = time.time()
    for epoch in range(args.epochs): 
        for i, data in enumerate(train_loader):
            # measure data loading time  
            model.train()

            inputs, targets, adj, k = data['image'].cuda(), data['label'].cuda(), data['adj'], data['k']
            adj_mats2 = {key: sparse_to_tensor(value).to_dense()[k:].cuda() for key, value in adj.items()}
            data_time.update(time.time() - end)
            output = model(inputs,  adj_mats2,  k=k)

            loss = criterion(output, targets[k:])
            losses.update(loss.item(), targets[k:].size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % iter_size == 0 or i == len(train_loader):
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses))
                
                loss_val, avgroc_val, _, = validate(validation_loader, model, criterion)
                loss_test, avgroc_test, roc_test = validate(test_loader, model, criterion)
                scheduler.step(avgroc_test)

                '''if avgroc_test>=best_test:
                    best_test=avgroc_test
                    torch.save({'epoch':epoch, 'i':i, 'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()},
                         '/media/models/KT-FSN/checkpoint_bestroc.pth.tar')'''
                      
                if avgroc_val>best_val:              
                    best_roc = roc_test
                    best_avgroc = avgroc_test
                    best_epoch = epoch+1
                    best_val=avgroc_val
                    not_renew=0
                else:
                    not_renew=not_renew+1
                print('best_avgroc',best_avgroc)
                print('best_roc',best_roc)
                print('best_epoch',best_epoch)
                print('now_epoch',epoch+1)
                print('not_renew',not_renew)
        
        end_time = time.time()    
        run_time = end_time - start_time    
        print('run_time',run_time)
        
    #return res    
    return losses.avg

def validate(val_loader, model, criterion):
    print('Validation......')
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        outputs=[]
        labels=[]
        for i, data in enumerate(val_loader):
            inputs, targets, adj, k = data['image'].cuda(), data['label'].cuda(), data['adj'], data['k']
            adj_mats2 = {key: sparse_to_tensor(value).to_dense()[k:].cuda() for key, value in adj.items()}

            output = model(inputs,  adj_mats2, k=k)
            loss = criterion(output, targets[k:])  
            losses.update(loss.item(),targets[k:].size(0))
            outputs.append(output)
            labels.append(targets[k:])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % 20 == 0:
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses))
        outputs =  torch.sigmoid(torch.cat(outputs)).cpu().numpy()   
        labels = torch.cat(labels).cpu().numpy()

    #roc = roc_auc_score(labels, outputs, average=None)     #ChestXray14     #ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
    roc = np.zeros(labels.shape[1])                         #CheXpert
    for i in  range(labels.shape[1]):                       #CheXpert
        lb = labels[:,i]                                    #CheXpert
        op = outputs[:,i]                                   #CheXpert
        roc[i] = roc_auc_score(lb[lb!=-1], op[lb!=-1])      #CheXpert

    avgroc = roc.mean()

    print('validate roc',roc)
    print('validate average roc',avgroc)

    return losses.avg, avgroc, roc

if use=='train':
    train(train_loader, validation_loader, test_loader, model, criterion, optimizer, 2000)

elif use=='test':
    cp = torch.load(join(folder,'models/KT-FSN/checkpoint_bestroc.pth.tar'))
    model.load_state_dict(cp['state_dict'])
    loss_test, avgroc_test, roc_test, pred = validate(test_loader, model, criterion)
    save_obj(pred,  join(folder,'results/KT-FSN/pred.pkl'))

    
    
    
    
    
    
    
    
    
    
    
    

