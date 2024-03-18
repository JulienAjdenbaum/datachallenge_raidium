#coding: utf-8
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import torchvision.models as models
import os
import argparse
import random
import sys
import matplotlib.pyplot as plt

import utils as ut
from mydataset import CVCClinicDB_Dataset
from unet import U_Net
from TransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from TransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from FCBFormer import models
from loss import tvMF_DiceLoss, Adaptive_tvMF_DiceLoss
from dsc import DiceScoreCoefficient
from measure import rand_index_dataset
from skimage.measure import label
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# training #
def train(epoch, iters, kappa):
    model.train()
    sum_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        inputs = inputs.cuda(device)
        targets = targets.cuda(device)
        targets = targets.long()
         
        output = model(inputs)

        if args.loss == 'Atvmf':
            loss = criterion(output, targets, kappa)
        else:
            loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        if iters+1%10==0:
            print("iter %d / %d  train_Loss: %.4f" % (iters+1, args.maxiter, loss))
        with open(PATH_1, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (iters+1, loss))

        iters += 1

        if iters == args.maxiter:
            sys.exit()

        adjust_learning_rate(optimizer, iters)
        
    return sum_loss/(batch_idx+1), iters
    
# validation #
def test(epoch,kappa):
    model.eval()
    predict = []
    answer = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.cuda(device)
            targets = targets.cuda(device)
            targets = targets.long()
                
            output = model(inputs)

            output = F.softmax(output, dim=1)
            output = output.cpu().numpy()
            targets = targets.cpu().numpy()
 
            for j in range(args.batchsize):
                predict.append(output[j])
                answer.append(targets[j])

    dsc = DiceScoreCoefficient(n_classes=args.classes)(predict, answer)
    
    predict = np.argmax(predict, axis=1)
    for i in range(len(predict)):
        predict[i] = label(predict[i])

    for i in range(len(answer)):
        answer[i] = label(answer[i])
        
    # plt.imshow(predict[i-1], cmap="tab20")
    # plt.savefig(str(np.random.randint(1, 1000)))
    # plt.imshow(answer[i-1], cmap="tab20")
    # plt.savefig(str(np.random.randint(1, 1000)))
    measure = rand_index_dataset(np.array(answer).reshape(len(answer), answer[0].shape[0]*answer[0].shape[1]), 
                              np.array(predict).reshape(len(predict), predict[0].shape[0]*predict[0].shape[1]))
    
    return dsc, measure

# adjust learning rate #
def adjust_learning_rate(optimizer, iters):
    lr = 0.01*(1 - iters/args.maxiter)**0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# adjust the parameter kappa #
def adjust_kappa(mm):
    return torch.Tensor(mm*args.lamda).cuda(device)

###### main ######
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tvMF Dice loss')
    parser.add_argument('--classes', '-c', type=int, default=2)
    parser.add_argument('--batchsize', '-b', type=int, default=24)
    parser.add_argument('--num_epochs', '-e', type=int, default=200)
    parser.add_argument('--maxiter', '-m', type=int, default=4000)
    parser.add_argument('--kappa', '-k', type=float, default=32.0)
    parser.add_argument('--lamda', '-lm', type=float, default=32.0)
    parser.add_argument('--path', '-i', default='./data')
    parser.add_argument('--out', '-o', type=str, default='result')
    parser.add_argument('--model', '-mo', type=str, default='unet')
    parser.add_argument('--loss', '-lo', type=str, default='tvmf')
    parser.add_argument('--gpu', '-g', type=str, default=-1)
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()
    gpu_flag = args.gpu

    # device #
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # save #
    if not os.path.exists("{}_{}".format(args.out, args.model)):
      	os.mkdir("{}_{}".format(args.out, args.model))
    if not os.path.exists(os.path.join("{}_{}".format(args.out, args.model), "model")):
      	os.mkdir(os.path.join("{}_{}".format(args.out, args.model), "model"))

    PATH_1 = "{}_{}/trainloss.txt".format(args.out, args.model)
    PATH_2 = "{}_{}/testloss.txt".format(args.out, args.model)
    PATH_3 = "{}_{}/DSC.txt".format(args.out, args.model)

    with open(PATH_1, mode = 'w') as f:
        pass
    with open(PATH_2, mode = 'w') as f:
        pass
    with open(PATH_3, mode = 'w') as f:
        pass

    # seed #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # preprocceing #
    train_transform = ut.ExtCompose([ut.ExtResize((224,224)),
                                     ut.ExtRandomRotation(degrees=90),
                                     ut.ExtRandomHorizontalFlip(),
                                     ut.ExtToTensor(),
                                     ])

    val_transform = ut.ExtCompose([ut.ExtResize((224,224)),
                                   ut.ExtToTensor(),
                                   ])

    # data loader #
    data_train = CVCClinicDB_Dataset(root = args.path, 
                                     dataset_type='train', 
                                     transform=train_transform) 
    data_val = CVCClinicDB_Dataset(root = args.path, 
                                   dataset_type='val', 
                                   transform=val_transform) 
    
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=args.batchsize, shuffle=True, drop_last=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=args.batchsize, shuffle=False, drop_last=True, num_workers=2)


    # model #
    if args.model=='unet':
        model = U_Net(output=args.classes).cuda(device)

    elif args.model=='trans_unet':
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.n_classes = args.classes
        config_vit.n_skip = 3
        if "R50-ViT-B_16".find('R50') != -1:
            config_vit.patches.grid = (int(224 / 16), int(224 / 16))
        model = ViT_seg(config_vit, img_size=224, num_classes=config_vit.n_classes).cuda(device)
        # model.load_from(weights=np.load(config_vit.pretrained_path))

    elif args.model=='fcb':
        model = models.FCBFormer().cuda(device)


    # loss function #
    if args.loss == 'tvmf':
        criterion = tvMF_DiceLoss(n_classes=args.classes, kappa=args.kappa)
    elif args.loss == 'Atvmf':
        criterion = Adaptive_tvMF_DiceLoss(n_classes=args.classes)


    # optimizer #
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)


    ### training & validation ###
    sample = 0
    sample_loss = 10000000
    iters = 0
    kappa = torch.Tensor(np.zeros((args.classes))).cuda(device)

    for epoch in range(args.num_epochs):
        loss_train, iters = train(epoch, iters, kappa)

        dsc, measure = test(epoch, kappa)
        
        if args.loss == 'Atvmf':
            kappa = adjust_kappa(dsc)

        with open(PATH_3, mode = 'a') as f:
            f.write("\t%d\t%f\n" % (epoch+1, np.mean(dsc)))


        PATH_train ="{}_{}/model/model_train.pth".format(args.out, args.model)
        print(PATH_train)
        torch.save(model.state_dict(), PATH_train)

        if measure >= sample:
            sample = measure
            PATH_dsc ="{}_{}/model/model_bestdsc.pth".format(args.out, args.model)
            torch.save(model.state_dict(), PATH_dsc)

        if loss_train < sample_loss:
           sample_loss = loss_train
           PATH_loss ="{}_{}/model/model_bestloss.pth".format(args.out, args.model)
           torch.save(model.state_dict(), PATH_loss)
            
        print("")
        print(args.model)
        print("Average DSC : %.4f" % np.mean(dsc))
        print("Average measure : %.4f" % measure)
        print("")
        
    if os.path.exists("results.csv"):
        df = pd.read_csv("results.csv")
    else:
        df = pd.DataFrame(columns=["out", "kappa", "lambda", "model", "loss", "DSC", "measure"])
    
    ligne = pd.DataFrame([[args.out, args.kappa, args.model, args.loss, np.mean(dsc), measure]], 
                         columns=["out", "kappa", "model", "loss", "DSC", "measure"])
    df = pd.concat([df, ligne])
    df.to_csv("results.csv", index=False)


