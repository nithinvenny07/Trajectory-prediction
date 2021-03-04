import argparse
import baselineUtils
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from transformer.batch import subsequent_mask
from torch.optim import Adam,SGD, RMSprop, Adagrad
from transformer.noam_opt import NoamOpt
import numpy as np
import scipy.io
import json
import pickle
import cv2


from torch.utils.tensorboard import SummaryWriter




def main():
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--dataset_folder',type=str,default='datasets')
    parser.add_argument('--dataset_name',type=str,default='zara1')
    parser.add_argument('--obs',type=int,default=8)
    parser.add_argument('--preds',type=int,default=12)
    parser.add_argument('--emb_size',type=int,default=512)
    parser.add_argument('--heads',type=int, default=8)
    parser.add_argument('--layers',type=int,default=6)
    parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--val_size',type=int, default=0)
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--max_epoch',type=int, default=1500)
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--validation_epoch_start', type=int, default=30)
    parser.add_argument('--resume_train',action='store_true')
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--name', type=str, default="zara1")
    parser.add_argument('--factor', type=float, default=1.)
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--model_pth', type=str)




    args=parser.parse_args()
    model_name=args.name

    #device=torch.device("cuda")

    #if args.cpu or not torch.cuda.is_available():
    device=torch.device("cpu")

    args.verbose=True

    test_dataset,_ =  baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False,eval=True,verbose=args.verbose)


    import individual_TF
    model=individual_TF.IndividualTF(2, 3, 3, N=args.layers,
                   d_model=args.emb_size, d_ff=2048, h=args.heads, dropout=args.dropout,mean=[0,0],std=[0,0]).to(device)

    
    model.load_state_dict(torch.load(f'models/Individual/my_data_train/00013.pth'))
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model.eval()
    gt = []
    pr = []
    inp_ = []
    peds = []
    frames = []
    dt = []
                
    for id_b,batch in enumerate(test_dl):
        #print(batch['src'].shape)
        #inp_.append(batch['src'])
        gt.append(batch['trg'][:,:,0:2])
        #frames.append(batch['frames'])
        #peds.append(batch['peds'])
        #dt.append(batch['dataset'])

        inp = batch['src'][:, 1:, 2:4].to(device) #- mean.to(device)) / std.to(device)
        #print(inp.shape)
        src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
        start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(
                        device)
        # print("start of seq")
        # print(start_of_seq[0])
        dec_inp=start_of_seq

        for i in range(args.preds):
            trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)
            # print("src_att shape")
            # print(src_att.shape)
            # print("trg_att shape")
            # print(trg_att.shape)
            out = model(inp, dec_inp, src_att, trg_att)
            # print("out shape")
            # print(out.shape)
            # print("-----------")
            dec_inp=torch.cat((dec_inp,out[:,-1:,:]),1)


        print("batch['src']")
        print(batch['src'].shape)
        preds_tr_b=(dec_inp[:,1:,0:2]).cpu().detach().numpy().cumsum(1)+batch['src'][:,-1:,0:2].cpu().detach().numpy()
        #print(preds_tr_b[1])
        pr.append(preds_tr_b)
        # print("test epoch %03i/%03i  batch %04i / %04i" % (
        #         epoch, args.max_epoch, id_b, len(test_dl)))
    gt = np.concatenate(gt, 0)
    #dt_names = test_dataset.data['dataset_name']
    pr = np.concatenate(pr, 0)
    mad, fad, errs = baselineUtils.distance_metrics(gt, pr)
    #print(frames)
    #print(dt.shape)
    #print(dt)
    #print(gt[1])
    #print(pr[1])
    #print("done!!!")
    #print("mad %f fad %f"%(mad,fad))
    # for i in range(pr.shape[0]):
    #     pathin = 'c_1 frames/c_1_'
    #     pathout = '5_1 frames_out/'
    #     img = cv2.imread(pathin+str(frames[i][8])+'.jpg')
    #     cg = (0,255,0) # green
    #     cp = (0,0,255) # red
    #     #print(gt[i])
    #     #print(pr[i])
    #     for j in range(12):
    #         gp = (int(gt[i,j,0]*1920),int(gt[i,j,1]*1080))
    #         pp = (int(pr[i,j,0]*1920),int(pr[i,j,1]*1080))
    #         img = cv2.circle(img,gp,3,cg,-1)
    #         img = cv2.circle(img,pp,3,cp,-1)
    #         #print(gp)
    #         #print(pp)
    #         #print(frames[i][8])
    #     cv2.imwrite(pathout+str(frames[i][8])+'.jpg',img)

if __name__=='__main__':
    main()


    