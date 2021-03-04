from __future__ import division

from object_detection.models import *
from object_detection.utils.dutils import *
from object_detection.utils.datasets import *

import os

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from os.path import isfile, join
import cv2
import time
import baselineUtils
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from transformer.batch import subsequent_mask
from torch.optim import Adam,SGD, RMSprop, Adagrad
from transformer.noam_opt import NoamOpt
import numpy as np
import scipy.io
import json
import pickle


Q = [[]]


def detect(image_folder = "object_detection/data/frames",model_def = "object_detection/config/yolov3-custom.cfg",weights_path = "object_detection/checkpoints/yolov3_ckpt_12.pth",class_path = "object_detection/data/custom/classes.names",conf_thres = 0.8,nms_thres = 0.4,batch_size=1,n_cpu=0,img_size=416,):
    #print("detecting objects from frames")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("object_detection/output", exist_ok=True)
    # Set up model
    model = Darknet(model_def, img_size=img_size).to(device)
    # Queue to hold locations
    
    if weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(weights_path,map_location=torch.device('cpu')))
    model.eval()  # Set in evaluation mode
    dataloader = DataLoader(
        ImageFolder(image_folder, img_size=img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
    )
    classes = load_classes(class_path)  # Extracts class labels from file
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        input_imgs = Variable(input_imgs.type(Tensor))
        # Get detections
        with torch.no_grad():
            print(input_imgs.shape)
            # print(input_imgs[0][1][0][0].numpy())
            # print(input_imgs[0][0][0][0].numpy())
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_thres, nms_thres)
        img = cv2.imread(img_paths[0])
        if detections[0] is not None:
            detections = rescale_boxes(detections[0], img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            #bbox_colors = random.sample(colors, n_cls_preds)
            x1, y1, x2, y2, conf, cls_conf, cls_pred = detections[0]
            Q[0].append([((x1+x2)/2).item()/img.shape[1],((y1+y2)/2).item()/img.shape[0]]) # todo need to replace this with dictionary when there are peds case
            print(len(Q[0]))
            if(len(Q[0])==8):
                Q_np = np.array(Q,dtype=np.float32)
                Q_d = Q_np[:,1:,0:2] - Q_np[:, :-1, 0:2]
                print(Q_d.shape)
                pr = trajectory(Q_d)
                pr = pr.cumsum(1)
                pr = pr+Q_np[:,-1:,0:2]
                #print(pr)
                Q[0].pop(0)

                co = (0,255,0) # green
                cp = (0,0,255) # red
                #print(gt[i])
                #print(pr[i])
                for j in range(12):
                    #op = (int(gt[i,j,0]*1920),int(gt[i,j,1]*1080))
                    pp = (int(pr[0,j,0]*1920),int(pr[0,j,1]*1080))
                    #print(pp)
                    #img = cv2.circle(img,gp,3,cg,-1)
                    img = cv2.circle(img,pp,3,cp,-1)
                #print("--------")
                for j in range(8):
                    op = (int(Q_np[0,j,0]*1920),int(Q_np[0,j,1]*1080))
                    #print(op)
                    img = cv2.circle(img,op,3,co,-1)
                filename = img_paths[0].split("\\")[-1].split(".")[0]
                cv2.imwrite("object_detection/output/"+str(filename)+'.png',img)
    print("detection completed")


def object_video(file_path,model_def = "object_detection/config/yolov3-custom.cfg",weights_path = "object_detection/checkpoints/yolov3_ckpt_12.pth",class_path = "object_detection/data/custom/classes.names",conf_thres = 0.8,nms_thres = 0.4,batch_size=1,n_cpu=0,img_size=416):
    files = [f for f in os.listdir("object_detection/data/frames") if isfile(join("object_detection/data/frames", f))]
    for i in range(len(files)):
        os.remove("object_detection/data/frames/"+files[i])
    files = [f for f in os.listdir('object_detection/output/') if isfile(join('object_detection/output/', f))]
    for i in range(len(files)):
        os.remove('object_detection/output/'+files[i])

    vidObj = cv2.VideoCapture(file_path)
    count = 0
    success = 1
    while success:
        success , image = vidObj.read()
        if success == 1:
            cv2.imwrite('object_detection/data/frames/'+str(count)+'.jpg', image)
            detect("object_detection/data/frames",model_def ,weights_path ,class_path ,conf_thres,nms_thres ,batch_size,n_cpu,img_size)
            files = [f for f in os.listdir('object_detection/output/') if isfile(join('object_detection/output/', f))]
            if(len(files)==1):
                im = cv2.imread('object_detection/output/'+files[0])
                cv2.imshow('output',im)
                k = cv2.waitKey(1)
                if k == 27:
                    cv2.destroyAllWindows()
                    break
            if os.path.exists('object_detection/data/frames/'+str(count)+'.jpg'):
                os.remove('object_detection/data/frames/'+str(count)+'.jpg')
            if os.path.exists('object_detection/output/'+str(count)+'.png'):
                os.remove('object_detection/output/'+str(count)+'.png')
        count+=1
    print("Frames generated")

def trajectory(inp):
    device=torch.device("cpu")
    import individual_TF
    model=individual_TF.IndividualTF(2, 3, 3, N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1,mean=[0,0],std=[0,0]).to(device)

    model.load_state_dict(torch.load(f'models/Individual/my_data_train/00099.pth'))
    model.eval()
    gt = []
    pr = []
    inp_ = []
    peds = []
    frames = []
    dt = []
    inp = np.array(inp,dtype=np.float32)
    inp = torch.from_numpy(inp)
    inp = inp.to(device)
    src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
    start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(
                    device)
    dec_inp=start_of_seq

    for i in range(12):
        trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)
        out = model(inp, dec_inp, src_att, trg_att)
        dec_inp=torch.cat((dec_inp,out[:,-1:,:]),1)

    preds_tr_b=(dec_inp[:,1:,0:2]).cpu().detach().numpy() #.cumsum(1)#+batch['src'][:,-1:,0:2].cpu().detach().numpy()
    pr.append(preds_tr_b)
    pr = np.concatenate(pr, 0)
    #print(pr)
    return pr


if __name__ == '__main__':
    model_def = "object_detection/config/yolov3-custom.cfg"
    weights_path = "object_detection/checkpoints/yolov3_ckpt_48.pth"
    class_path = "object_detection/data/custom/classes.names"
    conf_thres = 0.8
    nms_thres = 0.4
    batch_size=1
    n_cpu=0
    img_size=416
    object_video("C:\\Users\\venny\\Downloads\\MoreTvideos\\5_1.mp4",model_def,weights_path,class_path,conf_thres,nms_thres,batch_size,n_cpu,img_size)