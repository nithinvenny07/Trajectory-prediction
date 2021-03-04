from models import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *

import torch
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torch.autograd import Variable


# import os
# import time
import cv2
import numpy as np


class yolo_v3(object):
    def __init__(self, class_path, conf_thres,nms_thres,cfg ="config/yolov3-custom.cfg", weights_path = "weights/yolov3_ckpt_48.pth", img_size = 416):
        self.cfg = cfg
        self.weights_path = weights_path
        self.img_size = img_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Darknet(cfg, img_size=img_size).to(device)
        self.net.load_state_dict(torch.load(self.weights_path,map_location=torch.device('cpu')))
        print("yolo weights loaded successfully")
        self.net.eval()
        self.size = self.img_size, self.img_size
        self.classes = load_classes(class_path)
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

    def __call__(self, ori_img):
        img = ori_img.astype(np.float) / 255.
        img = cv2.resize(img, self.size)
        img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            img = img.to(device)
            out_boxes = self.net(img)
            #print(out_boxes)
            detections = non_max_suppression(out_boxes, self.conf_thres, self.nms_thres)
            return detections


if __name__ == '__main__':
    detector = yolo_v3("config/classes.names", 0.5, 0.4)
    # img = cv2.imread("C:/Users/venny/Downloads/Object Detection/Testing Videos/kuka frames/1278.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # out = detector(img)
    # out = out[0].numpy()
    # print(out)
    # img = cv2.resize(img, (416,416))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.rectangle(img, (int(out[0][0]),int(out[0][1])), (int(out[0][2]),int(out[0][3])), (255, 0, 0), 2)
    # cv2.imshow("eind",img)
    # cv2.waitKey(0)
    file_path = "C:/Users/venny/Downloads/Object Detection/Testing Videos/kuka_Trim.mp4"
    vidObj = cv2.VideoCapture(file_path)
    success = 1
    while success:
        success, ori_img =vidObj.read()
        if success == 1:
            img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (416, 416))
            out = detector(img)
            print(out)
            if out[0] is not None:
                out = out[0].numpy()
                for i in range(out.shape[0]):
                    img = cv2.rectangle(img, (int(out[i][0]), int(out[i][1])), (int(out[i][2]), int(out[i][3])),
                                        (255, 0, 0), 2)
                    cv2.imshow("eind", img)
                    cv2.waitKey(1)





















