import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results
from Trajectory import individual_TF
from Trajectory.transformer.batch import subsequent_mask
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.traj_ped = individual_TF.IndividualTF(2, 3, 3, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, mean=[0,0], std=[0,0]).to(device)
        self.traj_ped.load_state_dict(torch.load(f'Trajectory/models/Individual/eth_train/00013.pth', map_location=torch.device('cpu')))
        self.traj_ped.eval()
        self.traj_endeffector = individual_TF.IndividualTF(2, 3, 3, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, mean=[0,0], std=[0,0]).to(device)
        self.traj_endeffector.load_state_dict(torch.load(f'Trajectory/models/Individual/traj_endeffector.pth', map_location=torch.device('cpu')))
        self.traj_endeffector.eval()
        self.traj_arm = individual_TF.IndividualTF(2, 3, 3, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, mean=[0, 0], std=[0, 0]).to(device)
        self.traj_arm.load_state_dict(torch.load(f'Trajectory/models/Individual/traj_arm.pth', map_location=torch.device('cpu')))
        self.traj_arm.eval()
        self.traj_probe = individual_TF.IndividualTF(2, 3, 3, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, mean=[0, 0], std=[0, 0]).to(device)
        self.traj_probe.load_state_dict(torch.load(f'Trajectory/models/Individual/traj_probe.pth', map_location=torch.device('cpu')))
        self.traj_probe.eval()
        self.class_names = self.detector.class_names
        self.Q = { }

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        mean_end_effector = torch.tensor((-2.6612e-05, -7.8652e-05))
        std_end_effector = torch.tensor((0.0025, 0.0042))
        mean_arm = torch.tensor([-1.3265e-05, -6.5026e-06])
        std_arm = torch.tensor([0.0030, 0.0185])
        mean_probe = torch.tensor([-5.1165e-05, -7.1806e-05])
        std_probe = torch.tensor([0.0038, 0.0185])
        mean_ped = torch.tensor([0.0001, 0.0001])
        std_ped = torch.tensor([0.0001, 0.0001])
        while self.vdo.grab() :
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            height, width = ori_im.shape[:2]
            bbox_xywh , cls_conf, cls_ids = self.detector(im)
            print("cls_ids")
            print(cls_conf)
            print(cls_ids)
            for i in range(3):
                mask = cls_ids == i
                t_cls_conf = cls_conf[mask]
                t_bbox_xywh = bbox_xywh[mask]
                if t_cls_conf.size > 0:
                    pt = [t_bbox_xywh[np.argmax(t_cls_conf)][0] / width, t_bbox_xywh[np.argmax(t_cls_conf)][1] / height]
                    t_id = i
                    if t_id in self.Q:
                        self.Q[t_id][0].append(pt)
                    else:
                        self.Q[t_id] = [[pt]]
            # select person class
            mask = cls_ids == 3
            bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            #bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]

            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
            for i in range(len(outputs)):
                t_id = outputs[i][4]+5 # added with 5 so that ped id will not clash with id's of end_effector arm and probe
                pt = [(int(outputs[i][0]) + int(outputs[i][2])) / (2*width), (int(outputs[i][1]) + int(outputs[i][3])) / (2*height)]
                #print(pt)
                if t_id in self.Q:
                    self.Q[t_id][0].append(pt)
                else:
                    self.Q[t_id] = [[pt]]
            # print(self.Q)
            for i in self.Q:
                if (len(self.Q[i][0])) == 8:
                    Q_np = np.array(self.Q[i], dtype=np.float32)
                    Q_d = Q_np[:, 1:, 0:2] - Q_np[:, :-1, 0:2]
                    pr = []
                    inp = torch.from_numpy(Q_d)
                    #print(i)
                    #print(inp)
                    if i == 0:
                        inp = (inp.to(device) - mean_end_effector.to(device)) / std_end_effector.to(device)
                    elif i == 1:
                        inp = (inp.to(device) - mean_arm.to(device)) / std_arm.to(device)
                    elif i == 2:
                        inp = (inp.to(device) - mean_probe.to(device)) / std_probe.to(device)
                    else:
                        inp = (inp.to(device) - mean_ped.to(device)) / std_ped.to(device)
                    src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
                    start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(
                        device)
                    dec_inp = start_of_seq
                    print("predicting trajectory")
                    for itr in range(12):
                        trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)
                        if i == 0:
                            out = self.traj_endeffector(inp, dec_inp, src_att, trg_att)
                        elif i == 1:
                            out = self.traj_arm(inp, dec_inp, src_att, trg_att)
                        elif i == 2:
                            out = self.traj_probe(inp, dec_inp, src_att, trg_att)
                        else:
                            out = self.traj_ped(inp, dec_inp, src_att, trg_att)
                        dec_inp = torch.cat((dec_inp, out[:, -1:, :]), 1)
                    if i == 0:
                        preds_tr_b = (dec_inp[:, 1:, 0:2] * std_end_effector.to(device) + mean_end_effector.to(device)).detach().cpu().numpy().cumsum(1)+Q_np[:, -1:, 0:2]
                    elif i == 1:
                        preds_tr_b = (dec_inp[:, 1:, 0:2] * std_arm.to(device) + mean_arm.to(device)).detach().cpu().numpy().cumsum(1) + Q_np[:, -1:, 0:2]
                    elif i == 2:
                        preds_tr_b = (dec_inp[:, 1:, 0:2] * std_probe.to(device) + mean_probe.to(device)).detach().cpu().numpy().cumsum(1) + Q_np[:, -1:, 0:2]
                    else:
                        preds_tr_b = (dec_inp[:, 1:, 0:2] * std_ped.to(device) + mean_ped.to(device)).detach().cpu().numpy().cumsum(1) + Q_np[:, -1:, 0:2]
                    pr.append(preds_tr_b)
                    pr = np.concatenate(pr, 0)
                    self.Q[i][0].pop(0)
                    co = (0, 255, 0)  # green
                    cp = (0, 0, 255)  # red
                    #print(pr)
                    for j in range(11):
                        pp1 = (int(pr[0, j, 0]*width), int(pr[0, j, 1]*height))
                        pp2 = (int(pr[0, j+1, 0] * width), int(pr[0, j+1, 1] * height))
                        #ori_im = cv2.circle(ori_im, pp, 3, cp, -1)
                        ori_im = cv2.line(ori_im,pp1,pp2,cp,2)
                    for j in range(7):
                        op1 = (int(Q_np[0, j, 0]*width), int(Q_np[0, j, 1]*height))
                        op2 = (int(Q_np[0, j+1, 0] * width), int(Q_np[0, j+1, 1] * height))
                        #ori_im = cv2.circle(ori_im, op, 3, co, -1)
                        ori_im = cv2.line(ori_im, op1, op2, co, 2)
            cv2.imshow("test", ori_im)
            cv2.waitKey(1)
            # draw boxes for visualization
            # if len(outputs) > 0:
            #     bbox_tlwh = []
            #     bbox_xyxy = outputs[:, :4]
            #     identities = outputs[:, -1]
            #     ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
            #
            #     for bb_xyxy in bbox_xyxy:
            #         bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
            #
            #     results.append((idx_frame - 1, bbox_tlwh, identities))

            end = time.time()

            # if self.args.display:
            #     cv2.imshow("test", ori_im)
            #     cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            write_results(self.save_results_path, results, 'mot')


            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH", type=str, default="C:/Users/venny/Downloads/new_video_3.mp4")
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_false")
    parser.add_argument("--frame_interval", type=int, default=10)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()
