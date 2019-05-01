from __future__ import division

import os
import torch as t
from src.config import opt
from src.head_detector_vgg16 import Head_Detector_VGG16
from trainer import Head_Detector_Trainer
from PIL import Image
import numpy as np
from data.dataset import preprocess
import matplotlib.pyplot as plt 
import src.array_tool as at
from src.vis_tool import visdom_bbox
import argparse
import src.utils as utils
from src.config import opt
import time

SAVE_FLAG = 0

THRESH = 0.012 #0.01
IM_RESIZE = False

def read_img(path):
    f = Image.open(path)
    if IM_RESIZE:
        f = f.resize((640,480), Image.ANTIALIAS)

    f.convert('RGB')
    img_raw = np.asarray(f, dtype=np.uint8)
    img_raw_final = img_raw.copy()
    img = np.asarray(f, dtype=np.float32)
    img = img.transpose((2,0,1))
    _, H, W = img.shape
    img = preprocess(img)
    _, o_H, o_W = img.shape

    # scale
    scale = o_H / H
    return img, img_raw_final, scale

def detect2(img, img_raw, scale, model_path,file_file):
    head_detector = Head_Detector_VGG16(ratios=[1], anchor_scales=[2,4])
    trainer = Head_Detector_Trainer(head_detector).cuda()

    # load model
    trainer.load(model_path)
    img = at.totensor(img)
    img = img[None, : ,: ,:]
    img = img.cuda().float()

    # predict model
    pred_bboxes_, _ = head_detector.predict(img, scale, mode='evaluate', thresh=THRESH)

    for i in range(pred_bboxes_.shape[0]):
        ymin, xmin, ymax, xmax = pred_bboxes_[i,:]
        file_file.write(str(int(xmin/scale)) + ',' + str(int(ymin/scale)) + ',' + str(int(xmax/scale)) + ',' + str(int(ymax/scale)) + ',' + '0' + ' ')



