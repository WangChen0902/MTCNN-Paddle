import paddle
from paddle.io import Dataset
import os
from tqdm import tqdm
import cv2
import numpy as np


class MtcnnLoader(Dataset):
    def __init__(self, txt_path):
        self.img_paths = []
        self.labels = []
        self.bboxes = []
        self.landmarks = []
        imagelist = open(txt_path, 'r')
        for line in tqdm(imagelist.readlines()):
            info=line.strip().split(' ')
            self.img_paths.append(info[0])
            self.labels.append(int(info[1]))
            bbox = [0]*4
            landmark = [0]*10
            if len(info) == 6:
                bbox = [float(info[2]),float(info[3]),float(info[4]),float(info[5])]
            if len(info) == 12:
                landmark = [float(info[2]),float(info[3]),float(info[4]),float(info[5]),float(info[6]),
                            float(info[7]),float(info[8]),float(info[9]),float(info[10]),float(info[11])]
            self.bboxes.append(bbox)
            self.landmarks.append(landmark)
    
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = cv2.imread(self.img_paths[index])
        img_t = (img-127.5)/128.0
        label = np.array(self.labels[index])
        bbox = np.array(self.bboxes[index])
        landmark = np.array(self.landmarks[index])
        img_t = paddle.to_tensor(img_t).astype('float32')
        return img_t, label, bbox, landmark
