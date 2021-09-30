
# coding: utf-8

import paddle
import paddle.nn as nn
import numpy as np
#只把70%数据用作参数更新
num_keep_radio=0.7



class PNet(nn.Layer):
    def __init__(self, phase = 'train'):
        super().__init__()

        self.conv1 = nn.Conv2D(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2D(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2D(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2D(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2D(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(axis=1)
        self.conv4_2 = nn.Conv2D(32, 4, kernel_size=1)
        self.conv4_3 = nn.Conv2D(32, 10, kernel_size=1)
        self.phase = phase

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        cls_pred = self.conv4_1(x)
        cls_pred = self.softmax4_1(cls_pred)
        bbox_pred = self.conv4_2(x)        
        landmark_pred = self.conv4_3(x)
        if self.phase == 'train':
            return (cls_pred, bbox_pred, landmark_pred)
        else:
            cls_pred_test = paddle.squeeze(cls_pred, axis=0)
            bbox_pred_test = paddle.squeeze(bbox_pred, axis=0)
            landmark_pred_test = paddle.squeeze(landmark_pred, axis=0)
            return (cls_pred_test,bbox_pred_test,landmark_pred_test)



class RNet(nn.Layer):
    def __init__(self, phase = 'train'):
        super().__init__()

        self.conv1 = nn.Conv2D(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2D(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2D(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2D(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2D(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.flatten4 = nn.Flatten(1, 3)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(axis=1)
        self.dense5_2 = nn.Linear(128, 4)
        self.dense5_3 = nn.Linear(128, 10)
        self.phase = phase
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.flatten4(x)
        x = self.dense4(x)
        x = self.prelu4(x)
        cls_pred = self.dense5_1(x)
        cls_pred = self.softmax5_1(cls_pred)
        bbox_pred = self.dense5_2(x)
        landmark_pred = self.dense5_3(x)
        return (cls_pred, bbox_pred, landmark_pred)



class ONet(nn.Layer):
    def __init__(self, phase = 'train'):
        super().__init__()

        self.conv1 = nn.Conv2D(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2D(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2D(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2D(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2D(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2D(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2D(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.flatten5 = nn.Flatten(1, 3)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(axis=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)
        self.phase = phase

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = self.flatten5(x)
        x = self.dense5(x)
        x = self.prelu5(x)
        cls_pred = self.dense6_1(x)
        cls_pred = self.softmax6_1(cls_pred)
        bbox_pred = self.dense6_2(x)
        landmark_pred = self.dense6_3(x)
        return (cls_pred, bbox_pred, landmark_pred)

