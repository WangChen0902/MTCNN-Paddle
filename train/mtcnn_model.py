import paddle
import paddle.nn as nn
import numpy as np


class PNet(nn.Layer):
    def __init__(self):
        super(PNet, self).__init__()
        self.conv0 = nn.Conv2D(in_channels=3, out_channels=10, kernel_size=[3, 3])
        self.prelu0 = nn.PReLU(num_parameters=10)
        self.pool0 = nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv1 = nn.Conv2D(in_channels=10, out_channels=16, kernel_size=[3, 3])
        self.prelu1 = nn.PReLU(num_parameters=16)
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=[3, 3])
        self.prelu2 = nn.PReLU(num_parameters=32)
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=2, kernel_size=[1, 1])
        self.conv4 = nn.Conv2D(in_channels=32, out_channels=4, kernel_size=[1, 1])
        self.softmax0 = nn.Softmax(axis=1)

    def forward(self, data):
        conv1 = self.conv0(data)
        prelu1 = self.prelu0(conv1)
        pool1 = self.pool0(prelu1)
        conv2 = self.conv1(pool1)
        prelu2 = self.prelu1(conv2)
        conv3 = self.conv2(prelu2)
        prelu3 = self.prelu2(conv3)
        conv4_1 = self.conv3(prelu3)
        conv4_2 = self.conv4(prelu3)
        prob1 = self.softmax0(conv4_1)
        return prob1, conv4_2, None


class RNet(nn.Layer):
    def __init__(self):
        super(RNet, self).__init__()
        self.conv0 = nn.Conv2D(in_channels=3, out_channels=28, kernel_size=[3, 3])
        self.prelu0 = nn.PReLU(num_parameters=28)
        self.pool0 = nn.MaxPool2D(kernel_size=[3, 3], stride=2, ceil_mode=True)
        self.conv1 = nn.Conv2D(in_channels=28, out_channels=48, kernel_size=[3, 3])
        self.prelu1 = nn.PReLU(num_parameters=48)
        self.pool1 = nn.MaxPool2D(kernel_size=[3, 3], stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2D(in_channels=48, out_channels=64, kernel_size=[2, 2])
        self.prelu2 = nn.PReLU(num_parameters=64)
        self.linear0 = nn.Linear(in_features=576, out_features=128)
        self.prelu3 = nn.PReLU(num_parameters=128)
        self.linear1 = nn.Linear(in_features=128, out_features=2)
        self.linear2 = nn.Linear(in_features=128, out_features=4)
        self.softmax0 = nn.Softmax(axis=1)

    def forward(self, data):
        conv1 = self.conv0(data)
        prelu1 = self.prelu0(conv1)
        pool1 = self.pool0(prelu1)
        conv2 = self.conv1(pool1)
        prelu2 = self.prelu1(conv2)
        pool2 = self.pool1(prelu2)
        conv3 = self.conv2(pool2)
        prelu3 = self.prelu2(conv3)
        conv4 = paddle.reshape(x=prelu3, shape=[-1, 576])
        conv4 = self.linear0(conv4)
        prelu4 = self.prelu3(conv4)
        conv5_1 = self.linear1(prelu4)
        conv5_2 = self.linear2(prelu4)
        prob1 = self.softmax0(conv5_1)
        return prob1, conv5_2, None


class ONet(nn.Layer):
    def __init__(self):
        super(ONet, self).__init__()
        self.conv0 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=[3, 3])
        self.prelu0 = nn.PReLU(num_parameters=32)
        self.pool0 = nn.MaxPool2D(kernel_size=[3, 3], stride=2, ceil_mode=True)
        self.conv1 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=[3, 3])
        self.prelu1 = nn.PReLU(num_parameters=64)
        self.pool1 = nn.MaxPool2D(kernel_size=[3, 3], stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=[3, 3])
        self.prelu2 = nn.PReLU(num_parameters=64)
        self.pool2 = nn.MaxPool2D(kernel_size=[2, 2], stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=[2, 2])
        self.prelu3 = nn.PReLU(num_parameters=128)
        self.linear0 = nn.Linear(in_features=1152, out_features=256)
        self.prelu4 = nn.PReLU(num_parameters=256)
        self.linear1 = nn.Linear(in_features=256, out_features=2)
        self.linear2 = nn.Linear(in_features=256, out_features=4)
        self.linear3 = nn.Linear(in_features=256, out_features=10)
        self.softmax0 = nn.Softmax(axis=1)

    def forward(self, data):
        conv1 = self.conv0(data)
        prelu1 = self.prelu0(conv1)
        pool1 = self.pool0(prelu1)
        conv2 = self.conv1(pool1)
        prelu2 = self.prelu1(conv2)
        pool2 = self.pool1(prelu2)
        conv3 = self.conv2(pool2)
        prelu3 = self.prelu2(conv3)
        pool3 = self.pool2(prelu3)
        conv4 = self.conv3(pool3)
        prelu4 = self.prelu3(conv4)
        conv5 = paddle.reshape(x=prelu4, shape=[-1, 1152])
        conv5 = self.linear0(conv5)
        prelu5 = self.prelu4(conv5)
        conv6_1 = self.linear1(prelu5)
        conv6_2 = self.linear2(prelu5)
        conv6_3 = self.linear3(prelu5)
        prob1 = self.softmax0(conv6_1)
        return prob1, conv6_2, conv6_3