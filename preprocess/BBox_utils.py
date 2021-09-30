
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np


# In[2]:


def getDataFromTxt(txt,with_landmark=True):
    '''获取txt中的图像路径，人脸box，人脸关键点
    参数：
      txt：数据txt文件
      data_path:数据存储目录
      with_landmark:是否留有关键点
    返回值：
      result包含(图像路径，人脸box，关键点)
    '''
    with open(txt,'r') as f:
        lines=f.readlines()
    result=[]
    imgs_path = []
    words = []
    isFirst = True
    labels = []
    for line in lines:
        line = line.rstrip()
        if line.startswith('#'):
            if isFirst is True:
                isFirst = False
            else:
                labels_copy = labels.copy()
                words.append(labels_copy)
                labels.clear()
            path = line[2:]
            path = txt.replace('label.txt','images/') + path
            imgs_path.append(path)
        else:
            line = line.split(' ')
            label = [float(x) for x in line]
            labels.append(label)
    words.append(labels)

    for index in range(len(imgs_path)):
        img_path = imgs_path[index]
        labels = words[index]
        bboxes = []
        landmarks = []
        # annotations = np.zeros((0, 15))
        # if len(labels) == 0:
        #     return annotations
        for idx, label in enumerate(labels):
            # bbox
            box=(label[0], label[1], label[0]+label[2], label[1]+label[3])
            box=[float(_) for _ in box]
            box=list(map(int,box))
            # landmarks
            bboxes.append(BBox(box))
            if with_landmark:
                landmark=np.zeros((5, 2))
                for i in range(5):
                    rv = (float(label[4+3*i]), float(label[4+3*i+1]))
                    landmark[i]=rv
                landmarks.append(landmark)
        if not with_landmark:
            result.append((img_path, bboxes))
        else:
            result.append((img_path, bboxes, landmarks))
    return result


# In[4]:


class BBox:
    #人脸的box
    def __init__(self,box):
        self.left=box[0]
        self.top=box[1]
        self.right=box[2]
        self.bottom=box[3]
        
        self.x=box[0]
        self.y=box[1]
        self.w=box[2]-box[0]
        self.h=box[3]-box[1]
    
    def project(self,point):
        '''将关键点的绝对值转换为相对于左上角坐标偏移并归一化
        参数：
          point：某一关键点坐标(x,y)
        返回值：
          处理后偏移
        '''
        x=(point[0]-self.x)/self.w
        y=(point[1]-self.y)/self.h
        return np.asarray([x,y])
    def reproject(self,point):
        '''将关键点的相对值转换为绝对值，与project相反
        参数：
          point:某一关键点的相对归一化坐标
        返回值：
          处理后的绝对坐标
        '''
        x=self.x+self.w*point[0]
        y=self.y+self.h*point[1]
        return np.asarray([x,y])
    def reprojectLandmark(self,landmark):
        '''对所有关键点进行reproject操作'''
        p=np.zeros((len(landmark),2))
        for i in range(len(landmark)):
            p[i]=self.reproject(landmark[i])
        return p
    def projectLandmark(self,landmark):
        '''对所有关键点进行project操作'''
        p=np.zeros((len(landmark),2))
        for i in range(len(landmark)):
            p[i]=self.project(landmark[i])
        return p
        

