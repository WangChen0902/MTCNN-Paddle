
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def IOU(box,boxes):
    #box面积
    box_area=(box[2]-box[0]+1)*(box[3]-box[1]+1)
    #boxes面积,[n,]
    area=(boxes[:,2]-boxes[:,0]+1)*(boxes[:,3]-boxes[:,1]+1)
    #重叠部分左上右下坐标
    xx1=np.maximum(box[0],boxes[:,0])
    yy1=np.maximum(box[1],boxes[:,1])
    xx2=np.minimum(box[2],boxes[:,2])
    yy2=np.minimum(box[3],boxes[:,3])
    
    #重叠部分长宽
    w=np.maximum(0,xx2-xx1+1)
    h=np.maximum(0,yy2-yy1+1)
    #重叠部分面积
    inter=w*h
    return inter/(box_area+area-inter+1e-10)


# In[3]:

def read_annotation(base_dir, label_path):
    data = dict()
    images = []
    bboxes = []

    with open(label_path,'r') as f:
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
            path = label_path.replace('label.txt','images/') + path
            imgs_path.append(path)
        else:
            line = line.split(' ')
            label = [float(x) for x in line]
            labels.append(label)
    words.append(labels)

    for index in range(len(imgs_path)):
        img_path = imgs_path[index]
        labels = words[index]
        one_image_bboxes = [] 
        for idx, label in enumerate(labels):
            # bbox
            box=[label[0], label[1], label[0]+label[2], label[1]+label[3]]
            one_image_bboxes.append(box)
        images.append(img_path)
        bboxes.append(one_image_bboxes)

    data['images'] = images
    data['bboxes'] = bboxes
    return data


def convert_to_square(box):
    square_box=box.copy()
    h=box[:,3]-box[:,1]+1
    w=box[:,2]-box[:,0]+1
    #找寻正方形最大边长
    max_side=np.maximum(w,h)
    
    square_box[:,0]=box[:,0]+w*0.5-max_side*0.5
    square_box[:,1]=box[:,1]+h*0.5-max_side*0.5
    square_box[:,2]=square_box[:,0]+max_side-1
    square_box[:,3]=square_box[:,1]+max_side-1
    return square_box

