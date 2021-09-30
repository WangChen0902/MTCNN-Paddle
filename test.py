
# coding: utf-8

# In[1]:


import sys
from detection.MtcnnDetector import MtcnnDetector
from detection.detector import Detector
from detection.fcn_detector import FcnDetector
from train.model import PNet,RNet,ONet
import cv2
import os
import numpy as np
import train.config as config


# In[ ]:


test_mode=config.test_mode
thresh=config.thresh
min_face_size=config.min_face
stride=config.stride
detectors=[None,None,None]
# 模型放置位置
model_path=['./model/PNet/Final.pdparams', './model/RNet/Final.pdparams', './model/ONet/Final.pdparams']
batch_size=config.batches
pnet = PNet(phase='test')
rnet = RNet(phase='test')
onet = ONet(phase='test')
PNet=FcnDetector(pnet, model_path[0])
detectors[0]=PNet


if test_mode in ["RNet", "ONet"]:
    RNet = Detector(rnet, 24, batch_size[1], model_path[1])
    detectors[1] = RNet


if test_mode == "ONet":
    ONet = Detector(onet, 48, batch_size[2], model_path[2])
    detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh)
out_path=config.out_path
if config.input_mode=='1':
    testset_folders = {}
    testset_numbers = {}
    testset_folder = 'data/fddb/images/'
    testset_list = 'data/fddb/FDDB-folds/'
    testset_list_names = os.listdir(testset_list)
    for test_name in testset_list_names:
        print(test_name)
        if test_name[0]=='.':
            continue
        test_name_idx = int(test_name.split('-')[-1].split('.')[0])
        testset_file = os.path.join(testset_list, test_name)
        with open(testset_file, 'r') as fr:
            test_dataset = fr.read().split()
        num_images = len(test_dataset)
        testset_folders[test_name_idx] = test_dataset
        testset_numbers[test_name_idx] = num_images

    resize = 1

    # testing begin
    data_idx = 0
    for test_idx in testset_folders:
        data_idx += 1
        for i, img_name in enumerate(testset_folders[test_idx]):
            # print(img_name)
            image_path = testset_folder + img_name + '.jpg'
            img_raw = cv2.imread(image_path)

            boxes_c,landmarks = mtcnn_detector.detect(img_raw)

            # save dets
            save_name = 'FDDB_Evaluation/fddb_evaluation/' + str(test_idx) + '/' +  img_name.replace('/', '_') + ".txt"
            dirname = os.path.dirname(save_name)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            with open(save_name, "w") as fw:
                fw.write('{:s}\n'.format(img_name.replace('/', '_')))
                fw.write('{:.1f}\n'.format(boxes_c.shape[0]))
                for k in range(boxes_c.shape[0]):
                    xmin = boxes_c[k,0]
                    ymin = boxes_c[k,1]
                    xmax = boxes_c[k,2]
                    ymax = boxes_c[k,3]
                    score = boxes_c[k,4]
                    w = xmax - xmin + 1
                    h = ymax - ymin + 1
                    # fw.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.10f}\n'.format(xmin, ymin, w, h, score))
                    fw.write('{:d} {:d} {:d} {:d} {:.10f}\n'.format(int(xmin), int(ymin), int(w), int(h), score))
            print('im_detect: {:d}/{:d} '.format(i+1 + num_images*(data_idx-1), num_images*10))
