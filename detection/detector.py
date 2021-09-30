
# coding: utf-8

# In[1]:


import paddle
import numpy as np


# In[2]:


class Detector:
    '''识别多组图片'''
    def __init__(self,net_factory,data_size,batch_size,model_path, net_name='ONet'):
        self.net_factory = net_factory
        self.model_path = model_path
        self.data_size = data_size
        self.batch_size = batch_size
        pretrained_dict = paddle.load(self.model_path)
        self.net_factory.set_state_dict(pretrained_dict)
        self.net_name = net_name
    
    def predict(self, databatch):
        batch_size = self.batch_size
        minibatch = []
        cur = 0
        #所有数据总数
        n = databatch.shape[0]
        #将数据整理成固定batch
        while cur<n:
            minibatch.append(databatch[cur:min(cur+batch_size,n),:,:,:])
            cur += batch_size
        cls_pred_list = []
        bbox_pred_list = []
        landmark_pred_list = []
        for idx,data in enumerate(minibatch):
            m = data.shape[0]
            real_size = self.batch_size
            #最后一组数据不够一个batch的处理
            if m<batch_size:
                keep_inds = np.arange(m)
                gap = self.batch_size-m
                while gap>=len(keep_inds):
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds,keep_inds))
                if gap!=0:
                    keep_inds = np.concatenate((keep_inds,keep_inds[:gap]))
                data = data[keep_inds]
                real_size = m

            data = paddle.transpose(paddle.to_tensor(data), [0, 3, 1, 2])
            if self.net_name=='RNet':
                cls_pred, bbox_pred, _ = self.net_factory(data)
                cls_pred_list.append(cls_pred[:real_size])
                bbox_pred_list.append(bbox_pred[:real_size])
            else:
                cls_pred, bbox_pred, landmark_pred = self.net_factory(data)
                cls_pred_list.append(cls_pred[:real_size])
                bbox_pred_list.append(bbox_pred[:real_size])
                landmark_pred_list.append(landmark_pred[:real_size])
        if self.net_name=='RNet':
            return np.concatenate(cls_pred_list, axis=0), np.concatenate(bbox_pred_list, axis=0), None
        else:
            return np.concatenate(cls_pred_list, axis=0), np.concatenate(bbox_pred_list, axis=0), np.concatenate(landmark_pred_list, axis=0)

