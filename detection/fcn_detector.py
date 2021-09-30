
# coding: utf-8

# In[1]:


import paddle
import sys
sys.path.append('../')
import train.config as config


# In[2]:


class FcnDetector:
    '''识别单张图片'''
    def __init__(self,net_factory, model_path):
        self.net_factory = net_factory
        self.model_path = model_path
        pretrained_dict = paddle.load(self.model_path)
        self.net_factory.set_state_dict(pretrained_dict)

    def predict(self, databatch):
        height,width,_ = databatch.shape
        image_reshape = paddle.reshape(paddle.to_tensor(databatch), [1, height, width, 3])
        image_reshape = paddle.transpose(image_reshape, [0, 3, 1, 2])
        cls_pred, bbox_pred, _ = self.net_factory(image_reshape)
        cls_pred = paddle.squeeze(cls_pred, axis=0)
        bbox_pred = paddle.squeeze(bbox_pred, axis=0)
        cls_pred = paddle.transpose(cls_pred, [1, 2, 0])
        bbox_pred = paddle.transpose(bbox_pred, [1, 2, 0])
        # print(cls_pred)
        # print(bbox_pred)
        return cls_pred.numpy(), bbox_pred.numpy()

        

