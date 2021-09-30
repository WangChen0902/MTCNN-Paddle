import paddle
import paddle.nn as nn
import numpy as np

class PNetLoss(nn.Layer):
    def __init__(self):
        super(PNetLoss, self).__init__()
    
    def forward(self, predictions, targets):
        cls_pred, bbox_pred, landmark_pred = predictions
        label, bbox_target, landmark_target = targets

        cls_pred = paddle.squeeze(cls_pred, [1,2])#[batch,2]
        cls_loss = cls_ohem(cls_pred, label)
        
        bbox_pred = paddle.squeeze(bbox_pred,[1,2])#[bacth,4]
        bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
        
        landmark_pred = paddle.squeeze(landmark_pred, [1,2])#[batch,10]
        landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)
        
        accuracy = cal_accuracy(cls_pred, label)
        # print(cls_loss, bbox_loss, landmark_loss, accuracy)
        return cls_loss, bbox_loss, landmark_loss, accuracy


class RNetLoss(nn.Layer):
    def __init__(self):
        super(RNetLoss, self).__init__()
    
    def forward(self, predictions, targets):
        cls_pred, bbox_pred, landmark_pred = predictions
        label, bbox_target, landmark_target = targets

        cls_loss = cls_ohem(cls_pred, label)
        bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
        landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)
        accuracy = cal_accuracy(cls_pred, label)

        return cls_loss, bbox_loss, landmark_loss, accuracy


class ONetLoss(nn.Layer):
    def __init__(self):
        super(ONetLoss, self).__init__()
    
    def forward(self, predictions, targets):
        cls_pred, bbox_pred, landmark_pred = predictions
        label, bbox_target, landmark_target = targets

        cls_loss = cls_ohem(cls_pred, label)
        bbox_loss = bbox_ohem(bbox_pred, bbox_target, label)
        landmark_loss = landmark_ohem(landmark_pred, landmark_target, label)
        accuracy = cal_accuracy(cls_pred, label)

        return cls_loss, bbox_loss, landmark_loss, accuracy


def cls_ohem(cls_pred,label):
    '''计算类别损失
    参数：
      cls_pred：预测类别，是否有人
      label：真实值
    返回值：
      损失
    '''
    zeros = paddle.zeros_like(label)
    #只把pos的label设定为1,其余都为0
    label_filter_invalid = paddle.where(paddle.less_than(label, zeros), zeros, label)
    #类别size[2*batch]
    # print(cls_pred)
    # print(label)
    num_cls_pred = cls_pred.shape[0] * cls_pred.shape[1]
    cls_pred_reshpae = paddle.reshape(cls_pred, [num_cls_pred,-1])
    label_int = label_filter_invalid.astype('int32')
    #获取batch数
    num_row=cls_pred.shape[0]
    #对应某一batch而言，batch*2为非人类别概率，batch*2+1为人概率类别,indices为对应 cls_pred_reshpae
    #应该的真实值，后续用交叉熵计算损失
    row_np = np.array(list(range(num_row)))*2
    row = paddle.to_tensor(row_np)
    indices_ = row + label_int
    #真实标签对应的概率
    label_prob = paddle.squeeze(paddle.gather(cls_pred_reshpae,indices_))
    loss = -paddle.log(label_prob + 1e-10)
    zeros = paddle.zeros_like(label_prob, dtype='float32')
    ones = paddle.ones_like(label_prob, dtype='float32')
    #统计neg和pos的数量
    valid_inds = paddle.where(label<zeros,zeros,ones)
    num_valid = paddle.sum(valid_inds)
    #选取70%的数据
    keep_num = paddle.cast(num_valid*0.7, dtype='int32')
    #只选取neg，pos的70%损失
    loss = loss * valid_inds
    loss,_ = paddle.topk(loss, k=keep_num)
    return paddle.mean(loss)


def bbox_ohem(bbox_pred,bbox_target,label):
    '''计算box的损失'''
    zeros_index = paddle.zeros_like(label, dtype='float32')
    ones_index = paddle.ones_like(label, dtype='float32')
    label = label.astype('float32')
    #保留pos和part的数据
    valid_inds = paddle.where(paddle.equal(paddle.abs(label), ones_index), ones_index, zeros_index)
    #计算平方差损失
    square_error = paddle.square(paddle.squeeze(bbox_pred)-bbox_target)
    square_error = paddle.sum(square_error,axis=1)
    #保留的数据的个数
    num_valid = paddle.sum(valid_inds)
    keep_num = paddle.cast(num_valid,dtype='int32')
    #保留pos和part部分的损失
    square_error = square_error * valid_inds
    square_error, _ = paddle.topk(square_error, k=keep_num)
    return paddle.mean(square_error)
    

def landmark_ohem(landmark_pred,landmark_target,label):
    '''计算关键点损失'''
    ones = paddle.ones_like(label, dtype='float32')
    zeros = paddle.zeros_like(label, dtype='float32')
    neg_twos = paddle.full_like(label, -2, dtype='float32')
    label = label.astype('float32')
    #只保留landmark数据
    valid_inds = paddle.where(paddle.equal(label,neg_twos), ones, zeros)
    #计算平方差损失
    square_error = paddle.square(paddle.squeeze(landmark_pred)-landmark_target)
    square_error = paddle.sum(square_error,axis=1)
    #保留数据个数
    num_valid = paddle.sum(valid_inds)
    keep_num = paddle.cast(num_valid, dtype='int32')
    #保留landmark部分数据损失
    square_error = square_error*valid_inds
    square_error, _ = paddle.topk(square_error, k=keep_num)
    return paddle.mean(square_error)


def cal_accuracy(cls_pred,label):
    '''计算分类准确率'''
    #预测最大概率的类别，0代表无人，1代表有人
    pred = paddle.argmax(cls_pred,axis=1)
    label_int = paddle.cast(label, 'int64')
    zeros = paddle.zeros_like(label)
    #保留label>=0的数据，即pos和neg的数据
    cond = paddle.fluid.layers.where(paddle.greater_equal(label_int,zeros))
    picked = paddle.squeeze(cond)
    #获取pos和neg的label值
    label_picked = paddle.gather(label_int,picked)
    pred_picked = paddle.gather(pred,picked)
    #计算准确率
    accuracy_op = paddle.mean(paddle.cast(paddle.equal(label_picked,pred_picked), 'float32'))
    return accuracy_op
