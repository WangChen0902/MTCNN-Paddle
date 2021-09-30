
# coding: utf-8

# In[5]:


from model import PNet,RNet,ONet
from mtcnn_loss import PNetLoss, RNetLoss, ONetLoss
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import DataLoader
import argparse
import os
import sys
import config as FLAGS
import math
sys.path.append('../')
from data.mtcnn_loader import MtcnnLoader
# from train_model import train


def main(args):
    paddle.set_device('gpu')
    size=args.input_size
    resume_net = args.resume_net
    resume_epoch = args.resume_epoch
    base_dir=os.path.join('../data/', str(size))
    
    net = None
    if size==12:
        net_name = 'PNet'
        batch_size = FLAGS.batch_size[0]
        net = PNet()
        end_epoch = FLAGS.end_epoch[0]
        dataset = MtcnnLoader('../data/12/train_12.txt')
        criterion = PNetLoss()
        radio_cls_loss = [1.0, 0.5, 0.5]
    elif size==24:
        net_name = 'RNet'
        batch_size = FLAGS.batch_size[1]
        net = RNet()
        end_epoch = FLAGS.end_epoch[1]
        dataset = MtcnnLoader('../data/24/train_24.txt')
        criterion = RNetLoss()
        radio_cls_loss = [1.0, 0.5, 0.5]
    elif size==48:
        net_name='ONet'
        batch_size = FLAGS.batch_size[2]
        net = ONet()
        end_epoch=FLAGS.end_epoch[2]
        dataset = MtcnnLoader('../data/48/train_48.txt')
        criterion = ONetLoss()
        radio_cls_loss = [1.0, 0.5, 1.0]
    model_path = os.path.join('../model/', net_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    prefix = os.path.join(model_path, net_name)
    display = FLAGS.display
    lr = FLAGS.lr

    optimizer = optim.Momentum(learning_rate=lr, momentum=0.9, parameters=net.parameters())

    if resume_net is not None:
        print('Loading resume network...')
        net_state_dict = paddle.load(resume_net + '.pdparams')
        opt_state_dict = paddle.load(resume_net + '.pdopt')
        net.set_state_dict(net_state_dict)
        optimizer.set_state_dict(opt_state_dict)

    net.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    epoch_size = math.ceil(len(dataset)/batch_size)

    current_lr = lr
    for epoch in range(resume_epoch, end_epoch):
        if epoch % 2 == 0:
            paddle.save(net.state_dict(), os.path.join(model_path, 'epoch_' + str(epoch) + '.pdparams'))
            paddle.save(optimizer.state_dict(), os.path.join(model_path,  'epoch_' + str(epoch) + '.pdopt'))
        if epoch in FLAGS.LR_EPOCH:
            current_lr = current_lr * 0.1
            optimizer.set_lr(current_lr)
        for i, data in enumerate(loader()):
            img, label, bbox_target, landmark_target = data
            img = paddle.transpose(img, [0, 3, 1, 2])
            target = (label, bbox_target, landmark_target)
            out = net(img)
            cls_loss, bbox_loss, landmark_loss, accuracy = criterion(out, target)
            loss = radio_cls_loss[0]*cls_loss + radio_cls_loss[1]*bbox_loss + radio_cls_loss[2]*landmark_loss
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            print('Epoch:{}/{} || Epochiter: {}/{} || Cls: {:.4f} BBox: {:.4f} Landm: {:.4f} Loss: {:.4f} Acc: {:.4f} || LR: {:.8f}'
              .format(epoch+1, end_epoch, i, epoch_size, cls_loss.item(), bbox_loss.item(), landmark_loss.item(), loss.item(), accuracy.item(), current_lr))

    paddle.save(net.state_dict(), os.path.join(model_path, 'Final.pdparams'))
    paddle.save(optimizer.state_dict(), os.path.join(model_path,  'Final.pdopt'))
    # train(net_factory,prefix,end_epoch,base_dir,display,lr)


def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('input_size', type=int, help='The input size for specific net')
    parser.add_argument('--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume epoch for retraining')
    
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

