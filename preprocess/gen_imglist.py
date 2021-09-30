
# coding: utf-8

# In[1]:


import numpy as np
npr=np.random
import os
import sys
import argparse

def main(args):
    data_dir='../data/'

    '''将pos,part,neg,landmark四者混在一起'''
    size = args.input_size
    with open(os.path.join(data_dir,'%d/pos_%d.txt'%(size,size)),'r') as f:
        pos=f.readlines()
    with open(os.path.join(data_dir,'%d/neg_%d.txt'%(size,size)),'r') as f:
        neg=f.readlines()
    with open(os.path.join(data_dir,'%d/part_%d.txt'%(size,size)),'r') as f:
        part=f.readlines()
    with open(os.path.join(data_dir,'%d/landmark_%d_aug.txt'%(size,size)),'r') as f:
        landmark=f.readlines()
    dir_path=os.path.join(data_dir, '%d'%(size))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(os.path.join(dir_path,'train_%d.txt'%(size)),'w') as f:
        nums=[len(neg),len(pos),len(part)]
        base_num=250000
        print('neg数量：{} pos数量：{} part数量:{} landmark数量:{} 基数:{}'.format(len(neg),len(pos), len(landmark), len(part), base_num))
        if len(neg)>base_num*3:
            neg_keep=npr.choice(len(neg),size=base_num*3,replace=True)
        else:
            neg_keep=npr.choice(len(neg),size=len(neg),replace=True)
        sum_p=len(neg_keep)//3
        pos_keep=npr.choice(len(pos),sum_p,replace=True)
        part_keep=npr.choice(len(part),sum_p,replace=True)
        landmark_keep=npr.choice(len(landmark),sum_p,replace=True)
        print('neg数量：{} pos数量：{} part数量:{} landmark数量:{}'.format(len(neg_keep), len(pos_keep), len(part_keep), len(landmark_keep)))
        for i in pos_keep:
            f.write(pos[i])
        for i in neg_keep:
            f.write(neg[i])
        for i in part_keep:
            f.write(part[i])
        for i in landmark_keep:
            f.write(landmark[i])

def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('input_size', type=int, help='The input size for specific net')
    
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))