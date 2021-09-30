# MTCNN-Paddle
This is a Paddle version of Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks .

精度不够，不参与正式比赛的提交

### 运行
训练：<br><br>
将目录cd到preprocess上，<br>
python gen_12net_data.py生成三种pnet数据，<br>
python gen_landmark_aug.py 12 生成pnet的landmark数据，<br>
python gen_imglist.py 12 整理到一起，<br>
将目录cd到train上python train.py 12 训练pnet<br><br>
将目录cd到preprocess上，<br>
python gen_hard_example.py 12 生成三种rnet数据，<br>
python gen_landmark_aug.py 24 生成rnet的landmark数据,<br>
python gen_imglist.py 24 整理到一起，<br>
将目录cd到train上python train.py 24 训练rnet<br><br>
将目录cd到preprocess上，<br>
python gen_hard_example.py 24 生成三种onet数据，<br>
python gen_landmark_aug.py 48 生成onet的landmark数据,<br>
python gen_imglist.py 48 整理到一起，<br>
将目录cd到train上python train.py 48 训练onet<br><br>
测试:<br><br>
python test.py<br>
