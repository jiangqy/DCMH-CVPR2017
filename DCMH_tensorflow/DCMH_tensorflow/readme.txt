=============================DCMH Python Code============================

0.This package contains the source code for the following paper:
    Qing-Yuan Jiang and Wu-Jun Li. Deep Cross-Modal Hashing. CVPR 2017.
1.Author: Qing-Yuan Jiang and Wu-Jun Li.
    Contact: jiangqy@lamda.nju.edu.cn or liwujun@nju.edu.cn
2.We recommend you re-implement DCMH with matconvnet version. (http://lamda.nju.edu.cn/jiangqy/code/DCMH_matlab.zip)
3.We implement DCMH on tensorflow (https://www.tensorflow.org/). Tensorflow version: 1.0.
4.setup: (we use MIRFLICKR-25K dataset as a demo)
    4.0.before you run DCMH tensorflow demo.
    4.1.preprocessing:
        please preprocessing dataset to appropriate input format.
        or you can download dataset from pan.baidu.com
        MIRFLICKR25K:
            link: https://pan.baidu.com/s/1o5jSliFjAezBavyBOiJxew
            password: 8dub
        NUS-WIDE (top-21 concept)：
            link:
            password:
        NUS-WIDE (top-10 concept):
            link:
            password:
        For convenience, I divided all varibles into some single files. After download the dataset, pls merge all variables by yourself.
    4.2.put files in correct path :
            ./datafile
            ./data/imagenet-vgg-f.mat
    4.3.run DCMH_demo.py
5.description:
    DCMH_demo.py:           a demo for DCMH algorithm on MIRFLICKR-25K dataset.
    net_structure_img.py    net structure for image modality.
    net_structure_txt.py    net structure for text modality.
6.if you have any questions about this demo, please feel free to contact Jiang Qing-Yuan (jiangqy@lamda.nju.edu.cn)
