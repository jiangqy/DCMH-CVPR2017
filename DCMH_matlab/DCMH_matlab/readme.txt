=============================DCMH Matlab Code============================

0.This package contains the source code for the following paper:
    Qing-Yuan Jiang and Wu-Jun Li. Deep Cross-Modal Hashing. CVPR 2017.
1.Author: Qing-Yuan Jiang and Wu-Jun Li.
    Contact: jiangqy@lamda.nju.edu.cn or liwujun@nju.edu.cn
2.We implement DCMH on MatConvNet (http://www.vlfeat.org/matconvnet/). MatConvNet version: beta 20.
3.setup: (we use MIRFLICKR-25K dataset as a demo)
    3.0.before you run DCMH matconvnet demo, please make sure that you compile matconvnet on your machine. If not, run ./matconvnet/compilenn.m
    3.1.preprocessing:
        please preprocessing dataset to appropriate input format.
        or you can download dataset from pan.baidu.com
        MIRFLICKR25K:
            link: https://pan.baidu.com/s/1o5jSliFjAezBavyBOiJxew
            password: 8dub
        NUS-WIDE (top-21 concept)：
                link: https://pan.baidu.com/s/13TpINv9GoNTvVdd084K5zQ
                password: kmws
        NUS-WIDE (top-10 concept):
            link: https://pan.baidu.com/s/1GFljcAtWDQFDVhgx6Jv_nQ
            password: ml4y
        For convenience, I divided all varibles into some single files. After download the dataset, pls merge all variables by yourself.
    3.2.put files in correct path :
            ./datafile
            ./data/imagenet-vgg-f.mat
    3.3.setup matconvnet. (run matconvnent/setup.m)
    3.4.run DCMH_demo.m
4.description:
    DCMH_demo.m:            a demo for DCMH algorithm on MIRFLICKR-25K dataset.
    net_structure_img.m     net structure for image modality.
    net_structure_txt.m     net structure for text modality.
    preprocessing.m         download related datasets for the demo.
    update_net.m            sgd based updating method.
5.if you have any questions about this demo, please feel free to contact Jiang Qing-Yuan (jiangqy@lamda.nju.edu.cn)
