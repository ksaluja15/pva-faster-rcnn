import _init_paths
from fast_rcnn.test import test_image
from fast_rcnn.config import cfg,cfg_from_file
from datasets.custom_dataset_factory import get_imdb
import caffe
import argparse
import pprint
import time, os, sys

if __name__ == '__main__':

    image_path = '/home/kunal/gumgum_object_detection/datasets_and_results/data_devkits/GSW_devkit/data/Images/Adidas_KIA_23_4.JPEG'
    cfg_file ='/home/kunal/pva-faster-rcnn/models/pvanet/cfgs/submit_1019.yml'
    deploy = '/home/kunal/pva-faster-rcnn/models/pvanet/pva9.1/custom_dataset.pt'
    weights = '/home/kunal/pva-faster-rcnn/output/pvanet_frcnn_iter_100000.caffemodel'
    classes = ['__background__', # always index 0
                'XOJet_1', 'amex_1', 'amex_2', 'amex_3',
                         'amex_4', 'brita_1', 'budlight_1', 'dk_1', 'dk_2',
                         'dk_3', 'jbl_1', 'kia_1', 'kp_1','kp_2','kp_3','kp_4','kp_5','mcdonalds_1',
                         'mountaindew_1', 'nike_1', 'ps4_1',
                         'ps4_2', 'ps4_3', 'redbull_1', 'redbull_2','sf_1']

    GPU_ID = 0
    cfg_from_file(cfg_file)
    print('Using config:')
    pprint.pprint(cfg)


    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)
    net = caffe.Net(deploy, weights, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(weights))[0]

    test_image(net, image_path, classes, vis=True)
