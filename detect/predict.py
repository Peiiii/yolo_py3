# coding: utf-8

import numpy as np
from . import config as cfg
import cv2
import os
import glob
import tensorflow as tf
from .model.head.yolov3 import YOLOV3
import shutil
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import argparse
from .utils import tools
from .eval.evaluator import Evaluator


class Yolo_test(Evaluator):
    def __init__(self,ckpt=None):
        self.imread_mode=cv2.IMREAD_COLOR

        log_dir = os.path.join(cfg.LOG_DIR, 'test')

        with tf.name_scope('input'):
            input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            training = tf.placeholder(dtype=tf.bool, name='training')
        _, _, _, pred_sbbox, pred_mbbox, pred_lbbox = YOLOV3(training).build_nework(input_data)
        with tf.name_scope('summary'):
            tf.summary.FileWriter(log_dir).add_graph(tf.get_default_graph())
        self.__sess = tf.Session()
        saver = tf.train.Saver()
        if not ckpt:
            ckpt=tf.train.latest_checkpoint(cfg.WEIGHTS_DIR)

        print('restore from %s ..'%(ckpt))
        saver.restore(self.__sess, ckpt)
        print('restore model succeeded.')
        super(Yolo_test, self).__init__(self.__sess, input_data, training, pred_sbbox, pred_mbbox, pred_lbbox)

    def detect_image(self, image):
        original_image = np.copy(image)
        bboxes = self.get_bbox(image)
        image = tools.draw_bbox(original_image, bboxes, self._classes)
        return image
    def predict_from_file(self,fp):
        img=cv2.imread(fp)
        return self.detect_image(img)

    def parse_plate_str(self,fn):
        provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                     "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
        alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                     'W',
                     'X', 'Y', 'Z', 'O']
        ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
               'X',
               'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
        parts=fn.split('.')[0].split('-')
        char_indexes=parts[4].split('_')
        char_indexes=[int(i) for i in char_indexes]
        province=provinces[char_indexes[0]]
        alpha=alphabets[char_indexes[1]]
        ad=[ads[i] for i in char_indexes[2:]]
        plateStr=province+alpha+''.join(ad)
        return plateStr





if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # demo()

    pass
