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
        log_dir = os.path.join(cfg.LOG_DIR, 'test')
        # test_weight_path = os.path.join(cfg.WEIGHTS_DIR, test_weight)

        with tf.name_scope('input'):
            input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            training = tf.placeholder(dtype=tf.bool, name='training')
        _, _, _, pred_sbbox, pred_mbbox, pred_lbbox = YOLOV3(training).build_nework(input_data)
        with tf.name_scope('summary'):
            tf.summary.FileWriter(log_dir).add_graph(tf.get_default_graph())
        self.__sess = tf.Session()
        # net_vars = tf.get_collection('YoloV3')
        saver = tf.train.Saver()
        if not ckpt:
            ckpt=tf.train.latest_checkpoint(cfg.WEIGHTS_DIR)
        # ckpt=tf.train.latest_checkpoint(cfg.WEIGHTS_DIR)
        print(ckpt)
        saver.restore(self.__sess, ckpt)
        super(Yolo_test, self).__init__(self.__sess, input_data, training, pred_sbbox, pred_mbbox, pred_lbbox)

    def detect_image(self, image):
        original_image = np.copy(image)
        bboxes = self.get_bbox(image)
        image = tools.draw_bbox(original_image, bboxes, self._classes)
        # self.__sess.close()
        return image
    def predict_from_file(self,fp):
        img=cv2.imread(fp)
        print(img)
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

    def build_dataset(self,src_dir,tgt_dir):

        img_file_names=glob.glob(src_dir+'/*.jpg')
        shutil.rmtree(tgt_dir) if os.path.exists(tgt_dir) else None
        os.makedirs(tgt_dir) if not os.path.exists(tgt_dir) else None
        for i,fp in enumerate(img_file_names):
            fn=os.path.basename(fp)
            plate_str=self.parse_plate_str(fn)
            image = cv2.imread(fp)
            image = self.detect_image(image, 'new dir')
            if image is None:
                continue
            save_path =tgt_dir+'/'+str(i)+'_'+plate_str+'.jpg'
            # cv2.imwrite(save_path, image)
            cv2.imencode('.jpg', image)[1].tofile(save_path)
            print('detect img to %s'%(save_path))
def demo():
    src_dir='data/demo'
    dst_dir='data/output'
    tgt_dir=dst_dir
    T=Yolo_test() 
    img_file_names = glob.glob(src_dir + '/*.jpg')
    shutil.rmtree(tgt_dir) if os.path.exists(tgt_dir) else None
    os.makedirs(tgt_dir) if not os.path.exists(tgt_dir) else None
    for i, fp in enumerate(img_file_names):
        fn = os.path.basename(fp)
        plate_str = T.parse_plate_str(fn)
        image = cv2.imread(fp)
        image = T.detect_image(image)
        save_path = tgt_dir + '/' + str(i) + '_' + plate_str + '.jpg'
        # cv2.imwrite(save_path, image)
        cv2.imencode('.jpg', image)[1].tofile(save_path)
        print('detect img to %s' % (save_path))
def demo2():
    T=Yolo_test()
    img=cv2.imread('data/demo/6.jpg')
    img=T.detect_image(img)
    save_path='data/output/6.jpg'
    cv2.imencode('.jpg', img)[1].tofile(save_path)
    print('detect img to %s' % (save_path))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # main2()
    demo2()
