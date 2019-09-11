from detect.predict import Yolo_test as Detector
import glob,os,shutil
import cv2
import numpy as np
from PIL import Image


def demo():
	input_dir='data/demo'
	output_dir='data/output'
	output_file='data/result.json'
	from detect.apis import Detector
	D = Detector()
	D.detect_dir_to_dir(input_dir, output_dir, output_file,verbose=True)

def show():
	test_data_dir = '/home/ocr/wp/datasets/aihero/AI+Hero_测试数据集/AI Hero_测试数据集'
	test_output_dir = '/home/ocr/wp/datasets/outputs/aihero/detect_out_test'

	D = Detector()
	fs = glob.glob(test_data_dir + '/*.jpg')
	for f in fs:
		imgs, coors = D.predict_from_file(f)
		print(len(imgs))
		D.show_img(imgs[0])
		input(coors[0])


if __name__=='__main__':
	demo()

