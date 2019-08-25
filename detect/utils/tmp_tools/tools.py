# coding: utf-8

import numpy as np
import config as cfg
import cv2
import os,random
import glob
import shutil

def merge_and_split_dataset(src_dirs,train_dir,val_dir):
    # src_dirs=[
    #
    # ]
    # train_dir=''
    # val_dir=''
    shutil.rmtree(train_dir) if os.path.exists(train_dir) else None
    shutil.rmtree(val_dir) if os.path.exists(val_dir) else None

    os.makedirs(train_dir) if not os.path.exists(train_dir) else None
    os.makedirs(val_dir) if not os.path.exists(val_dir) else None
    all_imgs=[]
    for dir in src_dirs:
        all_imgs+=glob.glob(dir+'/*.jpg')
    random.shuffle(all_imgs)
    train_imgs=all_imgs[:-10000]
    val_imgs=all_imgs[-10000:]
    print('start copy files')
    for i,img in enumerate(train_imgs):
        fdst=train_dir + '/' + os.path.basename(img)
        print('i=%s, copy from %s to %s'%(i,img,fdst))
        shutil.copyfile(img, fdst)

    for i,img in enumerate(val_imgs):
        fdst=val_dir + '/' + os.path.basename(img)
        print('i=%s, copy from %s to %s'%(i,img,fdst))
        shutil.copyfile(img, fdst)

    print('success.')



