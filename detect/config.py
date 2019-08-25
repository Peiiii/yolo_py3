# coding:utf-8

# yolo
TRAIN_INPUT_SIZES = [320, 352, 384, 416]
TEST_INPUT_SIZE = 384
STRIDES = [8, 16, 32]
IOU_LOSS_THRESH = 0.5

# train
BATCH_SIZE = 8
LEARN_RATE_INIT = 1e-4 * BATCH_SIZE / 6
LEARN_RATE_END = 1e-6
WARMUP_PERIODS = 2
MAX_PERIODS = 80
SAVE_STEPS = 3000

DATA_DIR='/home/user/datasets/dataset/CCPD/home/booy/booy/ccpd_dataset/ccpd_base'

GT_PER_GRID = 3
ROOT='detect'



# test
SCORE_THRESHOLD = 0.50    # The threshold of the probability of the classes
IOU_THRESHOLD = 0.1     # The threshold of the IOU when implement NMS

# name and path
WEIGHTS_DIR = ROOT+'/weights'
WEIGHTS_INIT = 'weights/mobilenet_v2_1.0_224.ckpt'
LOG_DIR = 'data/log'
CLASSES = ['text']

