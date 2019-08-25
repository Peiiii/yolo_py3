from detect.predict import Yolo_test
from detect.train import Yolo_train
import cv2,os

def train():
    Y=Yolo_train()
    Y.train()
def test():
    T=Yolo_test()
    img=cv2.imread('data/demo/6.jpg')
    img=T.detect_image(img)
    save_path='data/output/6.jpg'

    cv2.imencode('.jpg', img)[1].tofile(save_path)
    print('detect img to %s' % (save_path))


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    train()
    test()