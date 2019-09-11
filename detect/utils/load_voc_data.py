import os
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import sys

# f='/home/ocr/wp/datasets/aihero/AI+Hero_数据集3/文字验证码框选标注数据集/tag_pic/0a0ad5ef75feb1c8023eff76d245b6d3.xml'
def loadxml(f):
    tree=ET.parse(f)
    root=tree.getroot()
    filename=root.find('filename').text
    # print(filename)
    objects=[]
    for object in root.findall('object'):
        bndbox=object.find('bndbox')
        xmin=bndbox.find('xmin').text
        ymin=bndbox.find('ymin').text
        xmax=bndbox.find('xmax').text
        ymax=bndbox.find('ymax').text
        obj=[xmin,ymin,xmax,ymax]
        objects.append(obj)
    return objects
