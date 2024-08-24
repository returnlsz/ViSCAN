# Generate dataset indices vid_train_coco.json and vid_val_coco.json under the annotations folder for YOLOX training. You need to modify the values of data_dir and jsonname at the bottom of the script.
# usage: python xml2coco.py
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET
import json
import re
import math
import numpy as np
class_names = ["corrosion"]  # 本人转换的标签只有1个
img_id = -1  # 全局变量
anno_id = -1  # 全局变量

def voc2coco(data_dir, dir_save, jsonname):  # 获取所有的xml文件列表
    dataset_path = data_dir
    # 输入train/val.npy
    train_xmls = np.load(dataset_path,allow_pickle=True).tolist()
    for lst in train_xmls:
        for i, item in enumerate(lst):
            lst[i] = re.sub(r'\.jpg$', '.xml', item) # 替换字符串结尾的jpg为xml
            lst[i] = re.sub(r'Data', 'Annotations', lst[i]) # 替换字符串中的Data为Annotations
    train_xmls = [item for sublist in train_xmls for item in sublist]
    # xml_dir = data_dir
    # # 多个文件夹操作
    # train_xmls = list()
    # for tt in tqdm(os.listdir(xml_dir)):  # 遍历文件夹获取xml文件列表
    #     one_dir_path = os.path.join(xml_dir, tt)
    #     train_xmls = train_xmls + [os.path.join(one_dir_path, n) for n in os.listdir(one_dir_path)]
    
    ## 单个文件夹操作
    # train_xmls = [os.path.join(xml_dir, n) for n in os.listdir(xml_dir)]
    # print(len(train_xmls))
    print('got xmls')

    train_coco = xml2coco(train_xmls)
    with open(os.path.join(dir_save, jsonname), 'w') as f:  # 保存进入指定的coco文件夹
        json.dump(train_coco, f, ensure_ascii=False, indent=2)
    print('done')


def xml2coco(xmls):  # 将多个xml文件转换成coco格式
    coco_anno = {'info': {"description": "yip_make"}, 'images': [], 'licenses': [], 'annotations': [], 'categories': []}
    coco_anno['categories'] = [{'supercategory': "", 'id': i, 'name': j} for i, j in enumerate(class_names)]
    global img_id, anno_id
    for fxml in tqdm(xmls):  # 逐一对xml文件进行处理
        try:
            tree = ET.parse(fxml)
            objects = tree.findall('object')
        except:
            print('err xml file: ', fxml)
            continue
        img_id += 1  # 无论图片有无标签，该图片也算img_id
        size = tree.find('size')
        ih = int(size.find('height').text)
        iw = int(size.find('width').text)
        img_name = fxml.replace("Annotations", "Data").replace("xml", "jpg")  # 获得原始图片的路径
        img_info = {}
        img_info['id'] = img_id
        img_info['file_name'] = img_name
        img_info['height'] = ih
        img_info['width'] = iw
        coco_anno['images'].append(img_info)
        if len(objects) < 1:
            print('no object in ', fxml)  # 打印没有bndbox标签的xml文件
            continue
        for obj in objects:  # 获取xml内的所有bndbox标签
            cls_name = obj.find('name').text
            bbox = obj.find('bndbox')
            x1 = int(math.floor(float(bbox.find('xmin').text)))
            y1 = int(math.floor(float(bbox.find('ymin').text)))
            x2 = int(math.floor(float(bbox.find('xmax').text)))
            y2 = int(math.floor(float(bbox.find('ymax').text)))
            if x2 < x1 or y2 < y1:
                print('bbox not valid: ', fxml)
                continue
            anno_id += 1
            bb = [x1, y1, x2 - x1, y2 - y1]
            categery_id = class_names.index(cls_name)
            area = (x2 - x1) * (y2 - y1)
            anno_info = {}
            anno_info['segmentation'] = []
            anno_info['area'] = area
            anno_info['image_id'] = img_id
            anno_info['bbox'] = bb
            anno_info['iscrowd'] = 0
            anno_info['category_id'] = categery_id
            anno_info['id'] = anno_id
            coco_anno['annotations'].append(anno_info)

    return coco_anno


if __name__ == '__main__':
    ## 单个文件夹
    # data_dir = 'ILSVRC2015/Annotations/VID/train/01_1667_0001-1500'
    
    # data_dir含多个文件夹，每个文件夹有多个xml文件
    # data_dir = 'ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0000/'  # VID格式xml训练集或测试集的文件夹
    # data_dir = 'ILSVRC2015/Annotations/VID/val/'  # VID格式xml训练集或测试集的文件夹
    # data_dir = 'train_seq.npy'
    data_dir = 'val_seq.npy'
    dir_save = 'annotations'  # coco文件保存的文件夹
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    # jsonname = 'vid_train_coco.json'  # coco文件名
    jsonname = 'vid_val_coco.json'  # coco文件名val
    voc2coco(data_dir, dir_save, jsonname)

