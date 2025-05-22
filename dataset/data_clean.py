##################################################################################
# 数据清洗
# 作者: 欧新宇 (Xinyu Ou, http://ouxinyu.cn)
# 数据集名称：摔倒检测（FallDown）
# 本程序功能:
# 1. 删除MacOS自动生成的文件'.DS_Store'
# 2. 修复标签文件中部分width和height为0导致的计算错误
# 3. 将无法读取或数据损坏的样本保存到badfile_list.txt文件夹，供划分数据时进行排除
###################################################################################

import os
import cv2
import codecs
from xml.dom.minidom import parse
import xml.dom.minidom

# 本地运行时，需要修改数据集的名称和绝对路径，注意和文件夹名称一致
dataset_name = 'dataset'
dataset_path = 'E:\\Soft\\code\\python\\ultralytics'
dataset_root_path = os.path.join(dataset_path, dataset_name)
excluded = ['.DS_Store']    # 被排除的文件
num_bad = 0
num_good = 0
num_folder = 0

# 检测数据集列表是否存在，如果存在则先删除。其中测试集列表是一次写入，因此可以通过'w'参数进行覆盖写入，而不用进行手动删除。
bad_list = os.path.join(dataset_root_path, 'badfile_list.txt')
if os.path.exists(bad_list):
    os.remove(bad_list)

# 执行数据清洗
count = 0
with codecs.open(bad_list, 'a', 'utf-8') as f_bad:
    images = os.listdir(os.path.join(dataset_root_path, 'JPEGImages'))
    for image in images:
        file_name, ext_name = os.path.splitext(image)
        img_path = os.path.join(dataset_root_path, 'JPEGImages', image)
        xml_path = os.path.join(dataset_root_path, 'Annotations', file_name+'.xml')
        try:
            img = cv2.imread(img_path, 1)
            img_shape = img.shape
            DOMTree = xml.dom.minidom.parse(xml_path)
            collection = DOMTree.documentElement
            width = collection.getElementsByTagName('width')
            height = collection.getElementsByTagName('height')
            folder = collection.getElementsByTagName('folder')
            filename = collection.getElementsByTagName('filename')
            path = collection.getElementsByTagName('path')
            width[0].firstChild.data = img_shape[1]
            height[0].firstChild.data = img_shape[0]
            folder[0].firstChild.data = 'JPEGImages'
            filename[0].firstChild.data = image
            path[0].firstChild.data = image
            with open(xml_path, 'w') as f_xml:
                DOMTree.writexml(f_xml)
            # cv2.imwrite(img_path, img)
            num_good += 1
            pass
        except:            
            f_bad.write('{}\n'.format(image))
            num_bad += 1
        if count % 10 == 0:
            print('\r 当前清洗进度：{}/{}，移除损坏文件{}个。'.format(count, len(images), num_bad), end='')
        count += 1
 
print('数据集清洗完成, 损坏文件{}个, 正常文件{}.'.format(num_bad, num_good))