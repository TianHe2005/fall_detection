#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练YOLO模型脚本
使用Ultralytics库训练自定义数据集
"""

import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO

# 设置基本路径
ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = ROOT / 'dataset'
OUTPUT_PATH = ROOT / 'runs'

# 创建YOLO格式数据集目录
def create_yolo_dataset():
    # 创建YOLO格式数据集目录
    yolo_dataset_path = ROOT / 'dataset_yolo'
    if yolo_dataset_path.exists():
        print(f"YOLO数据集目录已存在: {yolo_dataset_path}")
    else:
        os.makedirs(yolo_dataset_path, exist_ok=True)
        print(f"创建YOLO数据集目录: {yolo_dataset_path}")
    
    # 创建images和labels目录
    for split in ['train', 'val']:
        (yolo_dataset_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (yolo_dataset_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 读取类别列表
    with open(DATASET_PATH / 'label_list.txt', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # 处理训练集和验证集
    for split in ['train', 'val']:
        split_file = DATASET_PATH / f'{split}.txt'
        with open(split_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            img_path, xml_path = line.strip().split()
            img_path = DATASET_PATH / img_path
            xml_path = DATASET_PATH / xml_path
            
            # 复制图像文件到YOLO数据集目录
            img_filename = os.path.basename(img_path)
            dst_img_path = yolo_dataset_path / 'images' / split / img_filename
            shutil.copy(img_path, dst_img_path)
            
            # 转换XML标注为YOLO格式
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            dst_label_path = yolo_dataset_path / 'labels' / split / label_filename
            convert_voc_to_yolo(xml_path, dst_label_path, classes)
    
    # 创建数据集配置文件
    dataset_yaml = {
        'path': str(yolo_dataset_path),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(classes)}
    }
    
    with open(yolo_dataset_path / 'dataset.yaml', 'w') as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)
    
    print(f"YOLO格式数据集准备完成: {yolo_dataset_path}")
    return yolo_dataset_path / 'dataset.yaml'

# 将VOC XML格式转换为YOLO txt格式
def convert_voc_to_yolo(xml_file, txt_file, classes):
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # 获取图像尺寸
    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    with open(txt_file, 'w') as f:
        for obj in root.findall('object'):
            # 获取类别名称
            class_name = obj.find('name').text
            # 跳过不在类别列表中的对象
            if class_name not in classes:
                continue
            # 获取类别索引
            class_idx = classes.index(class_name)
            # 获取边界框坐标
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            # 转换为YOLO格式 (x_center, y_center, width, height) - 归一化坐标
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height
            # 写入YOLO格式标注
            f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

# 训练模型
def train_model(data_yaml, model_size='n', epochs=100, batch_size=16, imgsz=640, device='0'):
    # 选择模型大小 (n, s, m, l, x)
    # 使用本地模型文件路径，避免从GitHub下载
    model_path = ROOT / f'yolov8{model_size}.pt'
    if not model_path.exists():
        print(f"警告: 本地模型文件 {model_path} 不存在，将尝试使用默认路径")
        model_path = f'yolov8{model_size}.pt'
    
    print(f"加载模型: {model_path}")
    model = YOLO(str(model_path))
    
    # 设置环境变量，强制AMP检查使用CPU
    import os
    original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', device if isinstance(device, str) else str(device))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 临时禁用CUDA以进行AMP检查
    print("临时禁用CUDA以进行AMP检查...")
    
    # 设置训练参数
    try:
        # 恢复CUDA环境变量，以便后续训练使用GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
        print(f"AMP检查完成，恢复CUDA设备: {original_cuda_devices}")
        
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            patience=50,  # 早停耐心值
            save=True,    # 保存模型
            device=device,   # 使用指定设备
            project=str(OUTPUT_PATH),
            name=f'yolov8{model_size}_custom',
            pretrained=True,
            optimizer='Adam',
            lr0=0.01,     # 初始学习率
            lrf=0.01,     # 最终学习率
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            box=7.5,      # 边界框损失权重
            cls=0.5,      # 分类损失权重
            hsv_h=0.015,  # 色调增强
            hsv_s=0.7,    # 饱和度增强
            hsv_v=0.4,    # 亮度增强
            degrees=0.0,  # 旋转角度
            translate=0.1, # 平移
            scale=0.5,    # 缩放
            shear=0.0,    # 剪切
            perspective=0.0, # 透视
            flipud=0.0,   # 上下翻转
            fliplr=0.5,   # 左右翻转
            mosaic=1.0,   # 马赛克增强
            mixup=0.0,    # 混合增强
            copy_paste=0.0 # 复制粘贴增强
        )
    finally:
        # 恢复CUDA环境变量
        if device != 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = device if isinstance(device, str) else str(device)
    
    return results

# 验证模型
def validate_model(model_path, data_yaml):
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    return results

# 主函数
def main():
    print("开始准备数据集...")
    data_yaml = create_yolo_dataset()
    
    print("\n开始训练模型...")
    print(f"数据集配置文件: {data_yaml}")
    
    # 训练模型 (可以根据需要调整参数)
    # 模型大小选项: n(nano), s(small), m(medium), l(large), x(xlarge)
    model_size = 'n'  # 使用nano模型，适合入门级硬件
    epochs = 100
    batch_size = 16
    img_size = 640
    
    print(f"使用YOLOv8{model_size}模型训练 {epochs}轮，批次大小={batch_size}，图像尺寸={img_size}")
    results = train_model(
        data_yaml=data_yaml,
        model_size=model_size,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=img_size,
        device='0'  # 使用GPU训练，但AMP检查会在CPU上进行
    )
    
    # 获取最佳模型路径
    best_model = results.best
    
    print(f"\n训练完成! 最佳模型: {best_model}")
    print("\n开始验证模型...")
    
    # 验证最佳模型
    val_results = validate_model(best_model, data_yaml)
    print(f"验证结果: mAP50-95={val_results.box.map:.4f}, mAP50={val_results.box.map50:.4f}")
    
    print("\n模型训练和验证完成!")
    print(f"模型保存在: {best_model}")
    print(f"可以使用以下代码进行推理:\n\nfrom ultralytics import YOLO\nmodel = YOLO('{best_model}')\nresults = model.predict('your_image.jpg')\n")

if __name__ == "__main__":
    main()