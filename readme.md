# 跌倒检测系统

## 项目概述

本项目基于YOLOv8目标检测算法，实现了一个实时跌倒检测系统。系统包含两个主要部分：

1. **模型训练模块**：使用Ultralytics库训练自定义数据集，实现跌倒行为的检测
2. **图形用户界面**：基于PySide6开发的用户友好界面，支持摄像头实时检测和视频文件分析

## 模型训练过程

### 数据集准备

训练脚本`train_model.py`实现了从VOC格式数据集到YOLO格式的自动转换：

1. 读取原始数据集中的类别列表（`label_list.txt`）
2. 处理训练集和验证集图像及标注文件
3. 将VOC XML格式的标注转换为YOLO格式（归一化的中心点坐标和宽高）
4. 生成YOLO格式的数据集配置文件（YAML）

### 训练配置

模型训练使用以下配置：

- **模型**：YOLOv8n（nano版本，适合入门级硬件）
- **训练轮数**：100轮
- **批次大小**：16
- **图像尺寸**：640×640
- **优化器**：Adam
- **学习率**：初始0.01，最终0.01
- **早停策略**：patience=50
- **数据增强**：
  - 色调增强(hsv_h=0.015)
  - 饱和度增强(hsv_s=0.7)
  - 亮度增强(hsv_v=0.4)
  - 平移(translate=0.1)
  - 缩放(scale=0.5)
  - 左右翻转(fliplr=0.5)
  - 马赛克增强(mosaic=1.0)

### 训练结果

训练模型`yolov8n_custom3`的主要性能指标：

- **mAP50**：0.767（IoU阈值为0.5时的平均精度）
- **mAP50-95**：0.512（IoU阈值从0.5到0.95的平均精度）
- **精确率**：0.749（Precision）
- **召回率**：0.722（Recall）

训练过程中，模型性能稳步提升，在第99轮达到最佳性能。训练总时长约6127秒（约102分钟）。

## 图形用户界面

`fall_detection_gui.py`实现了一个基于PySide6的图形用户界面，主要功能包括：

### 界面组件

- 视频显示区域：显示检测结果
- 控制按钮：
  - 打开/关闭摄像头
  - 选择/停止播放视频文件

### 功能特点

1. **实时摄像头检测**：
   - 支持打开本地摄像头进行实时跌倒检测
   - 设置640×480分辨率，30ms刷新率

2. **视频文件分析**：
   - 支持选择本地视频文件（mp4、avi、mkv格式）
   - 对视频内容进行跌倒行为检测

3. **检测结果可视化**：
   - 在视频帧上绘制检测框
   - 显示类别和置信度信息
   - 保持图像比例适应显示区域

## 使用指南

### 模型训练

1. 准备数据集，确保数据集结构符合要求：
   ```
   dataset/
     ├── Annotations/       # XML格式标注
     ├── JPEGImages/        # 图像文件
     ├── label_list.txt     # 类别列表
     ├── train.txt          # 训练集列表
     └── val.txt           # 验证集列表
   ```

2. 运行训练脚本：
   ```bash
   python train_model.py
   ```

3. 训练完成后，最佳模型将保存在`runs/yolov8n_custom*/weights/best.pt`

### 运行检测应用

1. 确保已安装所需依赖：
   ```bash
   pip install ultralytics opencv-python PySide6
   ```

2. 运行GUI应用：
   ```bash
   python fall_detection_gui.py
   ```

3. 在界面中：
   - 点击「打开摄像头」进行实时检测
   - 点击「选择视频文件」分析本地视频

## 模型评估与可视化

训练过程生成了多种评估图表，保存在`runs/yolov8n_custom3/`目录：

- **混淆矩阵**：显示各类别的预测准确性
- **PR曲线**：精确率-召回率曲线
- **F1曲线**：不同置信度阈值下的F1分数
- **训练批次可视化**：展示训练过程中的检测效果

## YOLOv8n_custom3 模型可视化

以下是YOLOv8n_custom3模型的可视化结果：

![训练损失曲线](./yolov8n_custom3/results.png)

![混淆矩阵](./yolov8n_custom3/confusion_matrix.png)

![PR曲线](./yolov8n_custom3/PR_curve.png)

![F1曲线](./yolov8n_custom3/F1_curve.png)

- **训练批次可视化**：展示训练过程中的检测效果

## 未来改进方向

1. 增加更多类别，如站立、行走、坐下等动作识别
2. 优化模型性能，尝试更大的模型（YOLOv8s/m）提高准确率
3. 添加报警功能，当检测到跌倒时发出提醒
4. 支持多摄像头同时检测
5. 添加历史记录和数据统计功能

## 参考资料

- [Ultralytics YOLOv8文档](https://docs.ultralytics.com/)
- [PySide6官方文档](https://doc.qt.io/qtforpython-6/)
- [OpenCV-Python教程](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
