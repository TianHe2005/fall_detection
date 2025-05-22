import sys
import cv2
from ultralytics import YOLO
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PySide6.QtWidgets import QPushButton, QLabel, QFileDialog
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap

class FallDetectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("跌倒检测系统")
        self.setMinimumSize(800, 600)

        # 初始化变量
        self.model = YOLO('E:\\Soft\\code\\python\\ultralytics-main\\runs\\yolov8n_custom3\\weights\\best.pt')
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 创建主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 创建显示区域
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.display_label)

        # 创建按钮区域
        button_layout = QHBoxLayout()
        
        self.camera_button = QPushButton("打开摄像头")
        self.camera_button.clicked.connect(self.toggle_camera)
        
        self.video_button = QPushButton("选择视频文件")
        self.video_button.clicked.connect(self.open_video)
        
        button_layout.addWidget(self.camera_button)
        button_layout.addWidget(self.video_button)
        layout.addLayout(button_layout)

    def toggle_camera(self):
        if self.timer.isActive():
            self.stop_capture()
            self.camera_button.setText("打开摄像头")
        else:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            if self.cap.isOpened():
                self.timer.start(30)  # 30ms 刷新率
                self.camera_button.setText("关闭摄像头")

    def open_video(self):
        if self.timer.isActive():
            self.stop_capture()
        
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mkv)"
        )
        
        if file_name:
            self.cap = cv2.VideoCapture(file_name)
            if self.cap.isOpened():
                self.timer.start(30)
                self.video_button.setText("停止播放")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # 运行检测
            results = self.model(frame)
            annotated_frame = results[0].plot()

            # 获取检测结果并显示
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f'Class: {cls}, Conf: {conf:.2f}'
                    cv2.putText(annotated_frame, label, (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 转换图像格式并显示
            rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.display_label.setPixmap(scaled_pixmap)
        else:
            self.stop_capture()

    def stop_capture(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.camera_button.setText("打开摄像头")
        self.video_button.setText("选择视频文件")
        self.display_label.clear()

    def closeEvent(self, event):
        self.stop_capture()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FallDetectionWindow()
    window.show()
    sys.exit(app.exec())