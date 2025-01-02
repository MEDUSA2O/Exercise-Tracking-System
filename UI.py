import sys
import cv2
import numpy as np
import pyzed.sl as sl
from ultralytics import YOLO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QComboBox, QLabel, 
                           QFileDialog, QMessageBox, QListWidget, QBoxLayout,
                           QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class HandPredictor:
    def __init__(self):
        self.last_positions = []
        self.max_history = 5
        self.velocity = None
        
    def update(self, position):
        if position is not None:
            self.last_positions.append(position)
            if len(self.last_positions) > self.max_history:
                self.last_positions.pop(0)
            
            if len(self.last_positions) >= 2:
                self.velocity = (self.last_positions[-1] - self.last_positions[-2])

    def predict_next_position(self):
        if not self.last_positions:
            return None
            
        if self.velocity is not None:
            predicted_pos = self.last_positions[-1] + self.velocity
            return predicted_pos
        
        return self.last_positions[-1]

    def get_smoothed_position(self):
        if not self.last_positions:
            return None
        return np.mean(self.last_positions, axis=0)

class AspectRatioWidget(QWidget):
    def __init__(self, widget, aspect_ratio=4/3):
        super().__init__()
        self.aspect_ratio = aspect_ratio
        self.widget = widget
        layout = QBoxLayout(QBoxLayout.LeftToRight, self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)

    def resizeEvent(self, e):
        w = e.size().width()
        h = e.size().height()

        if w / h > self.aspect_ratio:  # too wide
            w = int(h * self.aspect_ratio)
        else:  # too tall
            h = int(w / self.aspect_ratio)

        self.widget.setFixedSize(w, h)
        super().resizeEvent(e)

class ExerciseTracker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initVariables()
        self.setupCamera()
        
    def initUI(self):
        self.setWindowTitle('Exercise Tracking System')
        self.setMinimumSize(1200, 800)
        # Set window size policy
        self.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.video_label)

        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Control buttons
        control_layout = QHBoxLayout()
        self.exercise_combo = QComboBox()
        self.exercise_combo.addItems(['深蹲', '臥推'])
        self.load_video_btn = QPushButton('讀影片')
        self.load_camera_btn = QPushButton('讀鏡頭')

        control_layout.addWidget(self.exercise_combo)
        control_layout.addWidget(self.load_video_btn)
        control_layout.addWidget(self.load_camera_btn)

        # Increase font size for labels
        font = self.font()
        font.setPointSize(12)

        self.distance_label = QLabel('Moving Distance: 0.00m')
        self.distance_label.setFont(font)
        self.coord_diff_label = QLabel('Diff - X:0.00m Y:0.00m Z:0.00m')
        self.coord_diff_label.setFont(font)
        self.current_coord_label = QLabel('Current - X:0.00m Y:0.00m Z:0.00m')
        self.current_coord_label.setFont(font)
        self.count_label = QLabel('Count: 0')
        self.count_label.setFont(font)
        self.current_speed_label = QLabel('Current Speed: 0.00 m/s')
        self.current_speed_label.setFont(font)

        # History list
        self.speed_list = QListWidget()
        self.speed_list.setMinimumHeight(200)
        self.speed_list.setFont(font)

        # Overall statistics
        self.overall_speed_label = QLabel('Overall Max/Avg Speed: 0.00/0.00 m/s')
        self.overall_speed_label.setFont(font)

        # New buttons layout - vertical arrangement
        button_layout = QVBoxLayout()
        self.init_btn = QPushButton('初始化')
        self.upload_btn = QPushButton('上傳資料')
        self.end_round_btn = QPushButton('結束該輪')

        # Set font and fixed height for buttons
        for btn in [self.init_btn, self.upload_btn, self.end_round_btn]:
            btn.setFont(font)
            btn.setFixedHeight(50)
            button_layout.addWidget(btn)

        # Add spacing between buttons
        button_layout.setSpacing(15)

        # Add all elements to the right panel
        right_layout.addLayout(control_layout)
        right_layout.addWidget(self.distance_label)
        right_layout.addWidget(self.coord_diff_label)
        right_layout.addWidget(self.current_coord_label)
        right_layout.addWidget(self.count_label)
        right_layout.addWidget(self.current_speed_label)
        right_layout.addWidget(QLabel('History:', font=font))
        right_layout.addWidget(self.speed_list)
        right_layout.addWidget(self.overall_speed_label)
        right_layout.addLayout(button_layout)  # Add new buttons
        right_layout.addStretch()

        # Add panels to main layout
        main_layout.addWidget(left_panel, 2)
        main_layout.addWidget(right_panel, 1)

        # Connect buttons
        self.load_video_btn.clicked.connect(self.load_video)
        self.load_camera_btn.clicked.connect(self.load_camera)
        self.exercise_combo.currentTextChanged.connect(self.on_exercise_changed)
        self.init_btn.clicked.connect(self.initialize_system)
        self.upload_btn.clicked.connect(self.upload_data)
        self.end_round_btn.clicked.connect(self.end_current_round)

        # Setup timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
    def initVariables(self):
        self.zed = None
        self.tracked_roi = None
        self.origin_point = None
        self.origin_point_3D = None
        self.count = 0
        self.lowest_point = None
        self.is_rising = False
        self.movement_started = False
        self.prev_time = None
        self.prev_coord = None
        self.max_speed = 0
        self.total_speed = 0
        self.speed_count = 0
        self.current_max_speed = 0
        self.current_avg_speed = 0
        self.speed_records = []
        self.fixed_roi = None
        self.current_exercise = '深蹲'
        self.last_frame = None  
        self.is_round_ended = False  

        # YOLO and tracking related
        self.model = YOLO('handsv2.pt')
        self.predictor = HandPredictor()
        self.prev_box = None
        self.frames_since_detection = 0
        self.max_frames_to_interpolate = 5
        self.TARGET_HAND = 0  # 0: left hand, 1: right hand for bench press
        self.MAX_MOVEMENT_THRESHOLD = 100
        
    def setupCamera(self):
        self.zed = sl.Camera()
        self.depth_image = sl.Mat()
        self.image_zed = sl.Mat()
        
    def calculate_euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

    def initialize_fixed_roi(self, image_ocv):
        height, width = image_ocv.shape[:2]
        roi_width = int(width * 0.5)
        roi_height = int(height * 1)
        self.fixed_roi = [
            (width - roi_width) // 2,
            (height - roi_height) // 2,
            (width + roi_width) // 2,
            (height + roi_height) // 2
        ]

    def on_exercise_changed(self, exercise):
        self.current_exercise = exercise
        self.TARGET_HAND = 1 if exercise == '深蹲' else 0  # 右手深蹲，左手臥推
        self.resetTracking()
        self.speed_list.clear()

    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "選擇影片檔案", "", 
                                                 "SVO Files (*.svo2);;All Files (*)")
        if file_name:
            self.stop_camera()
            init_params = sl.InitParameters()
            init_params.depth_mode = sl.DEPTH_MODE.ULTRA
            init_params.coordinate_units = sl.UNIT.METER
            init_params.svo_real_time_mode = False
            init_params.set_from_svo_file(file_name)
            
            err = self.zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                QMessageBox.critical(self, "錯誤", f"無法開啟影片檔案: {err}")
                return
            
            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
                image_ocv = self.image_zed.get_data()
                image_ocv = cv2.flip(image_ocv, -1)
                self.initialize_fixed_roi(image_ocv)
                
            self.timer.start(30)
            
    def load_camera(self):
        self.stop_camera()
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_units = sl.UNIT.METER
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 60
        
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            QMessageBox.critical(self, "錯誤", "無法開啟ZED相機")
            return
        
        if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
            image_ocv = self.image_zed.get_data()
            image_ocv = cv2.flip(image_ocv, -1)
            self.initialize_fixed_roi(image_ocv)
            
        self.timer.start(30)
        
    def stop_camera(self):
        self.timer.stop()
        if self.zed is not None:
            self.zed.close()
            self.zed = sl.Camera()
            self.resetTracking()

    def resetTracking(self):
        self.tracked_roi = None
        self.origin_point = None
        self.origin_point_3D = None
        self.count = 0
        self.lowest_point = None
        self.is_rising = False
        self.movement_started = False
        self.max_speed = 0
        self.total_speed = 0
        self.speed_count = 0
        self.current_max_speed = 0
        self.current_avg_speed = 0
        self.speed_records.clear()
        self.fixed_roi = None
        self.prev_box = None
        self.predictor = HandPredictor()
        self.frames_since_detection = 0

    def select_target_hand(self, results, prev_box=None):
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None
        
        boxes = results[0].boxes
        classes = boxes.cls.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        
        target_indices = np.where(classes == self.TARGET_HAND)[0]
        if len(target_indices) == 0:
            return None
        
        if prev_box is None:
            best_idx = target_indices[np.argmax(confidences[target_indices])]
            return xyxy[best_idx]
        else:
            prev_center = np.array([(prev_box[0] + prev_box[2])/2, 
                                   (prev_box[1] + prev_box[3])/2])
            
            min_dist = float('inf')
            best_box = None
            
            for idx in target_indices:
                curr_box = xyxy[idx]
                curr_center = np.array([(curr_box[0] + curr_box[2])/2, 
                                      (curr_box[1] + curr_box[3])/2])
                
                dist = np.sqrt(np.sum((prev_center - curr_center) ** 2))
                
                if dist < self.MAX_MOVEMENT_THRESHOLD and dist < min_dist:
                    min_dist = dist
                    best_box = curr_box
            
            return best_box

    def get_hand_position(self, results):
        box = self.select_target_hand(results, self.prev_box)
        
        if box is not None:
            center = np.array([(box[0] + box[2])/2, (box[1] + box[3])/2])
            return box, center
        return None, None

    def process_frame(self, image_ocv, depth_ocv):
        if self.fixed_roi is None:
            self.initialize_fixed_roi(image_ocv)
            
        cv2.rectangle(image_ocv, 
                    (self.fixed_roi[0], self.fixed_roi[1]), 
                    (self.fixed_roi[2], self.fixed_roi[3]), 
                    (0, 0, 255), 2)

        # YOLO detection
        roi_image = image_ocv[self.fixed_roi[1]:self.fixed_roi[3], 
                            self.fixed_roi[0]:self.fixed_roi[2]]
        results = self.model(roi_image, conf=0.1)

        # Get hand position
        box, center = self.get_hand_position(results)
        
        current_position = None
        using_predicted = False
        
        if box is not None:
            current_position = center + np.array([self.fixed_roi[0], self.fixed_roi[1]])
            self.frames_since_detection = 0
            self.predictor.update(current_position)
            self.prev_box = box
        else:
            self.frames_since_detection += 1
            if self.frames_since_detection <= self.max_frames_to_interpolate:
                linear_pred = self.predictor.predict_next_position()
                if linear_pred is not None:
                    current_position = linear_pred
                    using_predicted = True
        
        if current_position is not None:
            cx, cy = current_position.astype(int)
            
            if box is not None:
                half_width = int((box[2] - box[0]) / 2)
                half_height = int((box[3] - box[1]) / 2)
            else:
                half_width = half_height = 50 if self.prev_box is None else \
                            int((self.prev_box[2] - self.prev_box[0]) / 2)
            
            x1 = max(self.fixed_roi[0], cx - half_width)
            y1 = max(self.fixed_roi[1], cy - half_height)
            x2 = min(self.fixed_roi[2], cx + half_width)
            y2 = min(self.fixed_roi[3], cy + half_height)
            self.tracked_roi = [x1, y1, x2, y2]
            
            color = (255, 0, 0) if not using_predicted else (0, 255, 255)
            status = "Tracking" if not using_predicted else "Predicted"
            cv2.rectangle(image_ocv, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_ocv, f"{status} Hand", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.circle(image_ocv, (cx, cy), radius=5, color=(0, 255, 255), thickness=-1)

            # Process depth and measurements
            cx_flipped = image_ocv.shape[1] - cx
            cy_flipped = image_ocv.shape[0] - cy
            
            err, point3D = self.depth_image.get_value(cx_flipped, cy_flipped)
            
            if err == sl.ERROR_CODE.SUCCESS and not np.isnan(point3D[2]):
                self.process_tracking_data(point3D, cx, cy)

        return image_ocv

    def process_tracking_data(self, point3D, cx, cy):
        if self.origin_point_3D is None:
            self.origin_point_3D = point3D.copy()
            self.origin_point = (cx, cy)
        
        distance = self.calculate_euclidean_distance(point3D, self.origin_point_3D)
        self.distance_label.setText(f"Moving Distance: {distance:.2f}m")
        
        diff_x = point3D[0] - self.origin_point_3D[0]
        diff_y = point3D[1] - self.origin_point_3D[1]
        diff_z = point3D[2] - self.origin_point_3D[2]
        self.coord_diff_label.setText(f"Diff - X:{diff_x:.2f}m Y:{diff_y:.2f}m Z:{diff_z:.2f}m")
        
        self.current_coord_label.setText(
            f"Current - X:{point3D[0]:.2f}m Y:{point3D[1]:.2f}m Z:{point3D[2]:.2f}m")
        
        if self.current_exercise == '深蹲':
            self.process_squat_detection(point3D)
        else:
            self.process_bench_press_detection(point3D)

    def process_squat_detection(self, point3D):
        if self.lowest_point is None or point3D[1] < self.lowest_point:
            self.lowest_point = point3D[1]
            self.is_rising = False
      