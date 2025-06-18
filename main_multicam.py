import sys
import cv2
import torch
import time
import random
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from ultralytics import YOLO
from tracker import Tracker

class MultiCamReID(QWidget):
    def __init__(self, video_paths):
        super().__init__()
        self.setWindowTitle("Real-Time Multi-Camera ReID Viewer (Global IDs)")
        self.resize(1400, 800)

        # Video sources
        self.caps = [cv2.VideoCapture(path) for path in video_paths]
        self.frame_width = int(self.caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Single global tracker for both cameras
        self.global_tracker = Tracker()

        # Output writers (optional)
        self.writers = []
        fps = self.caps[0].get(cv2.CAP_PROP_FPS)
        for i in range(len(self.caps)):
            writer = cv2.VideoWriter(
                f"out_cam{i+1}.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (self.frame_width, self.frame_height)
            )
            self.writers.append(writer)

        # PyQt5 Labels
        self.labels = [QLabel(f"Camera {i+1}") for i in range(len(self.caps))]
        layout = QHBoxLayout()
        for label in self.labels:
            layout.addWidget(label)
        self.setLayout(layout)

        # YOLOv8 model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO("yolov8n.pt").to(self.device)
        if self.device == 'cuda':
            self.model.fuse()

        self.colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(500)]
        self.frame_id = 0
        self.CONF_THRESHOLD = 0.5

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(30)

    def update_frames(self):
        frames = []  # hold frames for each camera
        for cap in self.caps:
            ret, frame = cap.read()
            if not ret:
                frames.append(None)
            else:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                frames.append(frame)

        # Combined processing: detections and tracking
        for idx, frame in enumerate(frames):
            if frame is None:
                continue
            self.frame_id += 1
            start = time.perf_counter()

            # Detect
            results = self.model.predict(frame, device=self.device, stream=False, conf=self.CONF_THRESHOLD)
            detections = []
            for result in results:
                for r in result.boxes.data.tolist():
                    x1,y1,x2,y2,score,cls = r
                    if score >= self.CONF_THRESHOLD:
                        detections.append([int(x1),int(y1),int(x2),int(y2), score])

            # Update global tracker
            self.global_tracker.update(frame, detections)

            # Draw global tracks on this frame
            for track in self.global_tracker.tracks:
                x1,y1,x2,y2 = map(int, track.bbox)
                tid = track.track_id
                color = self.colors[tid % len(self.colors)]
                # Only draw if this bbox lies in current frame bounds
                if x2 <= self.frame_width and y2 <= self.frame_height:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(frame, f'ID: {tid}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.putText(frame, f'Cam {idx+1} | Active IDs: {len(self.global_tracker.tracks)}',
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            fps_val = 1/(time.perf_counter()-start)
            print(f"Cam {idx+1} | Frame {self.frame_id} | FPS: {fps_val:.2f}")

            # Write and display
            self.writers[idx].write(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
            self.labels[idx].setPixmap(QPixmap.fromImage(img))

    def closeEvent(self, event):
        for cap in self.caps:
            cap.release()
        for w in self.writers:
            w.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    paths = ["data/6387-191695740_small.mp4", "data/people.mp4"]
    gui = MultiCamReID(paths)
    gui.show()
    sys.exit(app.exec_())