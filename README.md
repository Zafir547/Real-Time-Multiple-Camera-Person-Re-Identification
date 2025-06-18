# 🎥 Multi-Camera Real-Time Person Re-Identification

A real-time system to **track individuals across multiple CCTV camera feeds** with consistent IDs, even when appearance changes. Built using **YOLOv8**, **OSNet (ReID)**, **DeepSORT**, and **PyQt5 GUI**.

---

## 📌 Features

- ✅ Real-time object detection using YOLOv8
- ✅ OSNet ReID model for consistent cross-camera ID tracking
- ✅ DeepSORT for tracking
- ✅ PyQt5 GUI for multiple camera streams
- ✅ Edge-computing ready

---

## ⚙️ Installation

**Step 1: Clone the repository**

```bash
git clone https://github.com/your-username/real-time-multicam-reid.git
cd real-time-multicam-reid
```

## 📦 Install Required Libraries
```bash
pip install PyQt5
pip install opencv-python
pip install opencv-python-headless
pip install ultralytics
pip install torchreid
```
## 🔥 PyTorch Installation (Choose One)
💻 For CPU:
```bash
pip install torch torchvision torchaudio
```
⚡ For GPU (CUDA 11.8):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
## 🔧 Additional Setup
```bash
pip install --upgrade pip setuptools wheel
pip install --no-build-isolation git+https://github.com/KaiyangZhou/deep-person-reid.git
```
## ▶️ How to Run
```bash
python main_multicam.py
```
By default, it will use two sample video files:
```bash
video_paths = ["data/6387-191695740_small.mp4", "data/people.mp4"]
```
✅ You can replace these with your own multi-camera feeds or IP camera streams.

## ## 📽️ Output Demo

### 🔗 [Watch the Output Video Here on YouTube](https://www.youtube.com/watch?v=fsNp70_NAgg)

[![Watch the Output Video](https://img.youtube.com/vi/fsNp70_NAgg/hqdefault.jpg)](https://www.youtube.com/watch?v=fsNp70_NAgg)

<img src="[https://example.com/demo-screenshot.png](https://github.com/Zafir547/Real-Time-Multiple-Camera-Person-Re-Identification/blob/main/out_cam1.mp4)" width="100%" />

## 🙏 Special Thanks
Huge appreciation to ITSOLERA Pvt Ltd and the entire development & AI team for their valuable support and guidance in making this project a success.
