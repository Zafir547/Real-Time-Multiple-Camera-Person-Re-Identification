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

# 📦 Install Required Libraries

pip install PyQt5
pip install opencv-python
pip install opencv-python-headless
pip install ultralytics
pip install torchreid
