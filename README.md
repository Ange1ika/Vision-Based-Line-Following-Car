# Vision-Based Line Following Car

Autonomous Raspberry Pi car that follows a black line using camera vision,  
real-time image processing, and adaptive maneuver control.  
The system uses **OpenCV** for segmentation and **L298N** motor driver for movement.  
Telemetry (speeds, states, and centroids) is logged in CSV format for further analysis.

---

##  Features

- 🎥 **Line Detection** using HSV thresholding and morphological operations  
- 🧭 **Angle Recognition** for 90° left/right turns  
- ⚙️ **Smooth Steering** with proportional (PID-like) correction  
- 🔄 **Autonomous Maneuvers** for sharp corners  
- 📊 **Telemetry Logging** (CSV logs with timestamps, speeds, and states)  
- 🧩 **Debug Visualization** (upper/lower centroids, ROI grid, maneuvers)

---

## 🗂 Project Structure
```bash
line_follower/
├── line_detector.py       # Line segmentation & ROI analysis
├── angle_analyzer.py      # Turn detection and confidence logic
├── motor_controller.py    # L298N GPIO-based motor control
├── vision_controller.py   # Core control loop, steering & telemetry
├── telemetry_log.csv      # Example recorded telemetry
└── README.md
```
---

## ⚙️ Requirements

- Python 3.9+  
- OpenCV 4.x  
- NumPy  
- RPi.GPIO (on Raspberry Pi)

Install with:
```bash
pip install opencv-python numpy RPi.GPIO
```

▶️ Run the System

On Raspberry Pi:
```bash
python3 vision_controller.py --debug
```

or (for test images):
```bash
python3 vision_controller.py
```
