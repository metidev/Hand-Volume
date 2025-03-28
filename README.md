# Smart Hand Volume Controller 🤚🔊

A Python-based application that allows users to control system volume using hand gestures via webcam.

![Demo](demo.gif)

## Features ✨
- **Gesture-Based Control**: Adjust volume by moving thumb and index finger
- **Volume Lock**: Show open palm to lock volume for 3 seconds
- **Perspective-Aware**: Maintains volume level regardless of hand distance from camera
- **Real-Time UI**: Interactive volume bar and status indicators
- **Cross-Platform**: Works on Windows, Linux, and macOS

## Requirements 📋
- Python 3.7+
- Webcam

## Installation ⚙️
```bash
# Clone repository
git clone https://github.com/metidev/hand-volume.git
cd hand-volume
```
# Install dependencies
```
pip install opencv-python mediapipe pycaw comtypes
```
Usage 🚀
```
python main.py
```
✊ Control Volume: Move thumb and index finger closer/farther

🖐️ Lock Volume: Show open palm (all fingers visible)

🟢 Green Bar: Current volume level

🔴 Red Text: Volume lock status with timer

Technical Details 🔧
Computer Vision: MediaPipe Hand Tracking

Audio Control: pycaw library (Windows) / ALSAAudio (Linux)

UI: OpenCV real-time rendering

License 📄
MIT License


