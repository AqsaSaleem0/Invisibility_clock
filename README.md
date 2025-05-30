#  Illusions of Invisibility: Dynamic Cloak Detection

Real-time invisibility effect using deep learning! This project detects a specially trained cloak using YOLOv5 and replaces its region with a pre-captured background to create an "invisible" effect — like magic, but powered by AI.


##  Project Overview

Traditional invisibility cloak projects rely on color detection, but this project uses **object detection (YOLOv5)** to dynamically identify a cloak regardless of lighting or camera quality.

Once detected, the cloak region is replaced with the static background, creating the illusion of invisibility in real time.

---

##  Project Structure

yolov5 the model trained on custom dataset
invisibility_effect.py    the core file that creates the invisibility effect


##  How It Works

1. *Model Training:** Train YOLOv5 on annotated data. 
2. **Background Capture:** Capture background frame without the subject.
3. **Real-Time Detection:** Run webcam feed, detect cloak, and overlay background in that region.


## Requirements

1. Python 3.8+
2. OpenCV
3. PyTorch
4. YOLOv5 dependencies (from Ultralytics repo)
