import torch
import cv2
import pathlib
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Fix path compatibility for Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Invisibility_cloak/yolov5/best.pt')

# Start webcam capture
cap = cv2.VideoCapture(0)  # Use 0 for default webcam
background_frames = []
num_background_frames = 30  
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Capturing background, stay still for 2 seconds...")
for i in range(num_background_frames):
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    background_frames.append(frame)

background = np.median(background_frames, axis=0).astype(dtype=np.uint8)  
print("Background captured successfully!")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(rgb_frame)
    print("Raw Detections:", results.xyxy[0]) 
    
    detections = results.xyxy[0].cpu().numpy()

    invisibility_frame = frame.copy()

    for x1, y1, x2, y2, conf, cls in detections:
        print(f"Class: {cls}, Confidence: {conf}")  
        if int(cls) == 1 and conf > 0.1:  
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if x2 > x1 and y2 > y1:
                invisibility_frame[y1:y2, x1:x2] = background[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("YOLOv5 Cloak Detection", frame)

    cv2.imshow("Invisibility Effect", invisibility_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
