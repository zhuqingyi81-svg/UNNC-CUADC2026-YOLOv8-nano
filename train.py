
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

# 把这里的路径，换成你刚刚右键复制的完整路径！注意保留两边的引号！
model.train(data=r'C:\Users\Mercury\OneDrive\Desktop\UNNC CUADC2026\mydata\YOLODataset\dataset.yaml', epochs=100, imgsz=1280，patience=15, batch=16)
