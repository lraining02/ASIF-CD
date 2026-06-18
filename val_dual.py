from ultralytics import YOLO
import sys
import os
import cv2

model = YOLO("runs/train/our-OGSOD/weights/best.pt")


# model.task = 'obb'


model.val(
    data="ultralytics/cfg/datasets/OGSOD.yaml",
    # data="ultralytics/cfg/datasets/M4SAR.yaml",
    ch=6,  # 多模态时设置为 6 ，单模态时设置为 3
    batch=16,        # 如果显存不够就调小
    imgsz=512,      # 必须与训练对齐
    conf=0.001,     # 验证时通常设得很低以获取完整的 PR 曲线
    iou=0.6,        # NMS 的阈值
    task="obb",
    project="run/detect/our-OGSOD",
    name="fuse",
    save_json=True  # 建议开启，方便后续分析
)

