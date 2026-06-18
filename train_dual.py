from ultralytics import YOLO
import sys
import os
import torch
import cv2


if __name__ == "__main__":
    print(f"Model strides: {model.stride}")
    model.train(
        # data="ultralytics/cfg/datasets/OGSOD.yaml",
        data="ultralytics/cfg/datasets/M4SAR.yaml",
        ch=6,  # 多模态时设置为 6 ，单模态时设置为 3
        task="obb",
        imgsz=512,
        epochs=400,
        batch=16,
        workers=8,
        # project="runs/train/M4SAR",

        resume=True,
        # resume=False,
        cos_lr=True,
        save_period=50,
        val=False,
        fraction=1,  # 只用全部数据的 ？% 进行训练 (0.1-1)
    )
