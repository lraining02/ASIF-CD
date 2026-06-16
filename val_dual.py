from ultralytics import YOLO
# 屏蔽 OpenCV 的输出（0=全部显示, 3=只显示严重错误）
import sys
import os
import cv2
# 强行重定向系统的 stderr 到空设备
# f = open(os.devnull, 'w')
# sys.stderr = f
os.environ["OPENCV_LOG_LEVEL"] = "FATAL" # 必须在加载 cv2 之前或非常靠前的位置
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
try:
    import cv2
    if hasattr(cv2, 'utils') and hasattr(cv2.utils, 'logging'):
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except (ImportError, AttributeError):
    pass
# 屏蔽底层 C++ 的 TIFF 警告
import warnings
warnings.filterwarnings("ignore")
# model = YOLO("runs/train/case1_Experiment/weights/best.pt")
# model = YOLO("runs/train/case3_Experiment/weights/best.pt")
# model = YOLO("runs/train/without distill/weights/best.pt")
# model = YOLO("runs/train/fuse-m/weights/best.pt")
# model = YOLO("runs/train/case2_Experiment-m/weights/best.pt")
# model = YOLO("runs/train/case2_Experiment-m/weights/best.pt")
# model = YOLO("runs/train/fuse2-m256x256-cos_lr/weights/best.pt")
# model = YOLO("runs/train/fuse2-m256x256/weights/best.pt")
# model = YOLO("runs/train/our-M4SAR/fuse-m/weights/last.pt")
# model = YOLO("runs/train/M4SAR/fuse(C2DFF/weights/best.pt")
# model = YOLO("runs/train/M4SAR/fuse(MAIENet-m/weights/last.pt")
# model = YOLO("runs/train/OGSOD/fuse(CMAFF/weights/best.pt")
# model = YOLO("runs/train/OGSOD/fuse(C2DFF/weights/best.pt")
# model = YOLO("runs/train/our-OGSOD/320-m/fuse-params-diss-1.0-bs/weights/epoch300.pt")
# model = YOLO("runs/train/our-OGSOD/320-m/fuse-params-thre-0.7/weights/best.pt")
# model = YOLO("runs/train/our-OGSOD/320-m/fuse-stage4and5/weights/best.pt")
model = YOLO("runs/train/our-OGSOD/320-m/only-diss/weights/best.pt")


# model.task = 'obb'



model.val(
    data="ultralytics/cfg/datasets/OGSOD.yaml",
    # data="ultralytics/cfg/datasets/M4SAR.yaml",
    ch=6,  # 多模态时设置为 6 ，单模态时设置为 3
    batch=16,        # 如果显存不够就调小
    imgsz=320,      # 必须与训练对齐
    conf=0.001,     # 验证时通常设得很低以获取完整的 PR 曲线
    iou=0.6,        # NMS 的阈值
    task="obb",
    project="run/detect/our-OGSOD/fps",
    name="fuse(without-fuse",
    save_json=True  # 建议开启，方便后续分析
)

