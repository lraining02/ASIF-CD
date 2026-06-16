from ultralytics import YOLO
# 屏蔽 OpenCV 的输出（0=全部显示, 3=只显示严重错误）
import sys
import os
import torch
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


if __name__ == "__main__":
    # model = YOLO("ultralytics/cfg/models/v11/yolo11.yaml")
    # model = YOLO("ultralytics/cfg/models/fuse/Feature-Fusion.yaml")
    # model = YOLO("ultralytics/cfg/models/fuse/Mid-level-Feature-Fusion.yaml")

    #--------------------------对比实验-----------------
    # model = YOLO("ultralytics/cfg/models/v5/yolov5fuse.yaml")
    # model = YOLO("ultralytics/cfg/models/fuse/DEYOLO.yaml")
    # model = YOLO("/home/liuruining/code/YOLOFUSE/runs/train/M4SAR/fuse(DEYOLO/weights/best.pt")
    # model = YOLO("ultralytics/cfg/models/v5/CMAFF.yaml")
    # model = YOLO("ultralytics/cfg/models/v8/c2DFF.yaml")
    # model = YOLO("ultralytics/cfg/models/v11/MAIENet.yaml")

    # --------------------------our-----------------
    # model = YOLO("ultralytics/cfg/models/ours/caseall.yaml")
    # model = YOLO("ultralytics/cfg/models/ours/casestage.yaml")
    model = YOLO("runs/train/our-M4SAR/without-diss/weights/last.pt")
    # model = YOLO("runs/train/our-OGSOD/320-m/n=2/weights/last.pt")

    # --------------------------消融-----------------
    # model = YOLO("ultralytics/cfg/models/ours/caseall.yaml")
    # model = YOLO("ultralytics/cfg/models/ours/caseFuse&Dis.yaml")
    # model = YOLO("ultralytics/cfg/models/ours/caseTre&Dis.yaml")
    # model = YOLO("ultralytics/cfg/models/ours/case2.yaml")
    # model = YOLO("ultralytics/cfg/models/ours/case3.1.yaml")
    # model = YOLO("ultralytics/cfg/models/ours/case1.1.yaml")

    print(f"Model strides: {model.stride}")
    model.train(
        # data="ultralytics/cfg/datasets/OGSOD.yaml",
        data="ultralytics/cfg/datasets/M4SAR.yaml",
        ch=6,  # 多模态时设置为 6 ，单模态时设置为 3
        task="obb",
        # imgsz=640,
        # imgsz=256,
        imgsz=512,
        # imgsz=320,
        epochs=400,
        batch=16,
        workers=8,
        # project="runs/train/M4SAR",
        # project="runs/train/OGSOD",
        # project="runs/train/our-OGSOD/320-m",、
        project="runs/train/our-M4SAR",
        # project="runs/Ablation_Study",
        # name="fuse-params-thre-0.9",
        # name="fuse(baseline1",
        # name="case2_Experiment-m",
        # name="case3_Experiment",
        # name="case1_Experiment",
        # name="only-fre",
        name="without-diss",
        # name="fuse(DEYOLO",
        # name="fuse(CMAFF",
        # name="fuse(C2DFF",
        # name="n=2",
        # name="fuse-m",

        resume=True,
        # resume=False,
        cos_lr=True,
        save_period=50,
        val=False,
        fraction=1,  # 只用全部数据的 ？% 进行训练 (0.1-1)
    )
