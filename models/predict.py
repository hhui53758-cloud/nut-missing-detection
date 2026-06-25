"""
批量推理脚本
使用训练好的模型对图像/视频进行批量目标检测

使用方法：
    # 修改 MODEL_PATH 为你的权重路径
    python predict.py
"""

import os
from ultralytics import YOLO

# ====================== 配置区（修改这里） ======================
MODEL_PATH = "weights/best.pt"       # 模型权重路径
SOURCE_DIR = "datasets/missing/images/val"  # 待检测素材目录
OUTPUT_DIR = "runs/predict_output"  # 输出目录
# =============================================================


def main():
    # 加载模型
    model = YOLO(MODEL_PATH)

    # 批量推理
    results = model.predict(
        source=SOURCE_DIR,
        save=True,            # 保存标注图
        save_txt=True,        # 保存 YOLO 格式标签
        save_conf=True,       # 标签中包含置信度
        imgsz=640,
        conf=0.3,             # 置信度阈值
        iou=0.45,             # NMS IoU 阈值
        project=OUTPUT_DIR,
        name="images",
        exist_ok=True,
        verbose=True,
    )

    print(f"\n✅ 推理完成！结果保存至：{OUTPUT_DIR}/images/")


if __name__ == "__main__":
    main()
