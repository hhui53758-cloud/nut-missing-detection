"""
模型训练脚本
基于 Ultralytics YOLOv11 训练螺母缺失检测模型

使用方法：
    python train.py

训练完成后，权重文件保存于 runs/detect/train*/weights/best.pt
建议将 best.pt 复制到 results/weights/ 目录以便版本管理。
"""

from ultralytics import YOLO


def main():
    # 加载预训练模型（YOLOv11n — nano 版本，速度快，资源占用低）
    # 也可替换为 yolo11s / yolo11m 获得更高精度
    model = YOLO("yolo11n.pt")

    # 开始训练
    model.train(
        data="missing.yaml",          # 数据集配置文件路径
        epochs=100,                   # 训练轮次（可根据数据量调整）
        imgsz=640,                    # 输入图像尺寸
        batch=-1,                    # 自动选择最优 batch size
        cache="ram",                  # 缓存数据到 RAM（加速训练）
        workers=2,                    # 数据加载线程数
        device=0,                    # GPU 设备，设为 'cpu' 则使用 CPU
        project="runs/detect",       # 输出项目目录
        name="train",                # 本次训练名称
        exist_ok=True,               # 允许覆盖同名目录
        pretrained=True,             # 使用预训练权重
        optimizer="auto",            # 自动选择优化器（SGD/Adam/余弦退火）
        verbose=True,                # 打印详细训练信息
        seed=0,                      # 随机种子（可复现）
        deterministic=True,          # 确定性算法（提升可复现性）
        plots=True,                  # 生成训练曲线图
    )

    # 验证集评估
    metrics = model.val()
    print(f"\n验证结果 — mAP50: {metrics.box.map50:.3f}  mAP50-95: {metrics.box.map:.3f}")


if __name__ == "__main__":
    main()
