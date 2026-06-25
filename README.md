# 🔩 基于深度学习的螺母缺失检测系统

> 利用 YOLOv11 深度学习模型，对工业图像中的**缺螺母（Nut_Missing）** 和**空孔（Empty_Hole）** 进行实时检测与定位。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv11-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📌 项目概述

本项目针对工业制造场景中的螺母检测需求，基于 YOLOv11 构建了一套完整的**目标检测系统**，实现对传送带上螺母状态的自动化质检。

### 检测目标

| 类别 ID | 类别名称 | 说明 |
|:---:|---|---|
| 0 | **Nut_Missing** | 螺母缺失（检测到孔位但无螺母） |
| 1 | **Empty_Hole** | 空孔（未安装任何零件的孔位） |

---

## 🗂️ 项目结构

```
nut-missing-detection/
├── models/
│   ├── train.py              # 模型训练脚本
│   └── predict.py           # 批量推理脚本
├── gui/
│   └── app.py               # GUI 客户端（tkinter）
├── datasets/
│   └── missing.yaml         # 数据集配置文件
├── results/
│   ├── weights/             # 训练权重（best.pt / last.pt）
│   └── metrics/             # 训练曲线、混淆矩阵等
├── docs/
│   └── 用户使用手册.md        # 详细使用文档
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚡ 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
cd models
python train.py
```

> 训练完成后权重保存至 `results/weights/best.pt`

### 3. 启动 GUI 检测工具

```bash
cd gui
python app.py
```

GUI 支持：
- 📁 文件夹批量导入
- 🖼️ 单张图片检测
- 🎬 视频流实时检测
- 💾 检测结果保存

---

## 📊 模型性能

| 指标 | 数值 |
|---|---|
| 训练轮次 | 100 epochs |
| 输入分辨率 | 640 × 640 |
| 基础模型 | YOLOv11n |
| 设备 | GPU（自动检测） |

> 详细训练曲线见 `results/` 目录

---

## 📂 数据集

- **训练集**：164 张图像
- **验证集**：40 张图像
- **测试集**：20 张图像
- **标注格式**：YOLO TXT（类别ID + 归一化坐标）

> ⚠️ 数据集文件不在本仓库中，请参考 `datasets/missing.yaml` 配置路径

---

## 🛠️ 技术栈

- **模型框架**：[Ultralytics YOLOv11](https://docs.ultralytics.com/)
- **GUI**：Tkinter（Python 内置）
- **图像处理**：OpenCV
- **数据标注**：支持 CVAT、LabelImg 等主流标注工具

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源，欢迎 Star ⭐ 和 Fork 🔱

---

## 👤 作者

- **课程**：科研实训
- **指导**：感谢指导老师的帮助与支持

---

> *"质量是制造的生命线，让深度学习为工业质检提速。"*
