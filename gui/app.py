"""
螺母缺失检测系统 - GUI 客户端
基于 Tkinter + YOLOv11 的工业质检工具

使用方法：
    python app.py

功能：
    ✅ 图片/视频/文件夹批量检测
    ✅ 实时播放控制（上一张/下一张）
    ✅ 视频进度条拖动
    ✅ 检测结果统计（Nut_Missing / Empty_Hole）
    ✅ 结果保存
"""

import ctypes
import sys
import threading
import os
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# ====================== 配置区（部署前请修改） ======================
DEFAULT_MODEL_PATH = "weights/best.pt"   # 默认模型路径（相对路径）
DEFAULT_CONFIDENCE = 0.3                 # 置信度阈值
DEFAULT_IOU = 0.45                       # NMS IoU 阈值
DEFAULT_IMGSZ = 416                      # 推理分辨率（越小越快）
# ===================================================================

# 解决 Windows 界面模糊
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()


class NutDetectionSystem:
    """螺母缺失智能检测系统 GUI"""

    def __init__(self, root):
        self.root = root
        self.root.title("螺母缺失智能检测系统")
        self.root.geometry("1400x950")
        self.root.minsize(1280, 850)
        self.root.configure(bg="#f8fafc")

        # 动态加载模型（首次选择素材时加载）
        self.model = None
        self.model_path = None

        # 播放控制
        self.cap = None
        self.running = False
        self.video_fps = 30
        self.video_total_frames = 0
        self.seeking = False
        self.skip_frame = 2  # 跳帧加速

        # 素材列表
        self.file_list = []
        self.current_index = -1
        self.valid_suffix = (".png", ".jpg", ".jpeg", ".bmp", ".mp4", ".avi", ".mkv", ".mov")

        self.current_frame = None
        self.display_max_width = 1000
        self.display_max_height = 600

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_window_close)

    # ──────────────── 配置方法 ────────────────
    def load_model(self, model_path=None):
        """懒加载模型，避免启动时卡顿"""
        if model_path is None:
            model_path = filedialog.askopenfilename(
                title="选择模型权重文件",
                filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
            )
        if not model_path or not os.path.exists(model_path):
            messagebox.showwarning("未找到模型", "请选择有效的模型权重文件（.pt）")
            return False

        try:
            from ultralytics import YOLO
            self.model_path = model_path
            self.model = YOLO(self.model_path)
            self.model.fuse()  # 融合层，加速推理
            print(f"[INFO] 模型加载成功：{self.model_path}")
            return True
        except Exception as e:
            messagebox.showerror("模型加载失败", str(e))
            return False

    # ──────────────── UI 初始化 ────────────────
    def setup_ui(self):
        # 标题栏
        title_frame = tk.Frame(self.root, bg="#1e293b", height=60)
        title_frame.pack(fill=tk.X)
        tk.Label(
            title_frame, text="🔩 螺母缺失智能检测系统",
            font=("微软雅黑", 20, "bold"), bg="#1e293b", fg="white"
        ).pack(pady=12)

        # 主体
        main_frame = tk.Frame(self.root, bg="#f8fafc")
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        # 左右导航按钮
        self.btn_prev = tk.Button(
            main_frame, text="◀ 上一个", font=("微软雅黑", 14, "bold"),
            bg="#94a3b8", fg="white", width=8, height=3, bd=0,
            state=tk.DISABLED, cursor="hand2"
        )
        self.btn_prev.pack(side=tk.LEFT, padx=10)
        self.btn_prev.bind("<ButtonPress-1>", self.prev_file)

        # 显示区
        display_container = tk.Frame(main_frame, bg="#f8fafc")
        display_container.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.display_frame = tk.Frame(display_container, bg="#e2e8f0", bd=1, relief=tk.SOLID)
        self.display_frame.pack(expand=True, fill=tk.BOTH)
        self.display_label = tk.Label(
            self.display_frame,
            text="请选择图片/视频/文件夹开始检测\n\n💡 支持格式：PNG / JPG / MP4 / AVI / MKV / MOV",
            font=("微软雅黑", 14), bg="#e2e8f0", fg="#64748b", justify=tk.CENTER
        )
        self.display_label.pack(expand=True, padx=2, pady=2)

        # 进度条
        self.progress_bar = tk.Scale(
            display_container, from_=0, to=100, orient=tk.HORIZONTAL,
            showvalue=0, bg="#f8fafc", troughcolor="#cbd5e1", fg="#3b82f6",
            state=tk.DISABLED
        )
        self.progress_bar.pack(fill=tk.X, pady=(10, 0))
        self.progress_bar.bind("<ButtonPress-1>", self.on_seek_start)
        self.progress_bar.bind("<ButtonRelease-1>", self.on_seek_end)

        self.count_label = tk.Label(
            display_container, text="无素材",
            font=("微软雅黑", 10), bg="#f8fafc", fg="#64748b"
        )
        self.count_label.pack(pady=5)

        # 右导航
        self.btn_next = tk.Button(
            main_frame, text="下一个 ▶", font=("微软雅黑", 14, "bold"),
            bg="#94a3b8", fg="white", width=8, height=3, bd=0,
            state=tk.DISABLED, cursor="hand2"
        )
        self.btn_next.pack(side=tk.LEFT, padx=10)
        self.btn_next.bind("<ButtonPress-1>", self.next_file)

        # 统计栏
        stats_frame = tk.Frame(self.root, bg="white", bd=0, relief=tk.FLAT)
        stats_frame.pack(fill=tk.X, padx=40, pady=10, ipady=15)

        self.empty_label = tk.Label(
            stats_frame, text="Empty_Hole（空孔）：0 个",
            font=("微软雅黑", 14, "bold"), bg="white", fg="#0ea5e9"
        )
        self.empty_label.pack(side=tk.LEFT, expand=True)

        self.nut_label = tk.Label(
            stats_frame, text="Nut_Missing（缺螺母）：0 个",
            font=("微软雅黑", 14, "bold"), bg="white", fg="#f59e0b"
        )
        self.nut_label.pack(side=tk.LEFT, expand=True)

        self.total_label = tk.Label(
            stats_frame, text="总计故障：0 个",
            font=("微软雅黑", 14, "bold"), bg="white", fg="#ef4444"
        )
        self.total_label.pack(side=tk.LEFT, expand=True)

        # 按钮栏
        btn_frame = tk.Frame(self.root, bg="#f8fafc")
        btn_frame.pack(side=tk.BOTTOM, pady=20)

        btn_cfg = {"font": ("微软雅黑", 12, "bold"), "width": 14,
                   "height": 2, "bd": 0, "relief": tk.FLAT, "fg": "white", "cursor": "hand2"}

        def make_btn(text, bg, active, cmd):
            b = tk.Button(btn_frame, text=text, bg=bg, activebackground=active, **btn_cfg)
            b.pack(side=tk.LEFT, padx=10)
            b.bind("<Enter>", lambda e: b.config(bg=active))
            b.bind("<Leave>", lambda e: b.config(bg=bg))
            b.config(command=cmd)
            return b

        make_btn("选择文件夹", "#8b5cf6", "#7c3aed", self.open_folder)
        make_btn("选择图片", "#10b981", "#059669", self.open_image)
        make_btn("选择视频", "#3b82f6", "#2563eb", self.open_video)
        make_btn("停止播放", "#ef4444", "#dc2626", self.stop_detection)
        make_btn("保存结果", "#f59e0b", "#d97706", self.save_result)
        make_btn("一键刷新", "#64748b", "#475569", self.refresh_all)
        make_btn("加载模型", "#6366f1", "#4f46e5", lambda: self.load_model())

    # ──────────────── 推理核心 ────────────────
    def infer(self, frame):
        """执行 YOLO 推理，返回（标注帧, 空孔数, 缺螺母数）"""
        if self.model is None:
            if not self.load_model():
                return frame, 0, 0

        res = self.model(
            frame,
            imgsz=DEFAULT_IMGSZ,
            conf=DEFAULT_CONFIDENCE,
            iou=DEFAULT_IOU,
            augment=False,
            half=True
        )
        annotated = res[0].plot()

        count_empty = sum(1 for b in res[0].boxes if int(b.cls[0]) == 0)
        count_nut = sum(1 for b in res[0].boxes if int(b.cls[0]) == 1)

        return annotated, count_empty, count_nut

    # ──────────────── 视频循环 ────────────────
    def video_loop(self):
        frame_interval = max(1, int(1000 / self.video_fps))
        frame_count = 0

        while self.running and self.cap is not None and self.cap.isOpened():
            if self.seeking:
                cv2.waitKey(10)
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % self.skip_frame != 0:
                continue

            annotated, ce, cn = self.infer(frame)
            self.root.after(0, lambda f=annotated, c=ce, n=cn: self.update_display(f, c, n))
            self.root.after(0, lambda: self.progress_bar.set(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
            cv2.waitKey(frame_interval)

        self.stop_detection()

    def update_display(self, frame, count_empty, count_nut):
        self.current_frame = frame
        self.empty_label.config(text=f"Empty_Hole（空孔）：{count_empty} 个")
        self.nut_label.config(text=f"Nut_Missing（缺螺母）：{count_nut} 个")
        self.total_label.config(text=f"总计故障：{count_empty + count_nut} 个")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        w, h = img.size
        scale = min(self.display_max_width / w, self.display_max_height / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)

        try:
            img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        except Exception:
            img = img.resize((nw, nh), Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(img)
        self.display_label.config(image=imgtk, text="")
        self.display_label.imgtk = imgtk

    # ──────────────── 文件操作 ────────────────
    def _load_file_list(self, path):
        """扫描目录或单文件"""
        if os.path.isfile(path):
            return [path]
        files = []
        for root, _, fs in os.walk(path):
            for f in fs:
                if f.lower().endswith(self.valid_suffix):
                    files.append(os.path.join(root, f))
        files.sort()
        return files

    def _ensure_model(self):
        if self.model is None:
            # 优先尝试默认路径
            if os.path.exists(DEFAULT_MODEL_PATH):
                self.load_model(DEFAULT_MODEL_PATH)
            else:
                self.load_model()
                if self.model is None:
                    return False
        return True

    def _load_current(self):
        if self.current_index < 0 or self.current_index >= len(self.file_list):
            return
        self.reset_stats()
        fp = self.file_list[self.current_index]

        if not fp.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
            # 图片推理
            self.progress_bar.config(state=tk.DISABLED)
            img = cv2.imread(fp)
            if img is None:
                return
            annotated, ce, cn = self.infer(img)
            self.update_display(annotated, ce, cn)
        else:
            # 视频推理
            self.cap = cv2.VideoCapture(fp)
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.video_total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.progress_bar.config(state=tk.NORMAL, to=int(self.video_total_frames))
            self.progress_bar.set(0)
            self.running = True
            threading.Thread(target=self.video_loop, daemon=True).start()

    def open_folder(self):
        self.stop_detection()
        dp = filedialog.askdirectory(title="选择素材文件夹")
        if not dp:
            return
        self.file_list = self._load_file_list(dp)
        self._open_list()

    def open_image(self):
        self.stop_detection()
        fp = filedialog.askopenfilename(
            title="选择图片", filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp")]
        )
        if fp:
            self.file_list = [fp]
            self._open_list()

    def open_video(self):
        self.stop_detection()
        fp = filedialog.askopenfilename(
            title="选择视频", filetypes=[("视频文件", "*.mp4 *.avi *.mkv *.mov")]
        )
        if fp:
            self.file_list = [fp]
            self._open_list()

    def _open_list(self):
        if not self.file_list:
            return
        self.current_index = 0
        total = len(self.file_list)
        self.count_label.config(text=f"1/{total}")
        self.btn_prev.config(state=tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if total > 1 else tk.DISABLED)
        self._load_current()

    def prev_file(self, _):
        if self.current_index > 0:
            self.stop_detection()
            self.current_index -= 1
            self._update_nav_and_load()

    def next_file(self, _):
        if self.current_index < len(self.file_list) - 1:
            self.stop_detection()
            self.current_index += 1
            self._update_nav_and_load()

    def _update_nav_and_load(self):
        total = len(self.file_list)
        now = self.current_index + 1
        self.count_label.config(text=f"{now}/{total}")
        self.btn_prev.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if self.current_index < total - 1 else tk.DISABLED)
        self._load_current()

    def on_seek_start(self, _):
        self.seeking = True

    def on_seek_end(self, _):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.progress_bar.get()))
        self.seeking = False

    def stop_detection(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def reset_stats(self):
        self.empty_label.config(text="Empty_Hole（空孔）：0 个")
        self.nut_label.config(text="Nut_Missing（缺螺母）：0 个")
        self.total_label.config(text="总计故障：0 个")

    def save_result(self):
        if self.current_frame is None:
            messagebox.showwarning("无内容", "当前无检测结果可保存")
            return
        fp = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG 图片", "*.png"), ("JPG 图片", "*.jpg")]
        )
        if fp:
            cv2.imwrite(fp, self.current_frame)
            messagebox.showinfo("成功", f"已保存至：\n{fp}")

    def refresh_all(self):
        self.stop_detection()
        self.file_list = []
        self.current_index = -1
        self.display_label.config(image="", text=(
            "请选择图片/视频/文件夹开始检测\n\n"
            "💡 支持格式：PNG / JPG / MP4 / AVI / MKV / MOV"
        ))
        self.display_label.imgtk = None
        self.current_frame = None
        self.reset_stats()
        self.progress_bar.config(state=tk.DISABLED, to=100)
        self.progress_bar.set(0)
        self.count_label.config(text="无素材")
        self.btn_prev.config(state=tk.DISABLED)
        self.btn_next.config(state=tk.DISABLED)

    def on_window_close(self):
        self.stop_detection()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = NutDetectionSystem(root)
    root.mainloop()
