#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上位机：按 “/温度/积分时间/000...100.png” 目录结构读取数据，
按“积分时间”选择后：
  1) 计算该积分时间下“中间温度”（或最近可用温度）文件夹内序列的时域均值图像并显示；
  2) 鼠标点击图像上的某个像素或在工具栏输入 X/Y 后“添加像素”，右侧显示该像素随温度变化的曲线。
  3) 支持多点对比、列表管理、右键删除最近、导出 CSV。

依赖：
  pip install PyQt5 matplotlib numpy opencv-python

运行：
  python app_it_temp_curve.py
"""
import os
import re
import sys
import glob
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
import csv
from datetime import datetime
from scipy.ndimage import median_filter

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QComboBox, QSplitter, QStatusBar, QMessageBox, QCheckBox,
    QListWidget, QGroupBox, QSpinBox
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
from matplotlib import font_manager
from PyQt5.QtGui import QFont

HEIGHT = 512
WIDTH = 640
# ------------------------------ 工具函数 ------------------------------
_num_pattern = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)")

def extract_first_number(name: str) -> Optional[float]:
    """从字符串中提取第一个数字，失败返回 None。"""
    m = _num_pattern.search(name)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def natural_key(s: str):
    """自然排序 key，让 2 < 10（针对文件名中的数字）。"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


# ------------------------------ 字体与本地化支持 ------------------------------
def setup_chinese_font():
    """为 Matplotlib 设置可用的中文字体，避免方框（豆腐块）。"""
    candidates = [
        ("NotoSansCJKsc-Regular.otf", "Noto Sans CJK SC"),
        ("SourceHanSansSC-Regular.otf", "Source Han Sans SC"),
        ("SimHei.ttf", "SimHei"),
        ("MSYH.TTC", "Microsoft YaHei"),
        ("PingFang.ttc", "PingFang SC"),
        ("WenQuanYi Zen Hei.ttf", "WenQuanYi Zen Hei"),
    ]
    added_family = None
    try_paths = []
    try:
        base_dir = os.path.dirname(__file__)
        try_paths.append(base_dir)
    except Exception:
        pass
    try_paths.append(os.getcwd())

    for filename, family in candidates:
        for base in try_paths:
            fp = os.path.join(base, filename)
            if os.path.isfile(fp):
                try:
                    font_manager.fontManager.addfont(fp)
                    added_family = family
                    break
                except Exception:
                    pass
        if added_family:
            break

    if added_family:
        matplotlib.rcParams['font.sans-serif'] = [added_family]
    else:
        matplotlib.rcParams['font.sans-serif'] = [fam for _fn, fam in candidates]
    matplotlib.rcParams['axes.unicode_minus'] = False


def set_app_font():
    """为 Qt 界面设置一个中文友好的字体（若系统可用）。"""
    families = [
        "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC",
        "Source Han Sans SC", "WenQuanYi Zen Hei", "SimHei",
    ]
    for fam in families:
        QApplication.setFont(QFont(fam))
        break


# ------------------------------ 数据管理 ------------------------------
class DataManager:
    """
    扫描根目录并组织： 积分时间 -> [(温度值, 温度目录路径, 积分时间目录名)]
    需要时计算并缓存：某温度序列的时域均值图像。
    """
    def __init__(self):
        self.root_dir: Optional[str] = None
        # it_map: key: it_name(str), value: List[(temp_value(float), temp_dir(str), it_dir_name(str))]
        self.it_map: Dict[str, List[Tuple[float, str, str]]] = {}
        # 缓存：key=(it_name, temp_value) -> Optional[np.ndarray(H,W)]
        self.mean_cache: Dict[Tuple[str, float], Optional[np.ndarray]] = {}
        self.read_gray: bool = True  # 以灰度读取

    def set_root(self, root: str):
        if not os.path.isdir(root):
            raise NotADirectoryError(root)
        self.root_dir = root
        self.it_map.clear()
        self.mean_cache.clear()
        self._scan_root()

    def _scan_root(self):
        assert self.root_dir is not None
        # 结构：root/<temp_dir>/<it_dir>/*.png
        temp_dirs = [d for d in glob.glob(os.path.join(self.root_dir, '*')) if os.path.isdir(d)]
        print(f"temp_dirs: {temp_dirs}")
        for tdir in temp_dirs:
            temp_name = os.path.basename(tdir)
            tval = extract_first_number(temp_name)
            if tval is None:
                continue
            vol_dirs = [d for d in glob.glob(os.path.join(tdir, '*')) if os.path.isdir(d)]
            print(f"vol_dirs: {vol_dirs}")
            for vol in vol_dirs:
                it_dirs = [d for d in glob.glob(os.path.join(vol, '*')) if os.path.isdir(d)]
                print(f"it_dirs: {it_dirs}")
                for idir in it_dirs:
                    it_name = os.path.basename(idir)
                    self.it_map.setdefault(it_name, []).append((tval, tdir, it_name))
            # it_dirs = [d for d in glob.glob(os.path.join(tdir, '*')) if os.path.isdir(d)]
            # for idir in it_dirs:
            #     it_name = os.path.basename(idir)
            #     self.it_map.setdefault(it_name, []).append((tval, tdir, it_name))
        print(f"it_maps: {self.it_map}")
        # 对每个 it 的温度列表按温度排序
        for it_name, lst in self.it_map.items():
            lst.sort(key=lambda x: x[0])

    def list_it(self) -> List[str]:
        def k(n):
            v = extract_first_number(n)
            return (0, v) if v is not None else (1, n.lower())
        return sorted(self.it_map.keys(), key=k)

    def temps_for_it(self, it_name: str) -> List[float]:
        return [t for (t, _tdir, _itn) in self.it_map.get(it_name, [])]

    def _list_pngs(self, temp_dir: str, it_name: str) -> List[str]:
        seq_dir = os.path.join(temp_dir, "3", it_name)
        print(f"seq_dir: {seq_dir}")
        if not os.path.isdir(seq_dir):
            return []
        files = [p for p in glob.glob(os.path.join(seq_dir, '*.png')) if os.path.isfile(p)]
        files.sort(key=lambda p: natural_key(os.path.basename(p)))
        return files

    def _read_image(self, path: str) -> np.ndarray:
        flag = cv2.IMREAD_UNCHANGED
        img = cv2.imread(path, flag)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {path}")
        if self.read_gray and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        a2, a1, a0 = self.load_nuc_params(r"G:\JKW_PZ\3_20000\multi_point_calib.npz")
        blind_params = self.load_blind_params(r"G:\JKW_PZ\3_20000\blind_pixels.npz")
        img = self.apply_nuc_correction(img, a2, a1, a0)
        img = self.apply_blind_pixel_detect(img, blind_params)
        return img

    def apply_nuc_correction(self, image, a2_arr, a1_arr, a0_arr, out_max=4095):
        """
        逐像素二次校正:
        corrected = a2 * image^2 + a1 * image + a0
        返回 uint16（截断到 0..out_max）
        """
        image = image.astype(np.float32)
        corrected = a2_arr * (image ** 2) + a1_arr * image + a0_arr
        corrected = np.clip(corrected, 0, out_max)
        # 返回 uint16，保留原量程值（0..out_max）
        return corrected.astype(np.uint16)

    def apply_blind_pixel_detect(self, image, blind_mask, method="median"):
        image = image.astype(np.float32)
        # img1, img2, img3, img4 = split_img(image)
        size = 8
        # median无法直接处理uint16, mean可以
        if method == "median":
            filtered = median_filter(image, size=size)
            # filtered1 = cv2.medianBlur(img1, 11)
            # filtered2 = cv2.medianBlur(img2, 11)
            # filtered3 = cv2.medianBlur(img3, 11)
            # filtered4 = cv2.medianBlur(img4, 11)
        # elif method == "mean":
        #     filtered1 = cv2.blur(img1, (6, 6))
        #     filtered2 = cv2.blur(img2, (6, 6))
        #     filtered3 = cv2.blur(img3, (6, 6))
        #     filtered4 = cv2.blur(img4, (6, 6))
        #     filtered = merge_img(filtered1, filtered2, filtered3, filtered4)
        else:
            raise ValueError("Method must be 'median' or 'mean'")

        # plt.figure(figsize=(8, 6))
        # plt.imshow(filtered, cmap='gray')
        # # plt.title("DOLP")
        # # plt.colorbar()
        # plt.axis('off')

        # 直接替换盲元像素
        compensated = image.copy()
        compensated[blind_mask] = filtered[blind_mask]
        return compensated

    def load_nuc_params(self, nuc_name):
        """从 .npz 加载 a2,a1,a0（必须命名为 a2,a1,a0）"""
        npz_data = np.load(nuc_name)
        a2 = npz_data["a2"]
        a1 = npz_data["a1"]
        a0 = npz_data["a0"]
        if a2.shape != (HEIGHT, WIDTH) or a1.shape != (HEIGHT, WIDTH) or a0.shape != (HEIGHT, WIDTH):
            raise ValueError(f"NUC 参数数组形状需要是 ({HEIGHT},{WIDTH})，当前为 {a2.shape}, {a1.shape}, {a0.shape}")
        # print(f"从 {nuc_name} 加载 NUC 参数 (a2,a1,a0)，形状 {a2.shape}")
        return a2, a1, a0

    def load_blind_params(self, blind_name):
        """从 .npz 加载 a2,a1,a0（必须命名为 a2,a1,a0）"""
        npz_data = np.load(blind_name)
        blind_params = npz_data["blind"].astype(bool)
        if blind_params.shape != (HEIGHT, WIDTH):
            raise ValueError(f"盲元参数数组形状需要是 ({HEIGHT},{WIDTH})，当前为 {blind_params.shape}")
        # print(f"从 {blind_name} 加载盲元参数，形状 {blind_params.shape}")
        return blind_params


    def mean_image(self, it_name: str, temp_val: float) -> Optional[np.ndarray]:
        key = (it_name, temp_val)
        if key in self.mean_cache:
            return self.mean_cache[key]
        # 找到对应温度目录
        pair = None
        for (t, tdir, itn) in self.it_map.get(it_name, []):
            if np.isclose(t, temp_val):
                pair = (tdir, itn)
                break
        if pair is None:
            self.mean_cache[key] = None
            return None
        temp_dir, itn = pair
        files = self._list_pngs(temp_dir, itn)
        if not files:
            self.mean_cache[key] = None
            return None
        # 累加平均
        acc = None
        count = 0
        shape0 = None
        for fp in files:
            img = self._read_image(fp)
            if shape0 is None:
                shape0 = img.shape
                acc = np.zeros(shape0, dtype=np.float64)
            else:
                if img.shape != shape0:
                    # 尺寸不一致，跳过该温度
                    self.mean_cache[key] = None
                    return None
            acc += img.astype(np.float64)
            count += 1
        if count == 0:
            self.mean_cache[key] = None
            return None
        mean = (acc / count).astype(np.float32)
        self.mean_cache[key] = mean
        return mean

    def pixel_curve(self, it_name: str, x: int, y: int) -> Tuple[List[float], List[float]]:
        temps_all = self.temps_for_it(it_name)
        if not temps_all:
            return [], []
        xs: List[float] = []
        ys: List[float] = []
        for t in temps_all:
            img = self.mean_image(it_name, t)
            if img is None:
                continue
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                xs.append(t)
                ys.append(float(img[y, x]))
        return xs, ys


# ------------------------------ Matplotlib Canvas ------------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)


# ------------------------------ 主窗口 ------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("积分时间-温度像素曲线 上位机 (PyQt5)")
        self.resize(1200, 700)

        self.dm = DataManager()

        # 顶部：选择根目录 + 读灰度选项 + 积分时间选择 + 载入按钮
        self.btn_root = QPushButton("选择根目录…")
        self.lbl_root = QLabel("未选择")
        self.lbl_root.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.cb_gray = QCheckBox("以灰度读取"); self.cb_gray.setChecked(True)
        self.combo_it = QComboBox(); self.combo_it.setMinimumWidth(200)
        self.btn_load = QPushButton("载入/刷新")

        top = QHBoxLayout()
        top.addWidget(self.btn_root)
        top.addWidget(self.lbl_root, 1)
        top.addWidget(self.cb_gray)
        top.addWidget(QLabel("积分时间:"))
        top.addWidget(self.combo_it)
        top.addWidget(self.btn_load)

        # 工具栏：多点对比、清空、导出 CSV + 指定像素(x,y)
        self.chk_compare = QCheckBox("多点对比")
        self.btn_clear = QPushButton("清空曲线")
        self.btn_export = QPushButton("导出 CSV")
        self.spin_x = QSpinBox(); self.spin_x.setRange(0, 0); self.spin_x.setPrefix("X:")
        self.spin_y = QSpinBox(); self.spin_y.setRange(0, 0); self.spin_y.setPrefix("Y:")
        self.btn_add_xy = QPushButton("添加像素")
        self.spin_x.setEnabled(False); self.spin_y.setEnabled(False); self.btn_add_xy.setEnabled(False)

        tools = QHBoxLayout()
        tools.addWidget(self.chk_compare)
        tools.addWidget(self.btn_clear)
        tools.addWidget(self.btn_export)
        tools.addSpacing(12)
        tools.addWidget(self.spin_x)
        tools.addWidget(self.spin_y)
        tools.addWidget(self.btn_add_xy)
        tools.addStretch(1)

        # 中间：左侧图像，右侧（曲线 + 列表）
        self.canvas_img = MplCanvas(width=6, height=5, dpi=100)
        self.canvas_plot = MplCanvas(width=6, height=5, dpi=100)
        self.canvas_img.fig.tight_layout(); self.canvas_plot.fig.tight_layout()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.canvas_plot)

        series_box = QGroupBox("已选像素（双击删除）")
        vbox = QVBoxLayout(series_box)
        self.list_series = QListWidget()
        vbox.addWidget(self.list_series)
        btn_row = QHBoxLayout()
        self.btn_del_sel = QPushButton("删除选中")
        self.btn_del_last = QPushButton("删除最近")
        btn_row.addWidget(self.btn_del_sel); btn_row.addWidget(self.btn_del_last)
        vbox.addLayout(btn_row)
        right_layout.addWidget(series_box)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas_img)
        splitter.addWidget(right_panel)
        splitter.setSizes([720, 480])

        # 底部状态栏
        self.status = QStatusBar(); self.status.showMessage("就绪")

        # 总体布局
        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(tools)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.status)
        self.setLayout(layout)

        # 事件绑定
        self.btn_root.clicked.connect(self.on_choose_root)
        self.btn_load.clicked.connect(self.on_load)
        self.combo_it.currentIndexChanged.connect(self.on_it_changed)
        self.cb_gray.stateChanged.connect(self.on_gray_changed)

        self.canvas_img.mpl_connect('button_press_event', self.on_img_canvas_click)
        self.canvas_plot.mpl_connect('button_press_event', self.on_plot_canvas_click)
        self.list_series.itemDoubleClicked.connect(self.on_delete_selected_series)
        self.btn_del_sel.clicked.connect(self.on_delete_selected_series)
        self.btn_del_last.clicked.connect(self.on_delete_last_series)
        self.chk_compare.stateChanged.connect(self.on_compare_mode_changed)
        self.btn_clear.clicked.connect(self.on_clear_curves)
        self.btn_export.clicked.connect(self.on_export_csv)
        self.btn_add_xy.clicked.connect(self.on_add_xy)

        # 状态
        self.current_it: Optional[str] = None
        self.current_mid_temp: Optional[float] = None
        self.current_img: Optional[np.ndarray] = None
        # 多点对比：保存多个曲线条目 {x,y,temps,vals}
        self.series: List[dict] = []

        # 字体
        set_app_font(); setup_chinese_font()

    # -------------------- 事件处理 --------------------
    def on_gray_changed(self, state):
        self.dm.read_gray = (state == Qt.Checked)
        self.dm.mean_cache.clear()
        self.series = []
        self._clear_plot()
        self._refresh_series_list()
        self.status.showMessage("读取模式变更：以灰度读取=%s，缓存与对比曲线已清空" % self.dm.read_gray, 5000)

    def on_choose_root(self):
        d = QFileDialog.getExistingDirectory(self, "选择根目录")
        if not d:
            return
        try:
            self.dm.set_root(d)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"设置根目录失败：\n{e}")
            return
        self.lbl_root.setText(d)
        self.combo_it.clear()
        its = self.dm.list_it()
        if not its:
            QMessageBox.warning(self, "提示", "未在该目录下发现任何 积分时间 子目录。\n结构应为：root/温度/积分时间/帧序列.png")
            return
        self.combo_it.addItems(its)
        self.on_it_changed()

    def on_load(self):
        self.on_it_changed()

    def on_it_changed(self):
        it = self.combo_it.currentText()
        if not it:
            return
        self.series = []
        self._refresh_series_list()
        self.current_it = it
        temps = self.dm.temps_for_it(it)
        if not temps:
            QMessageBox.warning(self, "提示", f"{it} 无可用温度。")
            self._disable_xy_inputs()
            return

        # 找到中位温度或最近有数据的温度
        mid_idx = (len(temps) - 1) // 2
        order = [mid_idx]
        for k in range(1, len(temps)):
            if mid_idx - k >= 0:
                order.append(mid_idx - k)
            if mid_idx + k < len(temps):
                order.append(mid_idx + k)

        chosen_img = None
        chosen_t = None
        for idx in order:
            t = temps[idx]
            img = self.dm.mean_image(it, t)
            if img is not None:
                chosen_img = img
                chosen_t = t
                break

        if chosen_img is None:
            QMessageBox.warning(self, "提示", f"{it} 下所有温度目录均无可用图像，已跳过显示。")
            self.current_img = None
            self.canvas_img.ax.clear(); self.canvas_img.draw()
            self._clear_plot()
            self._disable_xy_inputs()
            self.status.showMessage(f"{it} 无可用数据。")
            return

        self.current_mid_temp = chosen_t
        self.current_img = chosen_img

        # 启用并设置 X/Y 范围
        H, W = chosen_img.shape[:2]
        self.spin_x.blockSignals(True); self.spin_y.blockSignals(True)
        self.spin_x.setRange(0, max(0, W - 1))
        self.spin_y.setRange(0, max(0, H - 1))
        if self.spin_x.value() > max(0, W - 1): self.spin_x.setValue(0)
        if self.spin_y.value() > max(0, H - 1): self.spin_y.setValue(0)
        self.spin_x.blockSignals(False); self.spin_y.blockSignals(False)
        self.spin_x.setEnabled(True); self.spin_y.setEnabled(True); self.btn_add_xy.setEnabled(True)

        tip = "(中间温度无数据，已自动选择最近有数据的温度)" if chosen_t != temps[mid_idx] else ""
        self._show_image(chosen_img, title=f"IT={it} | 温度={chosen_t} {tip}")
        self._clear_plot()
        self.status.showMessage("点击图像选择像素；或在右上输入 X/Y 后点“添加像素”。")

    def _disable_xy_inputs(self):
        self.spin_x.setEnabled(False); self.spin_y.setEnabled(False); self.btn_add_xy.setEnabled(False)

    # ---- 画布与列表交互 ----
    def on_img_canvas_click(self, event):
        # 左键：添加/更新曲线；右键：删除最近曲线
        if event.inaxes != self.canvas_img.ax:
            return
        if self.current_it is None or self.current_img is None:
            return
        if event.button == 3:
            self.on_delete_last_series(); return
        if event.button != 1:
            return
        x = int(round(event.xdata)); y = int(round(event.ydata))
        self._add_xy_internal(x, y)

    def on_plot_canvas_click(self, event):
        # 在右侧曲线图上右键也可删除最近
        if event.inaxes != self.canvas_plot.ax:
            return
        if event.button == 3:
            self.on_delete_last_series()

    def on_add_xy(self):
        if self.current_it is None or self.current_img is None:
            return
        x = int(self.spin_x.value()); y = int(self.spin_y.value())
        self._add_xy_internal(x, y)

    def _add_xy_internal(self, x: int, y: int):
        H, W = self.current_img.shape[:2]
        if x < 0 or y < 0 or x >= W or y >= H:
            QMessageBox.information(self, "提示", f"像素越界: x={x}, y={y}, 合法范围: 0<=x<{W}, 0<=y<{H}")
            return
        try:
            temps, vals = self.dm.pixel_curve(self.current_it, x, y)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"提取像素曲线失败：\n{e}")
            return
        if not temps:
            QMessageBox.information(self, "提示", "该积分时间下没有可用于绘制曲线的数据（相关温度目录为空）。已跳过。")
            self._clear_plot()
            return

        if self.chk_compare.isChecked():
            # 多点对比：如已有相同坐标则更新，否则追加
            replaced = False
            for item in self.series:
                if item['x'] == x and item['y'] == y:
                    item['temps'] = temps; item['vals'] = vals
                    replaced = True; break
            if not replaced:
                self.series.append({'x': x, 'y': y, 'temps': temps, 'vals': vals})
            title = f"IT={self.current_it} 多点对比"
            self.status.showMessage(f"已更新/添加对比点 (x={x}, y={y})。 对比数量={len(self.series)}")
        else:
            # 单点模式：覆盖
            self.series = [{'x': x, 'y': y, 'temps': temps, 'vals': vals}]
            title = f"像素(x={x}, y={y}) 随温度变化 | IT={self.current_it}"
            self.status.showMessage(f"像素(x={x}, y={y}) 曲线已更新。")

        self._plot_series(title=title)
        self._refresh_series_list()
        # 同步 spin 显示
        self.spin_x.setValue(x); self.spin_y.setValue(y)

    def _refresh_series_list(self):
        self.list_series.clear()
        for idx, item in enumerate(self.series):
            self.list_series.addItem(f"{idx+1}. (x={item['x']}, y={item['y']})  点数:{len(item['temps'])}")

    def on_delete_selected_series(self):
        rows = sorted({i.row() for i in self.list_series.selectedIndexes()}, reverse=True)
        if not rows and self.list_series.count() == 1:
            rows = [0]
        if not rows:
            QMessageBox.information(self, "提示", "请先在列表中选择要删除的曲线（或双击某一项）。")
            return
        for r in rows:
            if 0 <= r < len(self.series):
                self.series.pop(r)
        self._plot_series(title=f"IT={self.current_it} 多点对比" if self.chk_compare.isChecked() else "像素随温度曲线")
        self._refresh_series_list()
        self.status.showMessage("已删除选中曲线。", 3000)

    def on_delete_last_series(self):
        if not self.series:
            self.status.showMessage("没有可删除的曲线。", 3000)
            return
        self.series.pop()
        self._plot_series(title=f"IT={self.current_it} 多点对比" if self.chk_compare.isChecked() else "像素随温度曲线")
        self._refresh_series_list()
        self.status.showMessage("已删除最近添加的曲线。", 3000)

    # -------------------- 绘图辅助 --------------------
    def on_compare_mode_changed(self, state):
        mode = "开启" if state == Qt.Checked else "关闭"
        self.status.showMessage(f"多点对比已{mode}。当前曲线数：{len(self.series)}", 3000)

    def on_clear_curves(self):
        self.series = []
        self._clear_plot()
        self._refresh_series_list()
        self.status.showMessage("已清空曲线。", 3000)

    def on_export_csv(self):
        if not self.series:
            QMessageBox.information(self, "提示", "没有可导出的曲线。请先添加至少一个像素。")
            return
        # 构造全体温度集合
        all_temps = set()
        for item in self.series:
            all_temps.update(item['temps'])
        temps_sorted = sorted(all_temps)
        # 为每条曲线构建温度->值映射
        maps = []
        for item in self.series:
            m = {float(t): float(v) for t, v in zip(item['temps'], item['vals'])}
            maps.append((item['x'], item['y'], m))
        # 选择保存路径
        it_tag = self.current_it or "IT"
        default_name = f"pixel_curves_{it_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "保存 CSV", default_name, "CSV Files (*.csv)")
        if not path:
            return
        # 写 CSV：第一列 temperature，后续每列是 x_y
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ["temperature"] + [f"x{x}_y{y}" for x, y, _ in maps]
                writer.writerow(header)
                for t in temps_sorted:
                    row = [t]
                    for x, y, m in maps:
                        row.append(m.get(float(t), ""))  # 缺失值留空
                    writer.writerow(row)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出 CSV 失败：\n{e}")
            return
        self.status.showMessage(f"CSV 已保存：{path}", 5000)

    def _show_image(self, img: np.ndarray, title: str = ""):
        ax = self.canvas_img.ax
        ax.clear()
        ax.imshow(img, cmap='gray', aspect='equal')  # 已移除 colorbar
        ax.set_title(title); ax.set_xlabel('X'); ax.set_ylabel('Y')
        self.canvas_img.draw()

    def _clear_plot(self):
        ax = self.canvas_plot.ax
        ax.clear()
        ax.set_title("像素随温度曲线")
        ax.set_xlabel("温度")
        ax.set_ylabel("像素值（时域均值）")
        self.canvas_plot.draw()

    def _plot_series(self, title: str = ""):
        ax = self.canvas_plot.ax
        ax.clear()
        for item in self.series:
            ax.plot(item['temps'], item['vals'], marker='o', linewidth=1.5,
                    label=f"(x={item['x']}, y={item['y']})")
        ax.set_title(title)
        ax.set_xlabel("温度")
        ax.set_ylabel("像素值（时域均值）")
        if len(self.series) > 1:
            ax.legend(loc='best', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        self.canvas_plot.draw()


# ------------------------------ main ------------------------------
def main():
    app = QApplication(sys.argv)
    set_app_font(); setup_chinese_font()
    w = MainWindow(); w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
