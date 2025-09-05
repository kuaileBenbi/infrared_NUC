#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
上位机 2：按 “/积分时间/*.npz” 目录结构读取校正文件（npz 字典，兼容不同 key）。
功能：
  • 选择根目录 → 自动发现所有“积分时间”与其中的 .npz 文件
  • 选择 “积分时间 / .npz / key（数组名）/ 通道(可选)” → 左侧显示该数组的 2D 系数图
  • 点击图像像素：右侧进行“不同像素的系数对比”
      - 单点模式：仅显示最近一次点击的像素值
      - 多点对比：勾选后可叠加多个像素（列表管理、右键删除最近/双击删除）
  • 导出 CSV：导出当前 key(+通道) 下所选像素的系数值

依赖：
  pip install PyQt5 matplotlib numpy

运行：
  python app_coeff_vis.py
"""
import os
import re
import sys
import glob
import csv
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QComboBox, QSplitter, QStatusBar, QMessageBox, QCheckBox,
    QListWidget, QGroupBox, QSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
from matplotlib import font_manager

# ------------------------------ 字体与本地化支持 ------------------------------
_num_pattern = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)")

def extract_first_number(name: str) -> Optional[float]:
    m = _num_pattern.search(name)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]

def setup_chinese_font():
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
    families = [
        "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC",
        "Source Han Sans SC", "WenQuanYi Zen Hei", "SimHei",
    ]
    for fam in families:
        QApplication.setFont(QFont(fam))
        break

# ------------------------------ Matplotlib Canvas ------------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

# ------------------------------ 数据管理 ------------------------------
class DataManager:
    """扫描目录并按 积分时间 -> [npz 文件路径] 建立索引，
    并按需加载 .npz 为 dict。兼容不同 key（只筛选适合可视化的数组）。"""
    def __init__(self):
        self.root_dir: Optional[str] = None
        # it_map: key: it_name(str) -> List[npz_path]
        self.it_map: Dict[str, List[str]] = {}
        # 缓存已加载的 npz 内容： (it_name, npz_path) -> dict(key->array)
        self.cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def set_root(self, root: str):
        if not os.path.isdir(root):
            raise NotADirectoryError(root)
        self.root_dir = root
        self.it_map.clear()
        self.cache.clear()
        self._scan_root()

    def _scan_root(self):
        assert self.root_dir is not None
        it_dirs = [d for d in glob.glob(os.path.join(self.root_dir, '*')) if os.path.isdir(d)]
        for idir in it_dirs:
            it_name = os.path.basename(idir)
            files = [p for p in glob.glob(os.path.join(idir, '*.npz')) if os.path.isfile(p)]
            files.sort(key=lambda p: natural_key(os.path.basename(p)))
            if files:
                self.it_map.setdefault(it_name, []).extend(files)
        # 对 it 按名称中的数字排序
        def it_sort_key(n):
            v = extract_first_number(n)
            return (0, v) if v is not None else (1, n.lower())
        self.it_map = dict(sorted(self.it_map.items(), key=lambda kv: it_sort_key(kv[0])))

    def list_it(self) -> List[str]:
        return list(self.it_map.keys())

    def list_npz(self, it_name: str) -> List[str]:
        return self.it_map.get(it_name, [])

    def load_npz(self, it_name: str, npz_path: str) -> Dict[str, Any]:
        key = (it_name, npz_path)
        if key in self.cache:
            return self.cache[key]
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            raise IOError(f"读取 npz 失败: {npz_path}\n{e}")
        d: Dict[str, Any] = {}
        for k in data.files:
            arr = data[k]
            d[k] = arr
        self.cache[key] = d
        return d

    @staticmethod
    def as_image(arr: np.ndarray, channel: Optional[int] = None) -> np.ndarray:
        """将不同形状/类型的数组转为 2D float32 图像。
        支持：(H,W)；(H,W,C) 需指定 channel；布尔/整型会转为 float32。
        """
        if arr is None:
            raise ValueError("空数组")
        if arr.ndim == 2:
            img = arr.astype(np.float32)
            return img
        if arr.ndim == 3:
            H, W, C = arr.shape
            if channel is None:
                channel = 0
            if not (0 <= channel < C):
                raise IndexError(f"通道越界: channel={channel}, C={C}")
            img = arr[..., int(channel)].astype(np.float32)
            return img
        raise ValueError(f"不支持的数组维度: {arr.shape}")

# ------------------------------ 主窗口 ------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("积分时间-校正系数可视化 上位机 (PyQt5)")
        self.resize(1280, 760)

        self.dm = DataManager()

        # 顶部：选择根目录 + 选择 IT + 选择 npz + key + channel + 工具
        self.btn_root = QPushButton("选择根目录…")
        self.lbl_root = QLabel("未选择")
        self.lbl_root.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.combo_it = QComboBox(); self.combo_it.setMinimumWidth(160)
        self.combo_npz = QComboBox(); self.combo_npz.setMinimumWidth(240)
        self.combo_key = QComboBox(); self.combo_key.setMinimumWidth(200)
        self.spin_channel = QSpinBox(); self.spin_channel.setMinimum(0); self.spin_channel.setMaximum(0)
        self.spin_channel.setPrefix("通道:"); self.spin_channel.setEnabled(False)
        self.btn_refresh = QPushButton("载入/刷新")

        top = QHBoxLayout()
        top.addWidget(self.btn_root)
        top.addWidget(self.lbl_root, 1)
        top.addWidget(QLabel("积分时间:"))
        top.addWidget(self.combo_it)
        top.addWidget(QLabel("文件:"))
        top.addWidget(self.combo_npz)
        top.addWidget(QLabel("键(key):"))
        top.addWidget(self.combo_key)
        top.addWidget(self.spin_channel)
        top.addWidget(self.btn_refresh)

        # 工具栏：多点对比、清空、导出 CSV + 指定像素(x,y)
        self.chk_compare = QCheckBox("多点对比")
        self.btn_clear = QPushButton("清空曲线")
        self.btn_export = QPushButton("导出 CSV")
        self.spin_x = QSpinBox(); self.spin_x.setRange(0, 0); self.spin_x.setPrefix("X:")
        self.spin_y = QSpinBox(); self.spin_y.setRange(0, 0); self.spin_y.setPrefix("Y:")
        self.btn_add_xy = QPushButton("添加像素")
        # 初始禁用，待载入图像后启用
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

        right_panel = QWidget(); right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.canvas_plot)

        series_box = QGroupBox("已选像素（双击删除）")
        vbox = QVBoxLayout(series_box)
        self.list_series = QListWidget(); vbox.addWidget(self.list_series)
        btn_row = QHBoxLayout()
        self.btn_del_sel = QPushButton("删除选中")
        self.btn_del_last = QPushButton("删除最近")
        btn_row.addWidget(self.btn_del_sel); btn_row.addWidget(self.btn_del_last)
        vbox.addLayout(btn_row)
        right_layout.addWidget(series_box)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.canvas_img)
        splitter.addWidget(right_panel)
        splitter.setSizes([780, 500])

        # 底部状态 + 统计
        self.lbl_stats = QLabel("")
        self.status = QStatusBar(); self.status.showMessage("就绪")

        # 总体布局
        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(tools)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.lbl_stats)
        layout.addWidget(self.status)
        self.setLayout(layout)

        # 事件绑定
        self.btn_root.clicked.connect(self.on_choose_root)
        self.combo_it.currentIndexChanged.connect(self.on_it_changed)
        self.combo_npz.currentIndexChanged.connect(self.on_npz_changed)
        self.combo_key.currentIndexChanged.connect(self.on_key_changed)
        self.spin_channel.valueChanged.connect(self.on_channel_changed)
        self.btn_refresh.clicked.connect(self.on_refresh)

        self.canvas_img.mpl_connect('button_press_event', self.on_img_canvas_click)
        self.canvas_plot.mpl_connect('button_press_event', self.on_plot_canvas_click)
        self.list_series.itemDoubleClicked.connect(self.on_delete_selected_series)
        self.btn_del_sel.clicked.connect(self.on_delete_selected_series)
        self.btn_del_last.clicked.connect(self.on_delete_last_series)
        self.btn_add_xy.clicked.connect(self.on_add_xy)
        self.chk_compare.stateChanged.connect(self.on_compare_mode_changed)
        self.btn_clear.clicked.connect(self.on_clear_curves)
        self.btn_export.clicked.connect(self.on_export_csv)

        # 运行时状态
        self.current_it: Optional[str] = None
        self.current_npz: Optional[str] = None
        self.current_key: Optional[str] = None
        self.current_channel: Optional[int] = None
        self.current_img: Optional[np.ndarray] = None
        self.series: List[dict] = []  # {'x':int,'y':int,'val':float}

        # 字体
        set_app_font(); setup_chinese_font()

    # -------------------- 顶部逻辑 --------------------
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
        self._populate_it()
        self.status.showMessage("根目录已载入。选择积分时间与文件。", 5000)

    def _populate_it(self):
        self.combo_it.blockSignals(True)
        self.combo_npz.blockSignals(True)
        self.combo_key.blockSignals(True)
        try:
            self.combo_it.clear(); self.combo_npz.clear(); self.combo_key.clear()
            its = self.dm.list_it()
            if not its:
                QMessageBox.information(self, "提示", "没有发现任何积分时间目录（内含 .npz）。")
                return
            self.combo_it.addItems(its)
        finally:
            self.combo_it.blockSignals(False)
            self.combo_npz.blockSignals(False)
            self.combo_key.blockSignals(False)
        # 触发加载
        self.on_it_changed()

    def on_it_changed(self):
        it = self.combo_it.currentText()
        self.current_it = it if it else None
        self.series = []; self._refresh_series_list(); self._clear_plot()
        self._populate_npz()

    def _populate_npz(self):
        self.combo_npz.blockSignals(True)
        try:
            self.combo_npz.clear(); self.combo_key.clear()
            if not self.current_it:
                return
            files = self.dm.list_npz(self.current_it)
            if not files:
                QMessageBox.information(self, "提示", f"{self.current_it} 下没有 .npz 文件。")
                return
            # 显示相对文件名
            for p in files:
                self.combo_npz.addItem(os.path.basename(p), p)
        finally:
            self.combo_npz.blockSignals(False)
        self.on_npz_changed()

    def on_npz_changed(self):
        idx = self.combo_npz.currentIndex()
        self.current_npz = self.combo_npz.itemData(idx) if idx >= 0 else None
        self.series = []; self._refresh_series_list(); self._clear_plot()
        self._populate_keys()

    def _populate_keys(self):
        self.combo_key.blockSignals(True)
        try:
            self.combo_key.clear()
            self.spin_channel.setEnabled(False); self.spin_channel.setValue(0); self.spin_channel.setMaximum(0)
            if not (self.current_it and self.current_npz):
                return
            data = self.dm.load_npz(self.current_it, self.current_npz)
            # 过滤出适合可视化的 key
            keys = []
            for k, v in data.items():
                if isinstance(v, np.ndarray) and v.ndim in (2, 3):
                    keys.append(k)
            if not keys:
                QMessageBox.information(self, "提示", f"文件 {os.path.basename(self.current_npz)} 中没有可视化的 2D/3D 数组 key。")
                return
            keys.sort()
            self.combo_key.addItems(keys)
        finally:
            self.combo_key.blockSignals(False)
        self.on_key_changed()

    def on_key_changed(self):
        key = self.combo_key.currentText()
        self.current_key = key if key else None
        self.series = []; self._refresh_series_list(); self._clear_plot()
        self._update_image()

    def on_channel_changed(self, val: int):
        self.current_channel = int(val)
        self._update_image()

    def on_refresh(self):
        self._update_image()

    # -------------------- 图像/统计与点击 --------------------
    def _update_image(self):
        if not (self.current_it and self.current_npz and self.current_key):
            return
        data = self.dm.load_npz(self.current_it, self.current_npz)
        arr = data.get(self.current_key, None)
        if arr is None:
            QMessageBox.warning(self, "提示", f"找不到 key: {self.current_key}")
            # 无图像时禁用 x,y 输入
            self.spin_x.setEnabled(False); self.spin_y.setEnabled(False); self.btn_add_xy.setEnabled(False)
            return
        # 通道设置
        if arr.ndim == 3:
            C = arr.shape[2]
            self.spin_channel.setEnabled(True)
            self.spin_channel.setMaximum(max(0, C - 1))
            ch = self.spin_channel.value()
            ch = min(ch, C - 1)
            self.spin_channel.blockSignals(True)
            self.spin_channel.setValue(ch)
            self.spin_channel.blockSignals(False)
            img = self.dm.as_image(arr, channel=ch)
            self.current_channel = ch
        else:
            self.spin_channel.setEnabled(False)
            self.current_channel = None
            img = self.dm.as_image(arr)
        self.current_img = img
        # 更新 x,y 输入范围并启用
        H, W = img.shape[:2]
        self.spin_x.blockSignals(True); self.spin_y.blockSignals(True)
        self.spin_x.setRange(0, max(0, W - 1))
        self.spin_y.setRange(0, max(0, H - 1))
        # 若当前值越界则回退到 0
        if self.spin_x.value() > max(0, W - 1):
            self.spin_x.setValue(0)
        if self.spin_y.value() > max(0, H - 1):
            self.spin_y.setValue(0)
        self.spin_x.blockSignals(False); self.spin_y.blockSignals(False)
        self.spin_x.setEnabled(True); self.spin_y.setEnabled(True); self.btn_add_xy.setEnabled(True)

        self._show_image(img, title=f"IT={self.current_it} | {os.path.basename(self.current_npz)} | {self.current_key}" + (f"[ch={self.current_channel}]" if self.current_channel is not None else ""))
        self._update_stats(img)

    def _update_stats(self, img: np.ndarray):
        vmin = float(np.nanmin(img)) if img.size else np.nan
        vmax = float(np.nanmax(img)) if img.size else np.nan
        vmean = float(np.nanmean(img)) if img.size else np.nan
        H, W = img.shape[:2]
        self.lbl_stats.setText(f"尺寸: {W}x{H} | min={vmin:.6g} max={vmax:.6g} mean={vmean:.6g}")

    def _show_image(self, img: np.ndarray, title: str = ""):
        ax = self.canvas_img.ax
        ax.clear()
        ax.imshow(img, cmap='gray', aspect='equal')  # 按需求默认不显示 colorbar
        ax.set_title(title)
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        self.canvas_img.draw()

    def on_img_canvas_click(self, event):
        # 左键：添加/更新曲线；右键：删除最近曲线
        if event.inaxes != self.canvas_img.ax:
            return
        if self.current_img is None or self.current_key is None:
            return
        if event.button == 3:
            self.on_delete_last_series(); return
        if event.button != 1:
            return
        x = int(round(event.xdata)); y = int(round(event.ydata))
        self._add_xy_internal(x, y)

    def on_add_xy(self):
        if self.current_img is None:
            return
        x = int(self.spin_x.value()); y = int(self.spin_y.value())
        self._add_xy_internal(x, y)

    def _add_xy_internal(self, x: int, y: int):
        H, W = self.current_img.shape[:2]
        if x < 0 or y < 0 or x >= W or y >= H:
            QMessageBox.information(self, "提示", f"像素越界: x={x}, y={y}, 合法范围: 0<=x<{W}, 0<=y<{H}")
            return
        val = float(self.current_img[y, x])
        label = {'x': x, 'y': y, 'val': val}
        if self.chk_compare.isChecked():
            replaced = False
            for item in self.series:
                if item['x'] == x and item['y'] == y:
                    item['val'] = val
                    replaced = True
                    break
            if not replaced:
                self.series.append(label)
            title = f"IT={self.current_it} | {self.current_key} 像素对比" + (f"[ch={self.current_channel}]" if self.current_channel is not None else "")
        else:
            self.series = [label]
            title = f"(x={x}, y={y}) 的系数值"
        self._plot_series(title)
        self._refresh_series_list()
        self.spin_x.setValue(x); self.spin_y.setValue(y)
        self.status.showMessage(f"已添加/更新像素 (x={x}, y={y})，值={val:.6g}")

    def on_plot_canvas_click(self, event):
        if event.inaxes != self.canvas_plot.ax:
            return
        if event.button == 3:
            self.on_delete_last_series()

    # -------------------- 曲线/列表/导出 --------------------
    def _clear_plot(self):
        ax = self.canvas_plot.ax
        ax.clear(); ax.set_title("像素系数对比"); ax.set_xlabel("像素索引"); ax.set_ylabel("系数值")
        self.canvas_plot.draw()

    def _plot_series(self, title: str = ""):
        ax = self.canvas_plot.ax
        ax.clear()
        if not self.series:
            self._clear_plot(); return
        xs = np.arange(len(self.series))
        ys = [s['val'] for s in self.series]
        labels = [f"({s['x']},{s['y']})" for s in self.series]
        ax.plot(xs, ys, marker='o', linewidth=1.5)
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_title(title)
        ax.set_xlabel("像素坐标")
        ax.set_ylabel("系数值")
        ax.grid(True, linestyle='--', alpha=0.4)
        self.canvas_plot.draw()

    def _refresh_series_list(self):
        self.list_series.clear()
        for idx, s in enumerate(self.series):
            self.list_series.addItem(f"{idx+1}. (x={s['x']}, y={s['y']})  值:{s['val']:.6g}")

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
        self._plot_series(title=f"IT={self.current_it} | {self.current_key} 像素对比" + (f"[ch={self.current_channel}]" if self.current_channel is not None else ""))
        self._refresh_series_list()
        self.status.showMessage("已删除选中项。", 3000)

    def on_delete_last_series(self):
        if not self.series:
            self.status.showMessage("没有可删除的项。", 3000); return
        self.series.pop()
        self._plot_series(title=f"IT={self.current_it} | {self.current_key} 像素对比" + (f"[ch={self.current_channel}]" if self.current_channel is not None else ""))
        self._refresh_series_list()
        self.status.showMessage("已删除最近添加的项。", 3000)

    def on_compare_mode_changed(self, state):
        mode = "开启" if state == Qt.Checked else "关闭"
        self.status.showMessage(f"多点对比已{mode}。当前数量：{len(self.series)}", 3000)

    def on_clear_curves(self):
        self.series = []
        self._clear_plot(); self._refresh_series_list()
        self.status.showMessage("已清空。", 3000)

    def on_export_csv(self):
        if not self.series:
            QMessageBox.information(self, "提示", "没有可导出的数据。请先在图像上点击像素。")
            return
        default_name = f"coeff_{self.current_it or 'IT'}_{(self.current_key or 'key').replace('/', '_')}" \
                       + (f"_ch{self.current_channel}" if self.current_channel is not None else "") + ".csv"
        path, _ = QFileDialog.getSaveFileName(self, "保存 CSV", default_name, "CSV Files (*.csv)")
        if not path:
            return
        try:
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                header = ["x", "y", "value", "it", "file", "key", "channel"]
                writer.writerow(header)
                for s in self.series:
                    writer.writerow([s['x'], s['y'], s['val'], self.current_it or '',
                                     os.path.basename(self.current_npz or ''), self.current_key or '',
                                     '' if self.current_channel is None else self.current_channel])
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出 CSV 失败：\n{e}")
            return
        self.status.showMessage(f"CSV 已保存：{path}", 5000)

# ------------------------------ main ------------------------------
def main():
    app = QApplication(sys.argv)
    set_app_font(); setup_chinese_font()
    w = MainWindow(); w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
