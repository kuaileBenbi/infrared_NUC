from bisect import bisect_left
import os
import glob
import numpy as np
from PIL import Image
import cv2
from typing import Union, Tuple, Dict


def load_mean_image_pil(
    folder: str, imgtype: str = "*.png"
) -> Union[Tuple[np.ndarray, float], Tuple[None, None]]:
    """
    加载某个文件夹下的所有图像，并计算平均值

    :param folder: 文件夹路径
    :param imgtype: 图像类型，默认'png'
    :param normalize_time: 归一化时间，默认None(可选)
    :return: 时域平均图像，时域平均值

    """

    files = sorted(glob.glob(os.path.join(folder, imgtype)))

    if not files:
        print(f"No files in {folder}")
        return None, None

    acc = None

    for f in files:
        img = np.array(Image.open(f), dtype=np.float32)
        acc = img if acc is None else acc + img

    mean_img = acc / len(files)

    return mean_img, mean_img.mean()


def load_mean_image_cv2(
    folder: str, imgtype: str = "*.png"
) -> Union[Tuple[np.ndarray, float], Tuple[None, None]]:
    """
    加载某个文件夹下的所有图像，并计算平均值

    :param folder: 文件夹路径
    :param imgtype: 图像类型，默认'png'
    :param normalize_time: 归一化时间，默认None(可选)
    :return: 时域平均图像，时域平均值
    """

    files = sorted(glob.glob(os.path.join(folder, imgtype)))

    if not files:
        print(f"No files in {folder}")
        return None

    acc = []
    for f in files:
        img = cv2.imread(f, -1).astype(np.float32)
        acc.append(img)

    mean_img = np.mean(acc, axis=0)

    return mean_img, np.mean(mean_img)


def load_median_image_pil(
    folder: str, imgtype: str = "*.png"
) -> Union[Tuple[np.ndarray, float], Tuple[None, None]]:
    """
    加载某个文件夹下的所有图像，并计算中位值

    :param folder: 文件夹路径
    :param imgtype: 图像类型，默认'png'
    :return: 时域中位值图像，时域中位值
    """
    files = sorted(glob.glob(os.path.join(folder, imgtype)))

    if not files:
        print(f"No files in {folder}")
        return None, None

    acc = np.stack([np.array(Image.open(p), dtype=np.float32) for p in files], axis=0)

    median_img = np.median(acc, axis=0)

    return median_img, np.median(median_img)


def load_median_image_cv2(
    folder: str, imgtype: str = "*.png"
) -> Union[Tuple[np.ndarray, float], Tuple[None, None]]:
    """
    加载某个文件夹下的所有图像，并计算中位值

    :param folder: 文件夹路径
    :param imgtype: 图像类型，默认'png'
    :return: 时域中位值图像，时域中位值
    """
    files = sorted(glob.glob(os.path.join(folder, imgtype)))

    if not files:
        print(f"No files in {folder}")
        return None, None

    acc = np.stack([cv2.imread(f, -1).astype(np.float32) for f in files], axis=0)
    median_img = np.median(acc, axis=0)

    return median_img, np.median(median_img)


def gray_key(val: float, step: float = 1.0) -> int:
    """把灰度浮点离散成整数键，避免浮点相等比较问题。"""
    return int(round(val / step))


def _quantile_lo_hi(
    stack: np.ndarray, q_lo=0.01, q_hi=0.99, margin=8.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    stack: (N,H,W) 或 (N,P)
    返回逐像元 lo/hi（按分位数并留安全边）
    """
    if stack.ndim == 3:
        lo = np.quantile(stack, q_lo, axis=0)
        hi = np.quantile(stack, q_hi, axis=0)
    else:
        # (N,P)
        lo = np.quantile(stack, q_lo, axis=0)
        hi = np.quantile(stack, q_hi, axis=0)
    return (lo + margin).astype(np.float32), (hi - margin).astype(np.float32)


def select_training_data(
    images_by_temp: Dict[float, np.ndarray],
    knots: Union[np.ndarray, list],
    train_set: Dict[int, float],
    key_step: float = 1.0,
    match: str = "left",  # "left" 或 "nearest"
    tol: Union[
        float, None
    ] = None,  # 若设置阈值，超过阈值则跳过该 knot（单位是"量化后键"的距离）
):
    """
    images_by_temp: { temp -> image(H,W) }
    knots:          分位选出的灰度节点（float）
    train_set:      { gray_key -> temp }（注意：键已量化的字典）
    key_step:       灰度量化步长（与构造 train_set 时保持一致）
    match:          "left" 用左边界；"nearest" 用最近邻匹配
    tol:            可选的匹配容忍度；None 表示不限制
    """
    # 现有可用的灰度键（已量化、排序）
    keys_sorted = np.array(sorted(train_set.keys()), dtype=np.int64)

    imgs, temps = [], []
    used_temps = set()

    for g in knots:
        kq = gray_key(float(g), step=key_step)  # 查询键

        if kq in train_set:
            t = train_set[kq]
        else:
            # 在 keys_sorted 里找匹配
            pos = bisect_left(keys_sorted.tolist(), kq)
            if match == "left":
                idx = max(0, min(pos, len(keys_sorted) - 1))
            elif match == "nearest":
                # 左右各看一个，谁更近用谁
                candidates = []
                if pos > 0:
                    candidates.append(keys_sorted[pos - 1])
                if pos < len(keys_sorted):
                    candidates.append(keys_sorted[min(pos, len(keys_sorted) - 1)])
                # 选距离最近的
                idx = np.argmin([abs(c - kq) for c in candidates])
                k_sel = candidates[idx]
                if (tol is not None) and (abs(k_sel - kq) > tol):
                    # 超过阈值，跳过这个 knot
                    continue
                t = train_set[int(k_sel)]
                # 后面还会检查 images_by_temp 是否有这个温度
                if t in images_by_temp and t not in used_temps:
                    imgs.append(images_by_temp[t])
                    temps.append(t)
                    used_temps.add(t)
                continue
            else:
                raise ValueError("match 应为 'left' 或 'nearest'")

            k_sel = int(keys_sorted[idx])
            if (tol is not None) and (abs(k_sel - kq) > tol):
                # 左边界匹配但距离太大，跳过
                continue
            t = train_set[k_sel]

        # 收集样本（去重）
        if t in images_by_temp and t not in used_temps:
            imgs.append(images_by_temp[t])
            temps.append(t)
            used_temps.add(t)

    if not imgs:
        raise RuntimeError(
            "select_training_data: 未匹配到任何训练样本。请检查 knots/train_set 对齐与 key_step。"
        )

    return np.stack(imgs, axis=0), np.asarray(temps, dtype=np.float32)
