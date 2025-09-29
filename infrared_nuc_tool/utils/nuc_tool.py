import cv2
import numpy as np
from .pre_tool import _quantile_lo_hi


def bright_dark_fit(
    wish_bright_value: int,
    wish_dark_value: int,
    bright_arr: np.ndarray,
    dark_arr: np.ndarray,
    save_path=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    适用于短波的两点校正
    数据: 明/暗两个灰度值点 (对于12位探测器选择了1024/3072两个灰度值点)
    校正公式: L=gain_arr*R+offset_arr
    gain_arr: 增益系数
    offset_arr: 偏移量
    目标: 将明/暗两个灰度值点校正到期望的明/暗值
    输出: 校正后的增益和偏移量

    :param wish_bright_value: 期望的明值
    :param wish_dark_value: 期望的暗值
    :param bright_arr: (H, W) 明值数组 时域均值
    :param dark_arr: (H, W) 暗值数组 时域均值
    :param save_path: 保存路径
    :return: 校正后的增益和偏移量 .npz文件

    """

    gain = (wish_bright_value - wish_dark_value) / (bright_arr - dark_arr + 1e-6)
    offset = wish_dark_value - gain * dark_arr
    ref = wish_bright_value - wish_dark_value

    if save_path is not None:

        print(f"[bright_dark_fit] Saving gain_map and offset_map to > {save_path}")
        np.savez_compressed(save_path, gain_map=gain, offset_map=offset, ref=ref)

    return gain, offset, ref


def bright_dark_apply(
    gain_map: np.ndarray,
    offset_map: np.ndarray,
    ref: float,
    raw_arr: np.ndarray,
    bit_max: int,
) -> np.ndarray:
    """
    适用于短波的两点校正
    对原始数据进行校正
    校正公式: L=gain_map*R+offset_map
    gain_map: 增益系数
    offset_map: 偏移量
    目标: 将原始数据校正到期望的明/暗值
    输出: 校正后的数据

    :param gain_map: 增益系数
    :param offset_map: 偏移量
    :param raw_arr: 原始数据
    :param bit_max: 比特位最大值
    :return: 校正后的数据

    Note:
        1. 输入图像应为 uint16 类型
        2. 输出图像为 uint16 类型
    """

    if not (gain_map.shape == offset_map.shape == raw_arr.shape):
        raise ValueError(
            f"[bright_dark_apply] gain_map, offset_arr and raw_arr must have the same shape"
        )

    raw_arr = raw_arr.astype(np.float32)

    corrected_arr = gain_map * raw_arr + offset_map

    corrected_arr = np.clip(corrected_arr, 0, ref)
    corrected_arr = (corrected_arr / ref) * bit_max

    print(
        f"[bright_dark_apply] Corrected done: raw mean {raw_arr.mean():.2f} -> corrected mean {corrected_arr.mean():.2f}"
    )

    return corrected_arr.astype(np.uint16)


def linear_fit(
    it_temps_arr: np.ndarray,
    x_arr: np.ndarray,
    bit_max: int,
    min_dynamic: float = 500.0,
    eps_gain: float = 1e-3,
    save_path=None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    适用于中/长波的一次线性校正
    数据: 某个积分时间下的不同温度点数据
    校正公式: y_ij^k ≈ a_ij * X_k + b_ij
    用最小二乘法迭代计算出a_ij和b_ij

    :param it_temps_arr: (N, H, W) 同一积分时间下, 多温度点的时域均值图像数组
    :param x_arr: (N,) 温度点/每个温度点的理想期望值
    :param bit_max: 原始位宽上限（用于判定近黑/近饱和）
    :param min_dynamic: 自变量动态范围下限 默认500.0 只在输入x_arr为灰度值时有效
    :param eps_gain: 增益下限 默认1e-3

    :param save_path: 保存路径
    :return: 校正后的增益和偏移量 .npz文件

    """

    assert it_temps_arr.ndim == 3, "it_temps_arr 形状应为 (N, H, W)"
    N, H, W = it_temps_arr.shape

    x = np.asarray(x_arr, dtype=np.float32)
    assert x.shape[0] == N, "x_vals 长度需与 it_temps_arr 帧数一致"

    # 动态范围检查
    dyn = float(x.max() - x.min())
    if dyn < min_dynamic:
        # 只警告不中断：有些相机动态范围天然较小
        print(
            f"[warn] 自变量动态范围较小：{dyn:.1f} < {min_dynamic:.1f}，标定可能偏弱。"
        )

    # —— 归一化 + 中心化 —— #
    x_max = float(max(x.max(), 1e-6))
    x_norm = x / x_max
    mean_x = float(x_norm.mean())
    var_x = float(x_norm.var())
    try:
        if var_x < 1e-12:
            raise RuntimeError("自变量方差过小，无法回归（所有帧灰度几乎相同）。")
    except RuntimeError:
        print(f"var_x: {var_x}")
        return np.zeros((H, W)), np.zeros((H, W)), 0, 0

    # —— 计算每像素回归系数（向量化） —— #
    # y ≈ a * x_norm + b
    Y = it_temps_arr.reshape(N, -1).astype(np.float32)  # (N, P)
    mean_Y = Y.mean(axis=0)  # (P,)
    cov_XY = ((x_norm[:, None] - mean_x) * (Y - mean_Y[None, :])).mean(axis=0)  # (P,)
    a_vec = cov_XY / var_x
    b_vec = mean_Y - a_vec * mean_x

    # 工程稳健：增益下限
    # a_vec = a_vec + eps_gain
    a_vec = np.where(a_vec > eps_gain, a_vec, eps_gain)

    a_map = a_vec.reshape(H, W).astype(np.float32)
    b_map = b_vec.reshape(H, W).astype(np.float32)

    global_a = float(a_vec.mean())
    global_b = float(b_vec.mean())

    if save_path is not None:
        print(f"[linear_fit] Saving a_map, b_map, global_a, global_b to > {save_path}")
        np.savez_compressed(
            save_path, a_map=a_map, b_map=b_map, ga=global_a, gb=global_b
        )

    return a_map, b_map, global_a, global_b


def linear_apply(
    a_map: np.ndarray,
    b_map: np.ndarray,
    global_a: float,
    global_b: float,
    raw_arr: np.ndarray,
    bit_max: int,
    eps_gain: float = 1e-3,
) -> np.ndarray:
    """
    适用于中/长波的一次线性校正, 映射到 bit_max
    对原始数据进行校正

    校正公式:
    corrected_arr = (raw_arr - b_map) / a_map          （像素级去非均匀）
    corrected_arr = corrected_arr * global_a + global_b 回到“全局参考”亮度

    :param a_map: (H, W) 单像素增益系数
    :param b_map: (H, W) 单像素偏移量
    :param global_a: 全局增益系数
    :param global_b: 全局偏移量
    :param raw_arr: 原始数据
    :param bit_max: 原始位宽上限（用于判定近黑/近饱和）
    :param eps_gain: 增益下限 默认1e-3
    :return: 校正后的数据

    Note:
        1. 输入图像应为 uint16 类型
        2. 输出图像为 uint16 类型
    """

    if not (a_map.shape == b_map.shape == raw_arr.shape):
        raise ValueError(
            f"[linear_apply] a_map, b_map and raw_arr must have the same shape"
        )

    raw_arr = raw_arr.astype(np.float32)

    # a_safe = a_map + eps_gain
    a_safe = np.where(a_map > eps_gain, a_map, eps_gain)

    corrected_arr = (raw_arr - b_map) / a_safe

    corrected_arr = corrected_arr * global_a + global_b

    corrected_arr = np.clip(corrected_arr, 0, bit_max)

    print(
        f"[linear_apply] Corrected done: raw mean {raw_arr.mean():.2f} -> corrected mean {corrected_arr.mean():.2f}"
    )

    return corrected_arr.astype(np.uint16)


def quadratic_fit(
    it_temps_arr: np.ndarray, x_arr: np.ndarray, bit_max: int, save_path=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    适用于中/长波的二次校正（鲁棒版，自适应高低积分时间），映射到 bit_max/期望灰度
    数据: 某个积分时间下的不同温度点数据
    拟合: 对每像元，用 (观测灰度 X -> 期望灰度 y) 拟合 y ≈ a2*X^2 + a1*X + a0

    Args:
        it_temps_arr: (N, H, W) 同一积分时间下，多温度点的时域均值图像数组（建议 uint16 / float32）
        x_arr:        (N,) 每个温度点对应的“期望灰度”或“理想响应”（可按你的标定定义）
        bit_max:      位深上限（用于判断 0 / 饱和）
        save_path:    若不为 None，则保存 npz（包含 a2,a1,a0,bad）

    Returns:
        (a2_arr, a1_arr, a0_arr, bad_pixel_map)
        bad_pixel_map: 1=坏点/回退（建议应用端做空间插值或单位映射兜底）
    """

    # ===== 可调鲁棒参数 =====
    min_pts_base = 3              # 最少有效点（归一化+二次至少需要3）
    huber_delta  = 20.0           # Huber 阈值（以 y 的单位计）
    mad_k        = 3.5            # MAD * k 之外的点视作离群
    max_iter     = 3              # 鲁棒迭代轮数
    neg_deriv_ratio_thresh = 0.6  # 导数为负比例超此阈值则判为不稳（退化到线性）
    sat_gate_ratio = 0.10         # 饱和占比门限，≥该值认为“高积分时间样态”

    images = np.asarray(it_temps_arr, dtype=np.float32)  # (N,H,W)
    wish_gray = np.asarray(x_arr, dtype=np.float32)      # (N,)
    N, H, W = images.shape

    a2_arr = np.zeros((H, W), dtype=np.float32)
    a1_arr = np.zeros((H, W), dtype=np.float32)
    a0_arr = np.zeros((H, W), dtype=np.float32)
    bad_pixel_map = np.zeros((H, W), dtype=np.uint8)     # 1=坏点/回退

    # ---------- 内部小工具 ----------
    def _mad(x):
        med = np.median(x)
        return np.median(np.abs(x - med)) + 1e-12

    def _huber_weights(r, delta):
        a = np.abs(r)
        w = np.ones_like(r, dtype=np.float32)
        big = a > delta
        # w = delta/|r|
        w[big] = delta / (a[big] + 1e-12)
        return w

    def _polyfit_w(x, y, deg, w=None):
        # numpy>=1.22 支持 w=样本权重
        if w is not None:
            return np.polyfit(x, y, deg=deg, w=w, rcond=None)
        return np.polyfit(x, y, deg=deg, rcond=None)

    def _drop_tail_plateau(x_sorted, y_sorted, dx_eps, min_run=3):
        """
        尾端平台剔除：从尾部起，连续若干相邻点满足 Δx<=dx_eps 则剔除这一段。
        x_sorted/y_sorted 已按 y 升序对齐（通常代表温度/期望递增）。
        """
        L = len(x_sorted)
        if L <= 3:
            return x_sorted, y_sorted
        dx = np.diff(x_sorted)
        run = 0
        for k in range(L - 2, -1, -1):
            if dx[k] <= dx_eps:
                run += 1
            else:
                break
        if run >= min_run:
            keep_n = L - run
            if keep_n >= 3:
                return x_sorted[:keep_n], y_sorted[:keep_n]
        return x_sorted, y_sorted

    def _to_original_coef_from_unit(coef_n, x_min, x_max, y_min, y_max):
        """
        已在单位区间上拟合: y_n = a2n*x_n^2 + a1n*x_n + a0n,
        映射: x_n=(x-x_min)/sx, y = y_min + ty*y_n
        反解出原尺度上的 (A2, A1, A0) 使得 y = A2*x^2 + A1*x + A0
        """
        a2n, a1n, a0n = coef_n if len(coef_n) == 3 else (0.0, coef_n[0], coef_n[1])
        sx = (x_max - x_min) + 1e-12
        ty = (y_max - y_min) + 1e-12

        A2 = ty * (a2n / (sx ** 2))
        A1 = ty * (a1n / sx - 2.0 * a2n * x_min / (sx ** 2))
        A0 = ty * (a2n * (x_min ** 2) / (sx ** 2) - a1n * x_min / sx + a0n) + y_min
        return float(A2), float(A1), float(A0)

    # ---------- 主循环（逐像元） ----------
    for i in range(H):
        # 可视需要加进度打印；此处保持安静
        for j in range(W):
            x = images[:, i, j]   # 该像元在 N 个温点上的观测灰度
            y = wish_gray         # 对应的期望灰度

            # 0) 基本有效性：去掉明确的 0 / 满刻度 / NaN/Inf
            finite = np.isfinite(x)
            base_valid = finite & (x > 0.0) & (x < float(bit_max))
            if base_valid.sum() < min_pts_base:
                # 回退为单位映射 + 标坏点
                a2_arr[i, j], a1_arr[i, j], a0_arr[i, j] = 0.0, 1.0, 0.0
                bad_pixel_map[i, j] = 1
                continue

            xv = x[base_valid]
            yv = y[base_valid]

            # 1) 自适应：依据饱和占比决定阈值与平台策略
            sat_ratio = (x >= 0.98 * float(bit_max)).mean()
            if sat_ratio >= sat_gate_ratio:
                # 高积分时间样态：放宽端点、允许平台剔除
                valid = (x > 0.0) & (x < float(bit_max)) & np.isfinite(x)
                xv, yv = x[valid], y[valid]
                # 平台阈值随位深缩放
                dx_eps = max(2.0, 0.001 * float(bit_max))  # 12bit≈4, 16bit≈65
                enable_drop_plateau = True
            else:
                # 低积分时间样态：更稳健的分位数裁剪，通常无需平台剔除
                xf = x[base_valid]
                if len(xf) >= 8:
                    lo, hi = np.percentile(xf, [1, 99])
                    valid = (x > lo) & (x < hi) & np.isfinite(x)
                    xv, yv = x[valid], y[valid]
                else:
                    # 样本少时沿用 base_valid
                    xv, yv = xv, yv
                dx_eps = None
                enable_drop_plateau = False

            if len(xv) < min_pts_base:
                a2_arr[i, j], a1_arr[i, j], a0_arr[i, j] = 0.0, 1.0, 0.0
                bad_pixel_map[i, j] = 1
                continue

            # 2) 为稳定性，按“期望值”排序（通常随温度单调）
            order = np.argsort(yv)
            xv, yv = xv[order], yv[order]

            # 3) 可选：尾端平台剔除（仅高积分时间样态启用）
            if enable_drop_plateau:
                xv, yv = _drop_tail_plateau(xv, yv, dx_eps=dx_eps, min_run=3)
                if len(xv) < min_pts_base:
                    a2_arr[i, j], a1_arr[i, j], a0_arr[i, j] = 0.0, 1.0, 0.0
                    bad_pixel_map[i, j] = 1
                    continue

            # 4) 归一化到 [0,1] 以避免高位深病态
            x_min, x_max = float(np.min(xv)), float(np.max(xv))
            y_min, y_max = float(np.min(yv)), float(np.max(yv))
            sx = (x_max - x_min) + 1e-12
            ty = (y_max - y_min) + 1e-12
            xvn = (xv - x_min) / sx
            yvn = (yv - y_min) / ty

            # 若归一化后仍点数不足，直接回退
            if len(xvn) < min_pts_base:
                a2_arr[i, j], a1_arr[i, j], a0_arr[i, j] = 0.0, 1.0, 0.0
                bad_pixel_map[i, j] = 1
                continue

            # 5) 初始二次拟合
            deg = 2 if len(xvn) >= 3 else 1
            try:
                coef_n = _polyfit_w(xvn, yvn, deg=deg)
            except Exception:
                a2_arr[i, j], a1_arr[i, j], a0_arr[i, j] = 0.0, 1.0, 0.0
                bad_pixel_map[i, j] = 1
                continue

            # 6) 鲁棒迭代：MAD裁剪 + Huber 重加权
            last_inlier = len(xvn)
            for _ in range(max_iter):
                yhat = np.polyval(coef_n, xvn)
                r = yvn - yhat
                mad = _mad(r)
                inliers = np.abs(r) <= (mad_k * mad)
                if inliers.sum() < (3 if deg == 2 else 2):
                    break
                x_in = xvn[inliers]
                y_in = yvn[inliers]
                r_in = r[inliers]
                w = _huber_weights(r_in, huber_delta / (ty + 1e-12))  # 将阈值缩放到 y_n 空间

                try:
                    coef_new = _polyfit_w(x_in, y_in, deg=deg, w=w)
                except Exception:
                    break

                coef_n = coef_new
                if inliers.sum() == last_inlier:
                    break
                last_inlier = inliers.sum()

            # 7) 稳定性检查：导数在 [0,1] 区间的大面积为负 → 线性回退
            if deg == 2:
                a2n, a1n, _ = coef_n
                xs = np.linspace(0.0, 1.0, num=9, dtype=np.float32)
                deriv = (ty / sx) * (2.0 * a2n * xs + a1n)  # dy/dx，比例因子为正，不影响符号
                if (deriv < 0).mean() > neg_deriv_ratio_thresh:
                    # 线性回退（在单位区间上重拟合）
                    try:
                        coef_n = _polyfit_w(xvn, yvn, deg=1)
                        deg = 1
                    except Exception:
                        a2_arr[i, j], a1_arr[i, j], a0_arr[i, j] = 0.0, 1.0, 0.0
                        bad_pixel_map[i, j] = 1
                        continue

            # 8) 将单位区间系数还原到原尺度
            try:
                A2, A1, A0 = _to_original_coef_from_unit(coef_n, x_min, x_max, y_min, y_max)
            except Exception:
                A2, A1, A0 = 0.0, 1.0, 0.0
                bad_pixel_map[i, j] = 1

            # 9) 最终写回
            a2_arr[i, j] = A2
            a1_arr[i, j] = A1
            a0_arr[i, j] = A0
            # 是否标坏点：若发生过回退/过度裁剪你也可置1，这里仅在明确失败时置1

    if save_path is not None:
        print(f"[quadratic_fit] Saving a2_arr, a1_arr, a0_arr, bad_pixel_map to > {save_path}")
        np.savez_compressed(save_path, a2=a2_arr, a1=a1_arr, a0=a0_arr, bad=bad_pixel_map)

    return a2_arr, a1_arr, a0_arr, bad_pixel_map


    # """
    # 适用于中/长波的二次校正, 映射到bit_max
    # 数据: 某个积分时间下的不同温度点数据
    # 校正公式: 对每像元，用 (X -> y) 拟合: y ≈ a2*X^2 + a1*X + a0
    # 用np.polyfit计算出a_ij, b_ij和c_ij

    # :param it_temps_arr: (N, H, W) 同一积分时间下, 多温度点的时域均值图像数组
    # :param x_arr: (N,) 温度点/每个温度点的理想期望值
    # :param bit_max: 原始位宽上限（用于判定近黑/近饱和）
    # :param save_path: 保存路径
    # :return: 校正后的系数 .npz文件

    # Note:
    #     1. 输入图像列表 images 中的每个元素应为 uint16 类型
    #     2. 期望的灰度值列表 x_arr 应与输入图像列表长度一致
    #     3. 坏点标记数组 bad_pixel_map 用于标记哪些像素在拟合过程中被认为是坏点
    #     4. 拟合过程中，如果像素值为 0 或 16383 则认为该像素为坏点
    # """

    # # ------- 可调鲁棒参数（按需修改）-------
    # black_lo_frac = 0.05  # 低端裁剪阈值: x <= black_lo_frac*bit_max 视作近黑无效
    # sat_hi_frac = 0.95  # 高端裁剪阈值: x >= sat_hi_frac*bit_max 视作近饱和无效
    # min_pts = 5  # 至少需要的有效点数（<5 时二次拟合容易不稳）
    # huber_delta = 20.0  # Huber 损失阈值（单位=“y”的单位; 你用的 x_arr 的尺度）
    # mad_k = 3.5  # MAD * k 之外的点视作离群点
    # max_iter = 3  # 鲁棒迭代次数（残差裁剪 + Huber 重加权）
    # allow_linear_fallback = True  # 二次拟合点数不足或不稳时回退到一次拟合
    # drop_tail_plateau = True  # 额外检测“早饱和平台”并剔尾
    # plateau_dx_eps = 1.0  # 平台检测：相邻两点 x 差<=此阈值，视作平台
    # plateau_min_run = 2  # 平台检测：连续平台点的最小长度（计尾端）

    # images = it_temps_arr.astype(np.float32)  # (N,H,W)
    # wish_gray = np.asarray(x_arr, dtype=np.float32)  # (N,)
    # N, H, W = images.shape

    # a2_arr = np.zeros((H, W), dtype=np.float32)
    # a1_arr = np.zeros((H, W), dtype=np.float32)
    # a0_arr = np.zeros((H, W), dtype=np.float32)
    # bad_pixel_map = np.zeros((H, W), dtype=np.uint8)  # 1=坏点

    # low_thr = black_lo_frac * float(bit_max)
    # high_thr = sat_hi_frac * float(bit_max)

    # def _polyfit_w(x, y, deg, w=None):
    #     # numpy>=1.22 支持 w=sample_weights
    #     if w is not None:
    #         return np.polyfit(x, y, deg=deg, w=w, rcond=None)
    #     return np.polyfit(x, y, deg=deg, rcond=None)

    # def _mad(x):
    #     med = np.median(x)
    #     return np.median(np.abs(x - med)) + 1e-6

    # def _huber_weights(r, delta):
    #     # |r|<=delta -> w=1; 否则 w=delta/|r|
    #     a = np.abs(r)
    #     w = np.ones_like(r, dtype=np.float32)
    #     big = a > delta
    #     w[big] = delta / a[big]
    #     return w

    # # 可选：检测尾端平台（“早饱和但未到 bit_max”的情况），剔除尾段平台点
    # def _drop_tail_plateau(x_sorted, y_sorted):
    #     """
    #     x_sorted, y_sorted 已按 y（或温度）顺序排列。
    #     若尾端出现连续平台(Δx很小)则剔除末尾这段。
    #     """
    #     if len(x_sorted) <= 3:
    #         return x_sorted, y_sorted
    #     dx = np.diff(x_sorted)
    #     # 从尾端往前找连续 dx<=阈值的 run
    #     run = 0
    #     for k in range(len(dx) - 1, -1, -1):
    #         if dx[k] <= plateau_dx_eps:
    #             run += 1
    #         else:
    #             break
    #     if run >= plateau_min_run:
    #         keep_n = len(x_sorted) - run
    #         if keep_n >= 3:
    #             return x_sorted[:keep_n], y_sorted[:keep_n]
    #     return x_sorted, y_sorted

    # for i in range(H):
    #     # 为了速度，这里只在列上循环；也可以完全双层循环，清晰但更慢
    #     for j in range(W):
    #         x = images[:, i, j]  # 像元在不同温度点的灰度（自变量）
    #         y = wish_gray  # 同一组“期望值”（因变量）

    #         # 1) 基于绝对阈值的端点裁剪（避免 0/bit_max 的硬编码误判）
    #         valid = (x > low_thr) & (x < high_thr) & np.isfinite(x)
    #         if valid.sum() < min_pts:
    #             # 数据太少 -> 记坏点
    #             bad_pixel_map[i, j] = 1
    #             continue

    #         xv = x[valid]
    #         yv = y[valid]

    #         # 为稳定性，按“温度/期望值”的顺序排序（通常 y 随温度单调）
    #         order = np.argsort(yv)
    #         xv, yv = xv[order], yv[order]

    #         # 2) 可选：尾端平台检测，剔除“早饱和”导致的平坦段
    #         if drop_tail_plateau:
    #             xv, yv = _drop_tail_plateau(xv, yv)
    #             if len(xv) < min_pts:
    #                 bad_pixel_map[i, j] = 1
    #                 continue

    #         # 3) 初始二次拟合（点数不足则考虑线性回退）
    #         deg = 2 if len(xv) >= 3 else 1
    #         try:
    #             coef = _polyfit_w(xv, yv, deg=deg)  # coef: [a2,a1,a0] 或 [a1,a0]
    #         except Exception:
    #             bad_pixel_map[i, j] = 1
    #             continue

    #         # 4) 鲁棒迭代：MAD裁剪 + Huber加权
    #         last_inlier_count = len(xv)
    #         for _ in range(max_iter):
    #             yhat = np.polyval(coef, xv)
    #             r = yv - yhat
    #             mad = _mad(r)
    #             inliers = np.abs(r) <= (mad_k * mad)
    #             if inliers.sum() < (3 if deg == 2 else 2):
    #                 # 太少则停止
    #                 break
    #             xv_in = xv[inliers]
    #             yv_in = yv[inliers]
    #             r_in = r[inliers]
    #             w = _huber_weights(r_in, huber_delta)

    #             try:
    #                 coef_new = _polyfit_w(xv_in, yv_in, deg=deg, w=w)
    #             except Exception:
    #                 break

    #             coef = coef_new
    #             if inliers.sum() == last_inlier_count:
    #                 # 收敛（inlier 数未变化）
    #                 break
    #             last_inlier_count = inliers.sum()

    #         # 5) 若二次不稳或点数仍偏少，尝试线性回退
    #         if allow_linear_fallback and deg == 2:
    #             # 简单稳健性检查：导数在样本域内大面积为负可能不合理（可根据业务选择）
    #             xs = np.linspace(np.min(xv), np.max(xv), num=5, dtype=np.float32)
    #             deriv = 2.0 * coef[0] * xs + coef[1]
    #             if (deriv < 0).mean() > 0.6 or last_inlier_count < min_pts:
    #                 # 回退到线性
    #                 try:
    #                     coef_lin = _polyfit_w(xv, yv, deg=1)
    #                     # 将线性系数扩展到二次形态，便于统一返回
    #                     a2, a1, a0 = 0.0, float(coef_lin[0]), float(coef_lin[1])
    #                 except Exception:
    #                     bad_pixel_map[i, j] = 1
    #                     continue
    #             else:
    #                 a2, a1, a0 = float(coef[0]), float(coef[1]), float(coef[2])
    #         else:
    #             # 统一到 (a2,a1,a0)
    #             if deg == 2:
    #                 a2, a1, a0 = float(coef[0]), float(coef[1]), float(coef[2])
    #             else:
    #                 a2, a1, a0 = 0.0, float(coef[0]), float(coef[1])

    #         a2_arr[i, j] = a2
    #         a1_arr[i, j] = a1
    #         a0_arr[i, j] = a0

    # if save_path is not None:
    #     print(
    #         f"[quadratic_fit] Saving a2_arr, a1_arr, a0_arr, bad_pixel_map to > {save_path}"
    #     )
    #     np.savez_compressed(
    #         save_path, a2=a2_arr, a1=a1_arr, a0=a0_arr, bad=bad_pixel_map
    #     )

    # return a2_arr, a1_arr, a0_arr, bad_pixel_map

    # images = it_temps_arr.astype(np.float32)
    # wish_gray = np.array(x_arr, dtype=np.float32)

    # height, width = images[0].shape
    # a2_arr = np.zeros((height, width), dtype=np.float32)
    # a1_arr = np.zeros((height, width), dtype=np.float32)
    # a0_arr = np.zeros((height, width), dtype=np.float32)
    # bad_pixel_map = np.zeros((height, width), dtype=np.uint8)  # 1 表示坏点

    # for i in range(height):

    #     for j in range(width):

    #         cur_gray = wish_gray
    #         pixel_values = np.array([img[i, j] for img in images])

    #         if pixel_values.min() == 0:
    #             a2, a1, a0 = 0, 0, 0
    #             bad_pixel_map[i, j] = 1
    #             continue

    #         while len(pixel_values) > 2:
    #             if pixel_values.max() == bit_max:
    #                 pixel_values = pixel_values[:-1]
    #                 cur_gray = cur_gray[:-1]
    #             else:
    #                 break

    #         if len(pixel_values) > 2:
    #             coeffs = np.polyfit(pixel_values, cur_gray, deg=2)
    #             a2, a1, a0 = coeffs
    #         else:
    #             a2, a1, a0 = 0, 0, 0
    #             bad_pixel_map[i, j] = 1

    #         a2_arr[i, j] = a2
    #         a1_arr[i, j] = a1
    #         a0_arr[i, j] = a0

    # if save_path is not None:
    #     print(
    #         f"[quadratic_fit] Saving a2_arr, a1_arr, a0_arr, bad_pixel_map to > {save_path}"
    #     )
    #     np.savez_compressed(save_path, a2=a2_arr, a1=a1_arr, a0=a0_arr)

    # return a2_arr, a1_arr, a0_arr, bad_pixel_map


def quadratic_apply(
    a2_arr: np.ndarray,
    a1_arr: np.ndarray,
    a0_arr: np.ndarray,
    raw_arr: np.ndarray,
    bit_max: int,
) -> np.ndarray:
    """
    适用于中/长波的二次校正, 映射到bit_max
    对原始数据进行校正
    校正公式: 对每像元，用 (X -> y) 拟合: y ≈ a2*X^2 + a1*X + a0

    :param a2_arr: (H, W) 二次项系数
    :param a1_arr: (H, W) 一次项系数
    :param a0_arr: (H, W) 常数项
    :param raw_arr: 原始数据
    :param bit_max: 原始位宽上限（用于判定近黑/近饱和）
    :return: 校正后的数据

    Note:
        1. 输入图像应为 uint16 类型
        2. 输出图像为 uint16 类型
    """

    raw_arr_float = raw_arr.astype(np.float32)

    # 预分配输出数组
    corrected_arr = np.empty_like(raw_arr_float, dtype=np.float32)

    # 原地计算：y = a2 * x^2 + a1 * x + a0
    np.multiply(
        raw_arr_float, raw_arr_float, out=corrected_arr
    )  # corrected_arr = raw_arr^2
    np.multiply(
        corrected_arr, a2_arr, out=corrected_arr
    )  # corrected_arr = a2 * raw_arr^2
    corrected_arr += a1_arr * raw_arr_float  # corrected_arr += a1 * raw_arr
    corrected_arr += a0_arr  # corrected_arr += a0

    corrected_arr = np.clip(corrected_arr, 0, bit_max)

    print(
        f"[quadratic_apply] Corrected done: raw mean {raw_arr.mean():.2f} -> corrected mean {corrected_arr.mean():.2f}"
    )

    return corrected_arr.astype(np.uint16)
