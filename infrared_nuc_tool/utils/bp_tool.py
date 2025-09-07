import numpy as np
import cv2
import os, glob
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import generic_filter


def detect_bp_by_voltage(data, temperatures) -> np.ndarray:
    """
    根据平均响应率和平均噪声电压，检测死像元和过热像元。

    参数:
    - data: (num_temperatures, num_samples, height, width) 的 4D 数组
            表示不同温度、不同样本和每个像素点的值。
    - temperatures: 温度数组

    返回:
    - dead_pixels: (height, width) 的布尔数组，表示死像元位置
    - hot_pixels: (height, width) 的布尔数组，表示过热像元位置
    - defective_pixels: (height, width) 的布尔数组，表示缺陷像元位置

    """

    _, _, height, width = data.shape

    # 用于存储响应率和噪声电压的矩阵
    response_rates = np.zeros((height, width))
    noise_voltages = np.zeros((height, width))

    # 遍历每个像素
    for i in range(height):
        for j in range(width):
            # 获取每个温度下该像素的所有样本值 (num_temperatures, num_samples)
            pixel_values = data[:, :, i, j]

            # 计算该像素在不同温度下的平均值 (num_temperatures,)
            pixel_mean_values = np.mean(pixel_values, axis=1)

            # 线性拟合获取响应率 (slope)
            slope, _ = np.polyfit(temperatures, pixel_mean_values, 1)

            # 响应率即为线性拟合的斜率
            response_rates[i, j] = slope

            # 计算噪声电压（使用各温度下的方差作为噪声指标）
            noise_variance = np.var(pixel_values, axis=1)

            # 平均噪声电压取各温度下方差的均值
            noise_voltages[i, j] = np.mean(noise_variance)

    # 计算所有像素的平均响应率和平均噪声电压
    average_response_rate = np.mean(response_rates)
    average_noise_voltage = np.mean(noise_voltages)

    # 判定死像元：响应率 < 平均响应率的 1/2
    dead_pixels = response_rates < (average_response_rate / 2)
    # 判定过热像元：噪声电压 > 平均噪声电压的 2 倍
    hot_pixels = noise_voltages > (average_noise_voltage * 2)

    return np.logical_or(dead_pixels, hot_pixels)


def detect_bp_by_3sigma(
    images: np.ndarray, sigma: float = 3.0, window_size: int = 33
) -> np.ndarray:
    """
    根据3sigma法则检测孤立点认为为盲元
    使用全局窗口计算均值和标准差

    参数:
    - images: 图像数组 (num_temperatures, height, width)
    - window_size: 窗口大小
    - sigma: 阈值系数

    返回:
    - 尺寸为(height, width)的布尔数组 True表示对应位置是盲元
    """

    bps = np.zeros((images.shape[1], images.shape[2]), dtype=bool)

    for image in images:
        half = window_size // 2
        padded = np.pad(image, half, mode="reflect")
        view = sliding_window_view(padded, (window_size, window_size))

        height, width = image.shape

        flat = view.reshape(-1, window_size * window_size)

        means = flat.mean(axis=1).reshape(height, width)
        stds = flat.std(axis=1).reshape(height, width)

        cur_bp = np.abs(image - means) > (sigma * stds + 1e-8)
        bps = np.logical_or(bps, cur_bp)

    return bps


def detect_bp_by_3sigma_generic_filter(
    images: np.ndarray, sigma: float = 3.0, window_size: int = 33
) -> np.ndarray:
    """
    根据3sigma法则检测孤立点认为为盲元
    使用局部窗口计算均值和标准差，添加padding优化边界处理

    参数:
    - images: 图像数组 (num_temperatures, height, width)
    - window_size: 窗口大小
    - sigma: 阈值系数

    返回:
    - 尺寸为(height, width)的布尔数组 True表示对应位置是盲元
    """

    def local_mean(window):
        return np.mean(window)

    def local_std(window):
        return np.std(window)

    height, width = images.shape[1], images.shape[2]
    bps = np.zeros((height, width), dtype=bool)

    # 确保window_size是奇数
    if window_size % 2 == 0:
        window_size += 1

    # 计算padding大小
    pad_size = window_size // 2

    for image in images:
        # 添加padding，使用reflect模式处理边界
        padded_image = np.pad(image, pad_size, mode='reflect')
        
        # 使用generic_filter计算局部均值和标准差
        local_means = generic_filter(padded_image, local_mean, size=window_size)
        local_stds = generic_filter(padded_image, local_std, size=window_size)
        
        # 移除padding，恢复到原始尺寸
        local_means = local_means[pad_size:pad_size+height, pad_size:pad_size+width]
        local_stds = local_stds[pad_size:pad_size+height, pad_size:pad_size+width]

        # 计算阈值范围
        lower_bound = local_means - sigma * local_stds
        upper_bound = local_means + sigma * local_stds

        # 找出超出阈值范围的像素
        blind_pixels = (image < lower_bound) | (image > upper_bound)

        # 更新盲元索引，只要在任意一幅图像中是盲元，就标记为True
        bps = np.logical_or(bps, blind_pixels)

    return bps


def remove_bps(raw_arr: np.ndarray, bps: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    去除盲元
    直接中值滤波：先对整幅图做 medianBlur 然后仅在坏点处回填。

    - raw_arr: 单通道灰度 uint16
    - ksize: 中值核大小，必须为奇数，建议 3/5/7/...越大越平滑

    返回:
    - 去除盲元后的数组 uint16

    """

    k = int(ksize)

    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1

    blurred = cv2.medianBlur(raw_arr, k)

    raw_arr[bps] = blurred[bps]

    return raw_arr.astype(np.uint16)


if __name__ == "__main__":
    path = "D:/Projects/2025/145corr/nuc/tanceqi/swir/swir/short_black_image"
    save_path = "output"
    os.makedirs(save_path, exist_ok=True)

    temps = os.listdir(path)

    for temp in temps:

        print(f"Processing {temp} C...")

        its = os.listdir(os.path.join(path, temp))

        for it in its:

            print(f"Processing {it}...")

            bp_save_path = os.path.join(save_path, "bp", "swir", temp, it)
            os.makedirs(bp_save_path, exist_ok=True)

            bp_file = os.path.join(bp_save_path, "blind_pixels.npz")

            bad_pixel_vis_path = os.path.join(save_path, "validation", "swir", temp, it)
            os.makedirs(bad_pixel_vis_path, exist_ok=True)

            imgs_names = os.listdir(os.path.join(path, temp, it))
            imgs_names = imgs_names[:30]

            imgs_names.sort()

            imgs = []
            for img_name in imgs_names:
                img = cv2.imread(os.path.join(path, temp, it, img_name), -1)
                imgs.append(img)

            imgs = np.array(imgs)

            bp_map = detect_bp_by_3sigma_generic_filter(imgs, sigma=3.0, window_size=47)
            np.savez_compressed(bp_file, blind=bp_map.astype(bool))

            img_med = imgs[0]
            
            img_med = img_med / 4095 * 255.0
            img_rgb = cv2.cvtColor(img_med.astype(np.uint8), cv2.COLOR_GRAY2RGB)

            img_rgb[bp_map] = [255, 0, 0]

            bad_pixel_img_path = os.path.join(bad_pixel_vis_path, f"bad_pixels_{it}ms.png")
            cv2.imwrite(bad_pixel_img_path, img_rgb)

            print(f"Processed {it}ms")
        
        print(f"Processed {temp} C")
    
    print("Done")



            