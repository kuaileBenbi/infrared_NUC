import os
from PIL import Image
import numpy as np

def load_image_folder(folder_path, expected_frames=100):
    """
    从文件夹加载PNG图像序列作为numpy数组
    参数：
        folder_path: 文件夹路径
        expected_frames: 预期的帧数
    返回：
        图像数据数组，shape=(F, M, N)
    """
    # 获取文件夹中所有PNG文件并按名称排序
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    if len(files) != expected_frames:
        print(f"警告: 文件夹{folder_path}中找到{len(files)}个文件，但预期{expected_frames}个")

    # 读取第一张图片获取尺寸
    first_img = np.array(Image.open(os.path.join(folder_path, files[0])))
    M, N = first_img.shape

    # 初始化数组
    data = np.zeros((len(files), M, N), dtype=np.float32)

    # 逐帧加载
    for i, filename in enumerate(files):
        img = Image.open(os.path.join(folder_path, filename))
        data[i] = np.array(img,dtype=np.float32)

    return data

def compute_pixel_noise_voltage(VDS_T0, gain):
    """
    计算像元噪声电压 VN
    根据公式(9)： VN(i,j) = sqrt( 1/(F-1) * Σ_{f=1}^{F} (VDS[(i,j),T0,f] - mean(VDS[(i,j),T0]) )^2 ) / gain
    参数：
        VDS_T0: 在温度 T0 条件下采集的 F 帧二维数据，shape=(F, M, N)
        gain: 系统增益K（放大倍数）
    返回：
        VN: 每个像元的噪声电压数组，shape=(M, N)
    """
    mean_VDS_T0 = np.mean(VDS_T0, axis=0)
    variance = np.sum((VDS_T0 - mean_VDS_T0) ** 2, axis=0) / (VDS_T0.shape[0] - 1)
    VN = np.sqrt(variance) / gain
    return VN

if __name__ == "__main__":
    # 示例用法
    folder_path = "D:/Projects/2025/145corr/nuc/biaoding/jingtou/0829_corr_guanglutiaozheng/0829_daijingtoubiaoding/lwir/50/2500"
    # sensor = "D:\\Projects\\2025\\145corr\\nuc\\biaoding\\tanceqi\\0902_lwir_corr_14bit_gaintiaozheng\\lwir\\40du\\2500"
    VDS_T0 = load_image_folder(folder_path, expected_frames=30)
    gain = 1  # 示例增益值
    VN = compute_pixel_noise_voltage(VDS_T0, gain)
    
    print("Computed VN shape:", VN, VN.mean())

    imgs = np.array(VDS_T0)
    std = np.std(imgs, ddof=1, axis=0)
    mean = np.mean(imgs, axis=0)
    std = np.mean(std)
    mean = np.mean(mean)
    print(f"std: {std}, mean: {mean}")

    # VDS_T0 = load_image_folder(sensor, expected_frames=30)
    # gain = 1  # 示例增益值
    # VN = compute_pixel_noise_voltage(VDS_T0, gain)
    
    # print("Computed VN shape:", VN, VN.mean())

    # imgs = np.array(VDS_T0)
    # std = np.std(imgs, ddof=1, axis=0)
    # mean = np.mean(imgs, axis=0)
    # std = np.mean(std)
    # mean = np.mean(mean)
    # print(f"std: {std}, mean: {mean}")