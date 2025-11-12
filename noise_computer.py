import os
from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def compute_noise_statistics(VDS_T0):
    """
    使用numpy计算图像序列的噪声统计信息
    参数：
        VDS_T0: 图像数据数组，shape=(F, M, N)
    返回：
        std_mean: 所有像元的标准差平均值
        mean_mean: 所有像元的均值平均值
        std_map: 每个像元的标准差图，shape=(M, N)
        mean_map: 每个像元的均值图，shape=(M, N)
    """
    imgs = np.array(VDS_T0)
    
    # 计算每个像元在时间维度上的标准差和均值
    std_map = np.std(imgs, ddof=1, axis=0)  # shape=(M, N)
    mean_map = np.mean(imgs, axis=0)       # shape=(M, N)
    
    # 计算所有像元的平均值
    std_mean = np.mean(std_map)
    mean_mean = np.mean(mean_map)
    
    return std_mean, mean_mean, std_map, mean_map

def compute_VDS_T0_mean(VDS_T0):
    """
    计算VDS_T0在时间维度上的均值
    参数：
        VDS_T0: 图像数据数组，shape=(F, M, N)
    返回：
        mean_VDS_T0: 每个像元的均值图，shape=(M, N)
    """
    mean_VDS_T0 = np.mean(VDS_T0, axis=0)
    return mean_VDS_T0

def plot_mean_grayscale_3d(mean_VDS_T0, save_path=None, downsample_factor=1):
    """
    绘制均值灰度三维分布图
    参数：
        mean_VDS_T0: 均值图，shape=(M, N)
        save_path: 保存路径（可选），如果为None则显示图像
        downsample_factor: 降采样因子，用于减少数据点数量（默认1，不降采样）
    """
    # 降采样以加快绘制速度（如果图像很大）
    if downsample_factor > 1:
        mean_VDS_T0 = mean_VDS_T0[::downsample_factor, ::downsample_factor]
    
    M, N = mean_VDS_T0.shape
    
    # 创建坐标网格
    x = np.arange(N)
    y = np.arange(M)
    X, Y = np.meshgrid(x, y)
    
    # 创建3D图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制3D表面图
    surf = ax.plot_surface(X, Y, mean_VDS_T0, cmap='viridis', 
                          linewidth=0, antialiased=True, alpha=0.9)
    
    # 设置标签
    ax.set_xlabel('X坐标 (像素)', fontsize=12)
    ax.set_ylabel('Y坐标 (像素)', fontsize=12)
    ax.set_zlabel('灰度均值', fontsize=12)
    ax.set_title('VDS_T0均值灰度三维分布', fontsize=14, fontweight='bold')
    
    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='灰度值')
    
    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='计算像元噪声电压')
    parser.add_argument('--folder_path', type=str, 
                       default="./wave/50/2500",
                       help='图像文件夹路径')
    parser.add_argument('--frames', type=int, default=30, 
                       help='预期帧数（默认30）')
    parser.add_argument('--gain', type=float, default=1, 
                       help='系统增益（默认1）')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    folder_path = args.folder_path
    expected_frames = args.frames
    gain = args.gain
    
    print(f"正在从 {folder_path} 加载图像数据...")
    VDS_T0 = load_image_folder(folder_path, expected_frames=expected_frames)
    
    # 使用封装的函数计算噪声电压
    VN = compute_pixel_noise_voltage(VDS_T0, gain)
    print(f"ymr --> VN mean: {VN.mean()}")

    # 使用封装的函数计算噪声统计信息
    std_mean, mean_mean, std_map, mean_map = compute_noise_statistics(VDS_T0)
    print(f"numpy --> std_mean: {std_mean}, mean_mean: {mean_mean}")