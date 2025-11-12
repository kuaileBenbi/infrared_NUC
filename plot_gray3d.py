import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import font_manager
import cv2

# 设置matplotlib中文字体支持
def setup_chinese_font():
    """设置matplotlib中文字体，避免中文显示为方块"""
    # Windows系统常见中文字体列表
    font_candidates = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'STSong']
    
    # 尝试设置可用的中文字体
    for font_name in font_candidates:
        try:
            # 检查字体是否可用
            available_fonts = [f.name for f in font_manager.fontManager.ttflist]
            if font_name in available_fonts:
                matplotlib.rcParams['font.sans-serif'] = [font_name] + matplotlib.rcParams['font.sans-serif']
                break
        except:
            continue
    else:
        # 如果都不可用，使用默认列表
        matplotlib.rcParams['font.sans-serif'] = font_candidates + matplotlib.rcParams['font.sans-serif']
    
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 初始化中文字体
setup_chinese_font() 


def load_image_folder(folder_path, expected_frames=100):
    """
    从文件夹或单个文件加载PNG图像作为numpy数组
    参数：
        folder_path: 文件夹路径或单个图像文件路径
        expected_frames: 预期的帧数（仅当folder_path是文件夹时有效）
    返回：
        图像数据数组，shape=(F, M, N) 或 shape=(1, M, N)（单个文件时）
    """
    # 判断是文件还是文件夹
    if os.path.isfile(folder_path):
        # 如果是单个文件，直接加载
        print(f"加载单个图像文件: {folder_path}")
        img = Image.open(folder_path)
        img_array = np.array(img, dtype=np.float32)
        # 添加时间维度，保持shape一致性
        data = img_array[np.newaxis, :, :]  # shape=(1, M, N)
        return data
    elif os.path.isdir(folder_path):
        # 如果是文件夹，按原来的逻辑处理
        # 获取文件夹中所有PNG文件并按名称排序
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        if len(files) == 0:
            raise ValueError(f"文件夹{folder_path}中未找到PNG文件")
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
            data[i] = np.array(img, dtype=np.float32)

        return data
    else:
        raise ValueError(f"路径不存在: {folder_path}")


def compute_VDS_T0_mean(VDS_T0):
    """
    计算VDS_T0在时间维度上的均值，如果是单个图像则直接返回
    参数：
        VDS_T0: 图像数据数组，shape=(F, M, N) 或 shape=(1, M, N) 或 shape=(M, N)
    返回：
        mean_VDS_T0: 每个像元的均值图，shape=(M, N)
    """
    # 判断输入维度
    if VDS_T0.ndim == 2:
        # 如果是2D数组（单个图像），直接返回
        print(f"输入为单个图像，直接输出")
        print(f"mean_value = {VDS_T0.mean()}")
        return VDS_T0
    elif VDS_T0.ndim == 3:
        # 如果是3D数组（图像序列）
        if VDS_T0.shape[0] == 1:
            # 如果只有一帧，直接返回该帧
            print(f"输入为单帧图像，直接输出")
            mean_VDS_T0 = VDS_T0[0]
        else:
            # 多帧图像，计算均值
            mean_VDS_T0 = np.mean(VDS_T0, axis=0)
        print(f"mean_value = {mean_VDS_T0.mean()}")
        return mean_VDS_T0
    else:
        raise ValueError(f"不支持的数组维度: {VDS_T0.ndim}，期望2D或3D数组")

def quantile_stretch(image, lower_percentile=0.01, upper_percentile=0.99, output_min=0, output_max=4095):
    """
    分位数拉伸函数
    参数：
        image: 输入图像数组，shape=(M, N)
        lower_percentile: 低分位数（默认0.01，表示1%）
        upper_percentile: 高分位数（默认0.99，表示99%）
        output_min: 输出最小值（默认0）
        output_max: 输出最大值（默认4095）
    返回：
        stretched_image: 拉伸后的图像，shape=(M, N)，dtype=np.float32
    """
    # 将小数形式转换为百分比形式（如果小于1，则乘以100）
    image = image.astype(np.float32)
    if lower_percentile < 1:
        lower_percentile_pct = lower_percentile * 100
    else:
        lower_percentile_pct = lower_percentile
    
    if upper_percentile < 1:
        upper_percentile_pct = upper_percentile * 100
    else:
        upper_percentile_pct = upper_percentile
    
    # 计算分位数
    lower_bound = np.percentile(image, lower_percentile_pct)
    upper_bound = np.percentile(image, upper_percentile_pct)
    
    # 避免除零
    if upper_bound == lower_bound:
        # 如果上下界相同，返回均匀值
        stretched_image = np.full_like(image, output_min, dtype=np.uint16)
    else:
        # 线性拉伸
        stretched_image = imadjust_vec(image, lower_bound, upper_bound, 0, output_max, gamma=1.0)
        print(f"stretched_image: {stretched_image.min()}, {stretched_image.max()}")
        np.clip(stretched_image, 0, output_max, out=stretched_image)
        stretched_image = stretched_image.astype(np.uint16)
    return stretched_image

def imadjust_vec(x, a, b, c, d, gamma=1.0):
    y = (np.clip((x - a) / (b - a), 0, 1) ** gamma) * (d - c) + c
    return y

def plot_mean_grayscale_3d(mean_VDS_T0, save_path=None, downsample_factor=1, 
                          apply_stretch=False, lower_percentile=0.01, upper_percentile=0.99, output_max=4095):
    """
    绘制均值灰度三维分布图
    参数：
        mean_VDS_T0: 均值图，shape=(M, N)
        save_path: 保存路径（可选），如果为None则显示图像
        downsample_factor: 降采样因子，用于减少数据点数量（默认1，不降采样）
        apply_stretch: 是否应用分位数拉伸（默认False）
        lower_percentile: 分位数拉伸的低分位数（默认0.01，表示1%）
        upper_percentile: 分位数拉伸的高分位数（默认0.99，表示99%）
    """
    # 保存原始数据用于2D显示
    mean_VDS_T0_original = mean_VDS_T0.copy()
    
    # 应用分位数拉伸（如果启用）
    if apply_stretch:
        mean_VDS_T0_stretched = quantile_stretch(mean_VDS_T0_original, 
                                                 lower_percentile=lower_percentile,
                                                 upper_percentile=upper_percentile,
                                                 output_max=output_max)
        print(f"分位数拉伸: {lower_percentile*100:.2f}%-{upper_percentile*100:.2f}%")
        print(f"拉伸前范围: [{mean_VDS_T0_original.min():.2f}, {mean_VDS_T0_original.max():.2f}]")
        print(f"拉伸后范围: [{mean_VDS_T0_stretched.min():.2f}, {mean_VDS_T0_stretched.max():.2f}]")
    else:
        mean_VDS_T0_stretched = None
    
    # 降采样以加快绘制速度（如果图像很大）
    if downsample_factor > 1:
        mean_VDS_T0 = mean_VDS_T0[::downsample_factor, ::downsample_factor]
        if apply_stretch:
            mean_VDS_T0_stretched = mean_VDS_T0_stretched[::downsample_factor, ::downsample_factor]
    
    M, N = mean_VDS_T0.shape
    
    # 创建坐标网格
    x = np.arange(N)
    y = np.arange(M)
    X, Y = np.meshgrid(x, y)
    
    # 根据是否应用拉伸决定子图数量
    if apply_stretch:
        # 创建图形，包含四个子图（2x2布局）
        fig = plt.figure(figsize=(20, 12))
        
        # 保存拉伸后的原始图像用于2D显示（不降采样）
        mean_VDS_T0_stretched_original = quantile_stretch(mean_VDS_T0_original, 
                                                           lower_percentile=lower_percentile,
                                                           upper_percentile=upper_percentile,
                                                           output_max=output_max)
        
        # 左上子图：显示原图（2D灰度图）
        ax1 = fig.add_subplot(221)
        im1 = ax1.imshow(mean_VDS_T0_original, cmap='gray', vmin=mean_VDS_T0_original.min(), vmax=mean_VDS_T0_original.max())
        ax1.set_xlabel('X坐标 (像素)', fontsize=12)
        ax1.set_ylabel('Y坐标 (像素)', fontsize=12)
        ax1.set_title('VDS_T0均值灰度原图', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='灰度值')
        
        # 右上子图：显示拉伸后的2D灰度图
        ax2 = fig.add_subplot(222)
        im2 = ax2.imshow(mean_VDS_T0_stretched_original, cmap='gray', 
                         vmin=mean_VDS_T0_stretched_original.min(), 
                         vmax=mean_VDS_T0_stretched_original.max())
        ax2.set_xlabel('X坐标 (像素)', fontsize=12)
        ax2.set_ylabel('Y坐标 (像素)', fontsize=12)
        ax2.set_title(f'VDS_T0均值灰度原图（分位数拉伸 {lower_percentile*100:.1f}%-{upper_percentile*100:.1f}%）', 
                     fontsize=14, fontweight='bold')
        plt.colorbar(im2, ax=ax2, label='灰度值')
        
        # 左下子图：显示原图3D分布图
        ax3 = fig.add_subplot(223, projection='3d')
        surf = ax3.plot_surface(X, Y, mean_VDS_T0, cmap='viridis', 
                              linewidth=0, antialiased=True, alpha=0.9)
        ax3.set_xlabel('X坐标 (像素)', fontsize=12)
        ax3.set_ylabel('Y坐标 (像素)', fontsize=12)
        ax3.set_zlabel('灰度均值', fontsize=12)
        ax3.set_title('VDS_T0均值灰度三维分布（原始）', fontsize=14, fontweight='bold')
        fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=5, label='灰度值')
        ax3.view_init(elev=30, azim=45)
        
        # 右下子图：显示拉伸后3D分布图
        ax4 = fig.add_subplot(224, projection='3d')
        surf_stretched = ax4.plot_surface(X, Y, mean_VDS_T0_stretched, cmap='viridis', 
                                         linewidth=0, antialiased=True, alpha=0.9)
        ax4.set_xlabel('X坐标 (像素)', fontsize=12)
        ax4.set_ylabel('Y坐标 (像素)', fontsize=12)
        ax4.set_zlabel('灰度均值（拉伸后）', fontsize=12)
        ax4.set_title(f'VDS_T0均值灰度三维分布（分位数拉伸 {lower_percentile*100:.1f}%-{upper_percentile*100:.1f}%）', 
                     fontsize=14, fontweight='bold')
        fig.colorbar(surf_stretched, ax=ax4, shrink=0.5, aspect=5, label='灰度值')
        ax4.view_init(elev=30, azim=45)
    else:
        # 创建图形，包含两个子图（原始版本）
        fig = plt.figure(figsize=(16, 8))
        
        # 左侧子图：显示原图（2D灰度图）
        ax1 = fig.add_subplot(121)
        im1 = ax1.imshow(mean_VDS_T0_original, cmap='gray', vmin=mean_VDS_T0_original.min(), vmax=mean_VDS_T0_original.max())
        ax1.set_xlabel('X坐标 (像素)', fontsize=12)
        ax1.set_ylabel('Y坐标 (像素)', fontsize=12)
        ax1.set_title('VDS_T0均值灰度原图', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='灰度值')
        
        # 右侧子图：显示3D分布图
        ax2 = fig.add_subplot(122, projection='3d')
        surf = ax2.plot_surface(X, Y, mean_VDS_T0, cmap='viridis', 
                              linewidth=0, antialiased=True, alpha=0.9)
        ax2.set_xlabel('X坐标 (像素)', fontsize=12)
        ax2.set_ylabel('Y坐标 (像素)', fontsize=12)
        ax2.set_zlabel('灰度均值', fontsize=12)
        ax2.set_title('VDS_T0均值灰度三维分布', fontsize=14, fontweight='bold')
        fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='灰度值')
        ax2.view_init(elev=30, azim=45)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='绘制均值灰度三维分布图')
    parser.add_argument('--folder_path', type=str, default="./wave/50/2500", help='文件夹路径')
    parser.add_argument('--expected_frames', type=int, default=100, help='预期帧数')
    parser.add_argument('--save_path', type=str, default=None, help='保存路径')
    parser.add_argument('--apply_stretch', action='store_true', help='是否应用分位数拉伸')
    parser.add_argument('--lower_percentile', type=float, default=0.01, help='分位数拉伸的低分位数（默认2）')
    parser.add_argument('--upper_percentile', type=float, default=0.99, help='分位数拉伸的高分位数（默认98）')
    parser.add_argument('--downsample_factor', type=int, default=1, help='降采样因子（默认1，不降采样）')
    parser.add_argument('--output_max', type=int, default=4095, help='输出最大值（默认4095）')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    folder_path = args.folder_path
    expected_frames = args.expected_frames
    save_path = args.save_path
    VDS_T0 = load_image_folder(folder_path, expected_frames)
    mean_VDS_T0 = compute_VDS_T0_mean(VDS_T0)
    plot_mean_grayscale_3d(mean_VDS_T0, save_path=save_path, 
                          downsample_factor=args.downsample_factor,
                          apply_stretch=args.apply_stretch,
                          lower_percentile=args.lower_percentile,
                          upper_percentile=args.upper_percentile,
                          output_max=args.output_max)