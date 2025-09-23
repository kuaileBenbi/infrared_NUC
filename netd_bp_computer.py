import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import argparse
def compute_pixel_response_voltage(VDS_T, VDS_T0, gain):
    """
    计算像元响应电压 VS
    根据公式(3)： VS(i,j) = [mean(VDS[(i,j),T] ) - mean(VDS[(i,j),T0])] / gain
    参数：
        VDS_T: 在温度 T 条件下采集的 F 帧二维数据，shape=(F, M, N)
        VDS_T0: 在温度 T0 条件下采集的 F 帧二维数据，shape=(F, M, N)
        gain: 系统增益K（放大倍数）
    返回：
        VS: 每个像元的响应电压数组，shape=(M, N)
    """
    mean_VDS_T = np.mean(VDS_T, axis=0)
    mean_VDS_T0 = np.mean(VDS_T0, axis=0)
    VS = (mean_VDS_T - mean_VDS_T0) / gain
    return VS


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


def compute_effective_pixel_mask(VS, VN, P):
    """
    根据标准4.5的定义，判断死像元和过热像元，返回一个有效像元的掩码
    死像元定义：像元响应电压 VS < 1/2 * (全体像元平均 VS)
    过热像元定义：对于非死像元，噪声电压 VN > 2 * (平均 VN of 非死像元)
    返回：
         effective_mask: 布尔型数组，True 表示该像元为有效像元
         dead_mask: 布尔型数组，True 表示为死像元
         overhot_mask: 布尔型数组，True 表示为过热像元
    """
    # 计算全体像元的平均响应电压
    R=VS / P
    mean_R_all = np.mean(R)
    # 死像元：响应电压小于全体平均的一半
    dead_mask = R < (0.5 * mean_R_all)

    # 对非死像元计算平均噪声电压
    VN_valid = VN[~dead_mask]
    if VN_valid.size == 0:
        avg_VN_valid = 0
    else:
        avg_VN_valid = np.mean(VN_valid)
    # 过热像元：噪声电压大于 2 倍非死像元的平均噪声电压
    overhot_mask = VN > (2 * avg_VN_valid)

    # 有效像元为既不为死像元也不为过热像元
    effective_mask = ~(dead_mask | overhot_mask)
    return effective_mask, dead_mask, overhot_mask


def compute_netd(VS, VN, T, T0):
    """
    计算每个像元的噪声等效温差 (NETD)
    根据公式(14)： NETD(i,j) = (T - T0) * (VN(i,j) / VS(i,j))
    注意：在计算时需防止除以零
    参数：
        VS: 每个像元响应电压数组，shape=(M, N)
        VN: 每个像元噪声电压数组，shape=(M, N)
        T: 黑体温度T（单位：K）
        T0: 黑体温度T0（单位：K）
    返回：
        NETD: 每个像元的噪声等效温差数组，shape=(M, N)
    """
    # 防止除以零，将零值位置赋予一个很小的数
    VS_safe = np.where(VS == 0, 1e-10, VS)
    NETD = (T - T0) * (VN / VS_safe)
    return NETD


def compute_average_netd(NETD, effective_mask):
    """
    计算所有有效像元的平均噪声等效温差
    参数：
        NETD: 每个像元的噪声等效温差数组
        effective_mask: 布尔型数组，True 表示该像元为有效像元
    返回：
        avg_NETD: 有效像元的平均 NETD
    """
    if np.any(effective_mask):
        avg_NETD = np.mean(NETD[effective_mask])
    else:
        avg_NETD = np.nan
    return avg_NETD


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
        data[i] = np.array(img,dtype=np.float32)/4096

    return data


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='计算NETD（噪声等效温差）')
    parser.add_argument('--vds_t0_folder', type=str, default="./netd_v3/10/3ms", 
                       help='VDS_T0文件夹路径（温度T0条件下的数据）')
    parser.add_argument('--vds_t_folder', type=str, default="./netd_v3/25/3ms", 
                       help='VDS_T文件夹路径（温度T条件下的数据）')
    parser.add_argument('--frames', type=int, default=100, 
                       help='预期帧数（默认100）')
    parser.add_argument('--gain', type=float, default=2/3, 
                       help='系统增益（默认2/3）')
    parser.add_argument('--t0', type=float, default=278, 
                       help='黑体温度T0（默认278K）')
    parser.add_argument('--t', type=float, default=288, 
                       help='黑体温度T（默认288K）')
    parser.add_argument('--ad', type=float, default=(30*1e-4)**2, 
                       help='像元面积，单位：cm^2（默认(30*1e-4)^2）')
    parser.add_argument('--ld', type=float, default=2, 
                       help='LD参数（默认2）')
    parser.add_argument('--n', type=float, default=1, 
                       help='n参数（默认1）')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 从命令行参数或默认值设置参数
    F = args.frames  # 帧数
    gain = args.gain  # 系统增益（放大倍数）
    T0 = args.t0  # 黑体温度T0
    T = args.t  # 黑体温度T

    AD = args.ad  # 像元面积，单位：cm^2
    LD = args.ld
    n = args.n
    # 辐照功率插值
    P = 5.673e-12*(T**4-T0**4)*AD/(4*LD**2+n)  # 辐照功率差，单位：W/m^2

    # 从命令行参数获取文件夹路径
    vds_t0_folder = args.vds_t0_folder
    vds_t_folder = args.vds_t_folder

    print(f"正在从{vds_t0_folder}加载VDS_T0数据...")
    VDS_T0 = load_image_folder(vds_t0_folder, expected_frames=F)

    print(f"正在从{vds_t_folder}加载VDS_T数据...")
    VDS_T = load_image_folder(vds_t_folder, expected_frames=F)

    # 获取图像尺寸
    M, N = VDS_T0.shape[1], VDS_T0.shape[2]
    print(f"加载完成，图像尺寸: {M}x{N}, 帧数: {F}")

    # 计算响应电压 VS 和噪声电压 VN
    VS = compute_pixel_response_voltage(VDS_T, VDS_T0, gain)
    VN = compute_pixel_noise_voltage(VDS_T0, gain)

    # 根据 VS 和 VN 计算有效像元掩码
    effective_mask, dead_mask, overhot_mask = compute_effective_pixel_mask(VS, VN, P)

    # 输出死像元、过热像元及有效像元率
    total_pixels = M * N
    dead_count = np.sum(dead_mask)
    overhot_count = np.sum(overhot_mask)
    effective_count = np.sum(effective_mask)
    effective_rate = effective_count / total_pixels * 100
    print("总像元数: {}".format(total_pixels))
    print("死像元数: {}".format(dead_count))
    print("过热像元数: {}".format(overhot_count))
    print("有效像元数: {} ({:.2f}%)".format(effective_count, effective_rate))

    # 计算每个像元的 NETD
    NETD = compute_netd(VS, VN, T, T0)

    # 仅对有效像元计算平均 NETD
    avg_NETD = compute_average_netd(NETD, effective_mask)
    print("有效像元平均 NETD: {:.4f} K".format(avg_NETD))
    
    # 创建掩码图像（有效像元=0，其他=1）
    mask_to_save = np.where(effective_mask, 0, 1).astype(bool)
    final_mask = {
        "blind": mask_to_save,
    }
    output_path = "mask.npz"
    np.savez(output_path,  ** final_mask)
    print(f"组合掩码已保存至 {output_path}")
