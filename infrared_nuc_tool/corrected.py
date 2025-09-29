import argparse
import os
import cv2
import numpy as np
import tqdm
import logging

from utils.nuc_tool import *
from utils.bp_tool import *
from utils.pre_tool import *

if not os.path.exists("log"):
    os.makedirs("log", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="log/corrected.log",
    filemode="w",
    encoding="utf-8",
    force=True,
)

logger = logging.getLogger(__name__)


# python .\corrected.py --data_path D:\Projects\2025\145corr\nuc\tanceqi\0902_lwir_corr_14bit_gaintiaozheng --exposure_minvalue 1500 --exposure_maxvalue 3500 --exposure_step 100 --temperatures_minvalue 4 --temperatures_maxvalue 80 --temperatures_step 2 --wave lwir --bit_max 16383


def run_pipeline(
    data_path, wave, temperatures_list, exposure_list, bit_max, output_path, keep_ratio
):

    for exposure in tqdm.tqdm(exposure_list, desc="Processing exposure"):

        logger.info(f"Processing exposure: {exposure/5000:.2f} ms")

        mean_y_by_temp: dict[float, float] = {}  # key: 温度, value: 理想灰度值
        mean_y_by_gray: dict[int, float] = {}  # key: 理想灰度值, value: 温度
        images_by_temp: dict[float, np.ndarray] = (
            {}
        )  # key: 温度, value: 某个温度下100张图像的时域均值

        temps_ok: list[float] = []
        grays_ok: list[float] = []

        nuc_dir = (
            f"corr_{exposure//5000}ms_multi_calibrate_10_20_30"
            if wave == "mwir"
            else f"corr_{exposure/5000:.2f}ms_multi_calibrate_10_20_30"
        )
        bp_dir = f"{exposure//5000}ms" if wave == "mwir" else f"{exposure/5000:.2f}ms"

        linear_nuc_path = os.path.join(output_path, "nuc", wave, "linear", nuc_dir)
        quadrast_nuc_path = os.path.join(output_path, "nuc", wave, "quadrast", nuc_dir)
        os.makedirs(linear_nuc_path, exist_ok=True)
        os.makedirs(quadrast_nuc_path, exist_ok=True)
        linear_nuc_validation_path = os.path.join(
            output_path, "validation", wave, "linear", nuc_dir
        )
        quadrast_nuc_validation_path = os.path.join(
            output_path, "validation", wave, "quadrast", nuc_dir
        )
        os.makedirs(linear_nuc_validation_path, exist_ok=True)
        os.makedirs(quadrast_nuc_validation_path, exist_ok=True)

        # 创建坏点可视化输出目录
        bad_pixel_vis_path = os.path.join(output_path, "validation", wave, "bad_pixel_visualization", bp_dir)
        os.makedirs(bad_pixel_vis_path, exist_ok=True)
        bad_pixel_mask_path = os.path.join(output_path, "bp", wave, bp_dir)
        os.makedirs(bad_pixel_mask_path, exist_ok=True)

        # 收集所有温度的图像数据用于坏点检测
        all_temp_images = []
        valid_temperatures = []

        for temperature in tqdm.tqdm(temperatures_list, desc="Processing temperature"):

            cur_temp_path = os.path.join(data_path, f"{temperature}du")

            if not os.path.exists(cur_temp_path):
                logger.info(f"Warning: {cur_temp_path} not found!")
                continue

            cur_it_temp_path = os.path.join(cur_temp_path, f"{exposure}")

            if not os.path.exists(cur_it_temp_path):
                logger.info(f"Warning: {cur_it_temp_path} not found!")
                continue

            img_med, val = load_median_image_cv2(cur_it_temp_path)

            if img_med is None or val is None:
                logger.info(f"Warning: {cur_it_temp_path} read mean value failed!")
                continue

            # if wave == "lwir" and val > 2000 and val < 12000:

            #     images_by_temp[temperature] = img_med
            #     mean_y_by_temp[temperature] = val
            #     all_temp_images.append(img_med)
            #     valid_temperatures.append(temperature)

            #     # 不用浮点作键，量化一下
            #     mean_y_by_gray[gray_key(val, step=1.0)] = temperature

            #     temps_ok.append(temperature)
            #     grays_ok.append(val)

            images_by_temp[temperature] = img_med
            mean_y_by_temp[temperature] = val
            all_temp_images.append(img_med)
            valid_temperatures.append(temperature)

            # 不用浮点作键，量化一下
            mean_y_by_gray[gray_key(val, step=1.0)] = temperature

            temps_ok.append(temperature)
            grays_ok.append(val)

        # 计算坏点（所有温度坏点的并集）
        if len(all_temp_images) > 0:
            logger.info(f"检测坏点中，共{len(all_temp_images)}个温度点...")
            all_temp_images_array_raw = np.array(all_temp_images)  # (num_temps, height, width)

            all_temp_images_array = all_temp_images_array_raw[::10]

            if len(all_temp_images_array) == 0:
                all_temp_images_array = all_temp_images_array_raw
            else:
                sampled_valid_temperatures = valid_temperatures[::5]
                sampled_all_temp_images = all_temp_images[::5]
            
            # 使用3sigma方法检测坏点
            bad_pixel_map = detect_bp_by_3sigma_generic_filter(all_temp_images_array, sigma=3.0, window_size=88)
            
            # 计算坏点统计
            total_pixels = bad_pixel_map.size
            bad_pixel_count = np.sum(bad_pixel_map)
            bad_pixel_ratio = bad_pixel_count / total_pixels * 100
            
            logger.info(f"坏点检测完成: {bad_pixel_count}/{total_pixels} ({bad_pixel_ratio:.2f}%)")
            
            # 为每个温度生成带坏点标记的中值图
            # 使用相同的采样间隔
            # sampled_valid_temperatures = valid_temperatures[::5]
            # sampled_all_temp_images = all_temp_images[::5]
            
            for i, temperature in enumerate(sampled_valid_temperatures):
                
                if i == len(sampled_valid_temperatures) // 2:
                    img_med = sampled_all_temp_images[i]
                    
                    # 创建RGB图像用于标记坏点
                    if len(img_med.shape) == 2:
                        # 灰度图转RGB
                        img_med = img_med / bit_max * 255.0
                        img_rgb = cv2.cvtColor(img_med.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                    else:
                        img_rgb = img_med.copy()
                        img_rgb = img_rgb / bit_max * 255.0
                    
                    # 将坏点标记为红色
                    img_rgb[bad_pixel_map] = [255, 0, 0]  # 红色通道
                    
                    # 保存带坏点标记的图像
                    bad_pixel_img_path = os.path.join(bad_pixel_vis_path, f"bad_pixels_{temperature}du.png")
                    cv2.imwrite(bad_pixel_img_path, img_rgb)
                
            # 保存坏点掩码
            bad_pixel_mask = os.path.join(bad_pixel_mask_path, "blind_pixels.npz")
            np.savez_compressed(bad_pixel_mask, blind=bad_pixel_map.astype(bool))
            
            logger.info(f"坏点可视化图像已保存到: {bad_pixel_vis_path}")
        else:
            logger.warning("没有可用的图像数据用于坏点检测")
            bad_pixel_map = None

        # ---------------- 分数法确定校准点（左边界） ----------------
        grays_arr = np.asarray(grays_ok, dtype=np.float32)
        temps_arr = np.asarray(temps_ok)

        if len(grays_arr) == 0:
            logger.info("没有可用的平场样本。")
            break

        # 建议至少保留 3 个节点
        M_knots = max(3, int(len(grays_arr) * keep_ratio))
        qs = np.linspace(0.0, 1.0, M_knots)

        # 在“灰度中值”上做分位
        knot_vals = np.quantile(grays_arr, qs).astype(np.float32)

        # 排序（按灰度从低到高）
        order = np.argsort(grays_arr)
        grays_sorted = grays_arr[order]
        temps_sorted = temps_arr[order]

        # 用左边界选择：对每个分位灰度，取 >= 该值的最左索引
        idx = np.searchsorted(grays_sorted, knot_vals, side="left")
        idx = np.clip(idx, 0, len(grays_sorted) - 1)

        # 去重，避免多个分位落到同一索引
        idx_unique = np.unique(idx)
        selected_grays = grays_sorted[idx_unique]
        selected_temps = temps_sorted[idx_unique]

        train_imgs, train_temps = select_training_data(
            images_by_temp, knot_vals, mean_y_by_gray
        )

        valid_temps = [
            t for t in temperatures_list if t not in train_temps and t in images_by_temp
        ]

        bit_max = int(bit_max)

        logger.info(f"选择的温度与灰度值为：{selected_temps} {selected_grays}")

        # # ---------------- 线性校正 ----------------

        logger.info(f"线性校正...")

        method_out_dir = os.path.join(linear_nuc_path, f"multi_point_calib.npz")

        a_linear, b_linear, ga_linear, gb_linear = linear_fit(
            it_temps_arr=train_imgs,
            x_arr=selected_grays,
            bit_max=bit_max,
            save_path=method_out_dir,
        )

        if a_linear is None:
            logger.info(
                f"Warning: {wave}-{exposure//5000}ms-{temperature} C linear calibration failed!"
            )
            continue
        
        t = valid_temps[len(valid_temps) // 2]
        img = images_by_temp[t]
        img_corr = linear_apply(
            a_linear, b_linear, ga_linear, gb_linear, img, bit_max=bit_max
        )
        img_corr = (img_corr / img_corr.max()) * 255.0
        img_corr = img_corr.astype(np.uint8)
        img_corr = Image.fromarray(img_corr)
        img_corr.save(os.path.join(linear_nuc_validation_path, f"{t}.jpg"))
        # for t in valid_temps:
        #     img = images_by_temp[t]
        #     img_corr = linear_apply(
        #         a_linear, b_linear, ga_linear, gb_linear, img, bit_max=bit_max
        #     )
        #     img_corr = (img_corr / img_corr.max()) * 255.0
        #     img_corr = img_corr.astype(np.uint8)
        #     img_corr = Image.fromarray(img_corr)
        #     img_corr.save(os.path.join(linear_nuc_validation_path, f"{t}.jpg"))

        logger.info(f"线性校正完成！")

        # ---------------- 二次曲线校正 ----------------

        logger.info(f"二次曲线校正...")

        method_out_dir = os.path.join(quadrast_nuc_path, f"multi_point_calib.npz")

        a2_quad, a1_quad, a0_quad, _ = quadratic_fit(
            it_temps_arr=train_imgs,
            x_arr=selected_grays,
            bit_max=bit_max,
            save_path=method_out_dir,
        )
        t = valid_temps[len(valid_temps) // 2]
        img = images_by_temp[t]
        img_corr = quadratic_apply(a2_quad, a1_quad, a0_quad, img, bit_max=bit_max)
        img_corr = (img_corr / img_corr.max()) * 255.0
        img_corr = img_corr.astype(np.uint8)
        img_corr = Image.fromarray(img_corr)
        img_corr.save(os.path.join(quadrast_nuc_validation_path, f"{t}.jpg"))
        # for t in valid_temps:
        #     img = images_by_temp[t]
        #     img_corr = quadratic_apply(a2_quad, a1_quad, a0_quad, img, bit_max=bit_max)
        #     img_corr = (img_corr / img_corr.max()) * 255.0
        #     img_corr = img_corr.astype(np.uint8)
        #     img_corr = Image.fromarray(img_corr)
        #     img_corr.save(os.path.join(quadrast_nuc_validation_path, f"{t}.jpg"))

        logger.info(f"二次曲线校正完成！")


# ------------------------ CLI ------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--exposure_minvalue", type=int, default=100)
    parser.add_argument("--exposure_maxvalue", type=int, default=1000)
    parser.add_argument("--exposure_step", type=int, default=100)
    parser.add_argument("--temperatures_minvalue", type=int, default=10)
    parser.add_argument("--temperatures_maxvalue", type=int, default=100)
    parser.add_argument("--temperatures_step", type=int, default=5)
    parser.add_argument("--wave", type=str, default="mwir")
    parser.add_argument("--bit_max", type=int, default=4095)
    parser.add_argument(
        "--keep_ratio", type=float, default=0.6, help="分位点选择比例（≥3 个点）"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data_path
    wave = args.wave
    bit_max = int(args.bit_max)
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    exposure_list = list(
        range(args.exposure_minvalue, args.exposure_maxvalue, args.exposure_step)
    )
    temperatures_list = list(
        range(
            args.temperatures_minvalue,
            args.temperatures_maxvalue,
            args.temperatures_step,
        )
    )

    run_pipeline(
        data_path=data_path,
        wave=wave,
        temperatures_list=temperatures_list,
        exposure_list=exposure_list,
        bit_max=bit_max,
        output_path=output_path,
        keep_ratio=float(args.keep_ratio),
    )


if __name__ == "__main__":
    main()
