#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红外图像非均匀性与盲元校正验证工具

使用方法:
python validater.py --input_dir <图像目录> --method <校正方法> [其他选项]

支持的校正方法:
- linear: 线性校正
- quadratic: 二次校正
- bright_dark: 明暗场校正
- multi_point: 多点校正

示例:
python validater.py --input_dir ./data/test_images --method quadratic --bit_max 4095
python validater.py --input_dir ./data/test_images --method linear --calib_path ./output/nuc/calib.npz
"""

import argparse
import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import time
import json
from datetime import datetime

# 添加utils目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

try:
    from utils.nuc_tool import *
    from utils.bp_tool import *
    from utils.pre_tool import *
except ImportError as e:
    print(f"错误：无法导入工具模块 - {e}")
    print("请确保utils目录下包含nuc_tool.py, bp_tool.py, pre_tool.py文件")
    sys.exit(1)


class ImageValidator:
    """图像校正验证器"""

    def __init__(
        self,
        input_dir: str,
        method: str,
        output_dir: str = None,
        calib_path: str = None,
        bit_max: int = 4095,
        bad_pixel_path: str = None,
        log_level: str = "INFO",
    ):
        """
        初始化验证器

        :param input_dir: 输入图像目录
        :param method: 校正方法
        :param output_dir: 输出目录
        :param calib_path: 校正参数文件路径
        :param bit_max: 位宽上限
        :param bad_pixel_path: 盲元文件路径
        :param log_level: 日志级别
        """
        self.input_dir = Path(input_dir)
        self.method = method.lower()
        self.output_dir = Path(output_dir) if output_dir else Path("output/validation")
        self.calib_path = calib_path
        self.bit_max = bit_max
        self.bad_pixel_path = bad_pixel_path
        self.log_level = log_level

        # 支持的校正方法
        self.supported_methods = {
            "linear": self._linear_correction,
            "quadratic": self._quadratic_correction,
            "bright_dark": self._bright_dark_correction,
        }

        # 支持的图像格式
        self.supported_formats = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".tif",
            ".raw",
        }

        # 设置日志
        self._setup_logging()

        # 验证输入
        self._validate_inputs()

        # 加载校正参数
        self.calib_params = self._load_calibration_params()

        # 加载盲元信息
        self.bad_pixel_map = self._load_bad_pixel_map()

        # 统计信息
        self.stats = {
            "total_images": 0,
            "processed_images": 0,
            "failed_images": 0,
            "processing_time": 0.0,
            "correction_method": method,
            "timestamp": datetime.now().isoformat(),
        }

    def _setup_logging(self):
        """设置日志"""
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        log_file = (
            self.output_dir
            / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
        )

        self.logger = logging.getLogger(__name__)
        input_type = "文件" if self.input_dir.is_file() else "目录"
        self.logger.info(
            f"初始化验证器 - 输入{input_type}: {self.input_dir}, 校正方法: {self.method}"
        )

    def _validate_inputs(self):
        """验证输入参数"""
        # 检查输入路径（文件或目录）
        if not self.input_dir.exists():
            raise FileNotFoundError(f"输入路径不存在: {self.input_dir}")

        # 检查校正方法
        if self.method not in self.supported_methods:
            raise ValueError(
                f"不支持的校正方法: {self.method}. 支持的方法: {list(self.supported_methods.keys())}"
            )

        # 检查图像文件
        image_files = self._get_image_files()
        if not image_files:
            if self.input_dir.is_file():
                raise ValueError(f"文件 {self.input_dir} 不是支持的图像格式")
            else:
                raise ValueError(f"在目录 {self.input_dir} 中未找到支持的图像文件")

        self.logger.info(f"找到 {len(image_files)} 个图像文件")

    def _get_image_files(self) -> List[Path]:
        """获取图像文件列表，支持单个文件或目录"""
        # 判断是文件还是目录
        if self.input_dir.is_file():
            # 如果是单个文件，检查扩展名是否支持
            file_ext = self.input_dir.suffix.lower()
            if file_ext in self.supported_formats:
                return [self.input_dir]
            else:
                # 文件扩展名不支持，返回空列表
                return []
        elif self.input_dir.is_dir():
            # 如果是目录，按原来的逻辑处理
            image_files = []
            for ext in self.supported_formats:
                image_files.extend(self.input_dir.glob(f"*{ext}"))
                image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
            return sorted(image_files)
        else:
            # 既不是文件也不是目录
            return []

    def _load_calibration_params(self) -> Optional[Dict[str, Any]]:
        """加载校正参数"""
        if not self.calib_path:
            self.logger.warning("未指定校正参数文件，将使用默认参数")
            return None

        calib_path = Path(self.calib_path)
        if not calib_path.exists():
            raise FileNotFoundError(f"校正参数文件不存在: {calib_path}")

        try:
            if calib_path.suffix == ".npz":
                data = np.load(calib_path)
                params = {key: data[key] for key in data.files}
                self.logger.info(f"成功加载校正参数: {calib_path}")
                return params
            else:
                raise ValueError(f"不支持的校正参数文件格式: {calib_path.suffix}")
        except Exception as e:
            self.logger.error(f"加载校正参数失败: {e}")
            return None

    def _load_bad_pixel_map(self) -> Optional[np.ndarray]:
        """加载盲元信息"""
        if not self.bad_pixel_path:
            self.logger.info("未指定盲元文件，跳过盲元校正")
            return None

        bad_pixel_path = Path(self.bad_pixel_path)
        if not bad_pixel_path.exists():
            self.logger.warning(f"盲元文件不存在: {bad_pixel_path}")
            return None

        try:
            if bad_pixel_path.suffix == ".npz":
                data = np.load(bad_pixel_path)
                bad_pixel_map = data.get("blind")
                if bad_pixel_map is not None:
                    self.logger.info(f"成功加载盲元信息: {bad_pixel_path}")
                    return bad_pixel_map
                else:
                    self.logger.warning("盲元文件中未找到有效的盲元数据")
                    return None
            else:
                raise ValueError(f"不支持的盲元文件格式: {bad_pixel_path.suffix}")
        except Exception as e:
            self.logger.error(f"加载盲元信息失败: {e}")
            return None

    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        """加载图像"""
        try:
            if image_path.suffix.lower() == ".raw":
                # 假设RAW文件是16位，需要根据实际情况调整
                image = np.fromfile(image_path, dtype=np.uint16)
                # 需要知道图像尺寸，这里使用默认值
                height, width = 512, 640  # 根据实际情况调整
                image = image.reshape((height, width))
            else:
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError("无法读取图像")

            # 确保图像是16位
            if image.dtype != np.uint16:
                if image.dtype == np.uint8:
                    image = image.astype(np.uint16) * 256
                else:
                    image = image.astype(np.uint16)

            return image
        except Exception as e:
            self.logger.error(f"加载图像失败 {image_path}: {e}")
            return None

    def _save_image(self, image: np.ndarray, output_path: Path, quality: int = 95):
        """保存图像"""
        try:
            # 确保输出目录存在
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix.lower() in [".jpg", ".jpeg"]:
                # 对于JPEG，需要转换为8位
                if image.dtype == np.uint16:
                    image_8bit = (image / self.bit_max * 255).astype(np.uint8)
                else:
                    image_8bit = image.astype(np.uint8)
                cv2.imwrite(
                    str(output_path), image_8bit, [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
            else:
                cv2.imwrite(str(output_path), image)

            self.logger.debug(f"保存图像: {output_path}")
        except Exception as e:
            self.logger.error(f"保存图像失败 {output_path}: {e}")

    def _linear_correction(self, image: np.ndarray) -> np.ndarray:
        """线性校正"""
        if self.calib_params is None:
            self.logger.warning("未提供线性校正参数，返回原图")
            return image

        try:
            a_map = self.calib_params.get("a_map")
            b_map = self.calib_params.get("b_map")
            ga = self.calib_params.get("ga")
            gb = self.calib_params.get("gb")

            if a_map is None or b_map is None:
                self.logger.warning("线性校正参数不完整，返回原图")
                return image

            # 应用线性校正: corrected = gain * raw + offset
            corrected = linear_apply(a_map, b_map, ga, gb, image, self.bit_max)

            return corrected
        except Exception as e:
            self.logger.error(f"线性校正失败: {e}")
            return image

    def _quadratic_correction(self, image: np.ndarray) -> np.ndarray:
        """二次校正"""
        if self.calib_params is None:
            self.logger.warning("未提供二次校正参数，返回原图")
            return image

        try:
            a2 = self.calib_params.get("a2")
            a1 = self.calib_params.get("a1")
            a0 = self.calib_params.get("a0")

            if a2 is None or a1 is None or a0 is None:
                self.logger.warning("二次校正参数不完整，返回原图")
                return image

            # 使用优化的二次校正
            corrected = quadratic_apply(a2, a1, a0, image, self.bit_max)
            return corrected
        except Exception as e:
            self.logger.error(f"二次校正失败: {e}")
            return image

    def _bright_dark_correction(self, image: np.ndarray) -> np.ndarray:
        """明暗场校正"""
        if self.calib_params is None:
            self.logger.warning("未提供明暗场校正参数，返回原图")
            return image

        try:
            gain_arr = self.calib_params.get("gain_map")
            offset_arr = self.calib_params.get("offset_map")
            ref = self.calib_params.get("ref")

            if gain_arr is None or offset_arr is None:
                self.logger.warning("明暗场校正参数不完整，返回原图")
                return image

            corrected = bright_dark_apply(
                gain_arr, offset_arr, ref, image, self.bit_max
            )

            return corrected
        except Exception as e:
            self.logger.error(f"明暗场校正失败: {e}")
            return image

    def _apply_bad_pixel_correction(self, image: np.ndarray) -> np.ndarray:
        """应用盲元校正"""
        if self.bad_pixel_map is None:
            return image

        try:
            self.bad_pixel_map.astype(bool)
            corrected = remove_bps(image, self.bad_pixel_map)
            self.logger.debug(f"校正了 {np.sum(self.bad_pixel_map)} 个盲元")

            return corrected
        except Exception as e:
            self.logger.error(f"盲元校正失败: {e}")
            return image

    def process_image(self, image_path: Path) -> bool:
        """处理单个图像"""
        try:
            # 加载图像
            image = self._load_image(image_path)
            if image is None:
                return False

            # 应用校正
            correction_func = self.supported_methods[self.method]
            corrected_image = correction_func(image)

            # 应用盲元校正
            corrected_image = self._apply_bad_pixel_correction(corrected_image)

            # 保存结果
            output_path = self.output_dir / f"corrected_{image_path.name}"
            self._save_image(corrected_image, output_path)

            self.stats["processed_images"] += 1
            return True

        except Exception as e:
            self.logger.error(f"处理图像失败 {image_path}: {e}")
            self.stats["failed_images"] += 1
            return False

    def run_validation(self) -> Dict[str, Any]:
        """运行验证"""
        self.logger.info("开始图像校正验证...")
        start_time = time.time()

        # 获取图像文件列表
        image_files = self._get_image_files()
        self.stats["total_images"] = len(image_files)

        # 处理每个图像
        for image_path in image_files:
            self.logger.info(f"处理图像: {image_path.name}")
            self.process_image(image_path)

        # 计算处理时间
        end_time = time.time()
        self.stats["processing_time"] = end_time - start_time

        # 保存统计信息
        self._save_statistics()

        # 打印结果
        self._print_results()

        return self.stats

    def _save_statistics(self):
        """保存统计信息"""
        stats_file = self.output_dir / "validation_stats.json"
        try:
            with open(stats_file, "w", encoding="utf-8") as f:
                json.dump(self.stats, f, indent=2, ensure_ascii=False)
            self.logger.info(f"统计信息已保存: {stats_file}")
        except Exception as e:
            self.logger.error(f"保存统计信息失败: {e}")

    def _print_results(self):
        """打印结果"""
        print("\n" + "=" * 60)
        print("图像校正验证结果")
        print("=" * 60)
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"校正方法: {self.method}")
        print(f"总图像数: {self.stats['total_images']}")
        print(f"成功处理: {self.stats['processed_images']}")
        print(f"处理失败: {self.stats['failed_images']}")
        print(f"处理时间: {self.stats['processing_time']:.2f} 秒")
        print(
            f"平均速度: {self.stats['processed_images']/self.stats['processing_time']:.2f} 图像/秒"
        )
        print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="红外图像非均匀性与盲元校正验证工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python validater.py --input_dir ./data/test_images --method quadratic
  python validater.py --input_dir ./data/test_images --method linear --calib_path ./calib.npz
  python validater.py --input_dir ./data/test_images --method bright_dark --bit_max 16383
        """,
    )

    parser.add_argument("--input_dir", "-i", required=True, help="输入图像目录路径")
    parser.add_argument(
        "--method",
        "-m",
        required=True,
        choices=["linear", "quadratic", "bright_dark"],
        help="校正方法",
    )
    parser.add_argument(
        "--output_dir", "-o", help="输出目录路径 (默认: output/validation)"
    )
    parser.add_argument("--calib_path", "-c", help="校正参数文件路径 (.npz格式)")
    parser.add_argument("--bad_pixel_path", "-b", help="盲元文件路径 (.npz格式)")
    parser.add_argument(
        "--bit_max", type=int, default=4095, help="位宽上限 (默认: 4095)"
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)",
    )

    args = parser.parse_args()

    try:
        # 创建验证器
        validator = ImageValidator(
            input_dir=args.input_dir,
            method=args.method,
            output_dir=args.output_dir,
            calib_path=args.calib_path,
            bit_max=args.bit_max,
            bad_pixel_path=args.bad_pixel_path,
            log_level=args.log_level,
        )

        # 运行验证
        stats = validator.run_validation()

        # 根据处理结果设置退出码
        if stats["failed_images"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
