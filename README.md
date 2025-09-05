# 红外图像非均匀性校正(NUC)工具集

红外图像非均匀性校正工具集，包含校正算法、可视化和验证。

## 项目概述

- **校正算法**：实现线性、二次、明暗场等多种校正方法
- **可视化工具**：提供PyQt5图形界面进行数据分析和结果展示
- **验证工具**：支持校正效果的验证和性能评估

## 项目结构

```
infrared_nuc/
├── infrared_nu_app/          # 上位机GUI应用
│   ├── app_it_temp_curve.py  # 积分时间-温度像素曲线分析器
│   ├── app_npz_curve.py      # 校正系数可视化器
│   └── README.md             # 应用说明文档
├── infrared_nuc_tool/        # 核心工具库
│   ├── corrected.py          # 校正主程序
│   ├── validater.py          # 验证工具
│   └── utils/                # 工具函数库
│       ├── nuc_tool.py       # 非均匀性校正算法
│       ├── bp_tool.py        # 盲元检测与校正
│       ├── pre_tool.py       # 数据预处理工具
│       └── performance_test.py # 性能测试工具
├── .gitignore               # Git忽略文件配置
└── README.md               # 项目说明文档
```

## 功能特性

### 1. 校正算法

- **线性校正**：基于多点校准的线性拟合
- **二次校正**：二次多项式拟合校正
- **明暗场校正**：基于明暗场的两点校正
- **盲元校正**：死像元、过热像元检测与校正

### 2. 可视化工具

- **积分时间-温度像素曲线分析器**：分析像素随温度变化的响应曲线
- **校正系数可视化器**：可视化校正参数的空间分布
- **多点对比分析**：支持多个像素点的对比分析
- **数据导出**：支持CSV格式数据导出

## 使用方法

### 1. 数据校正

```bash
# 基本用法
python infrared_nuc_tool/corrected.py --data_path ./data --output_path ./output --wave lwir

# 完整参数示例
python infrared_nuc_tool/corrected.py \
    --data_path D:/Projects/2025/145corr/nuc/0902_corr \
    --exposure_minvalue 1500 \
    --exposure_maxvalue 3500 \
    --exposure_step 100 \
    --temperatures_minvalue 4 \
    --temperatures_maxvalue 80 \
    --temperatures_step 2 \
    --wave lwir \
    --bit_max 16383
```

### 2. 校正验证

```bash
# 线性校正验证
python infrared_nuc_tool/validater.py \
    --input_dir ./data/test_images \
    --method linear \
    --calib_path ./output/nuc/calib.npz

# 二次校正验证
python infrared_nuc_tool/validater.py \
    --input_dir ./data/test_images \
    --method quadratic \
    --calib_path ./output/nuc/calib.npz
```

### 3. 可视化分析

```bash
# 积分时间-温度像素曲线分析
python infrared_nu_app/app_it_temp_curve.py

# 校正系数可视化
python infrared_nu_app/app_npz_curve.py
```

## 数据格式要求

### 输入数据目录结构

```
data/
├── 4du/                    # 温度目录（4°C）
│   ├── 1500/              # 积分时间目录（1500μs）
│   │   ├── 000.png
│   │   ├── 001.png
│   │   └── ...
│   ├── 1600/
│   └── ...
├── 6du/                   # 温度目录（6°C）
│   └── ...
└── ...
```

### 输出数据目录结构

```
output/
├── nuc/                   # 校正参数
│   ├── lwir/             # 长波红外
│   │   ├── linear/       # 线性校正参数
│   │   └── quadrast/     # 二次校正参数
│   └── mwir/             # 中波红外
├── validation/            # 验证结果
└── log/                  # 日志文件
```

## 参数说明

### 校正参数

- `--data_path`: 输入数据目录路径
- `--output_path`: 输出结果目录路径
- `--wave`: 波段类型（lwir/mwir）
- `--bit_max`: 位宽上限（默认4095）
- `--exposure_minvalue/maxvalue/step`: 积分时间范围
- `--temperatures_minvalue/maxvalue/step`: 温度范围
- `--keep_ratio`: 分位点选择比例（默认0.6）

### 验证参数

- `--input_dir`: 输入图像目录
- `--method`: 校正方法（linear/quadratic/bright_dark）
- `--calib_path`: 校正参数文件路径
- `--bad_pixel_path`: 盲元文件路径
- `--bit_max`: 位宽上限

## 算法原理

### 线性校正

基于多点校准数据，通过最小二乘法拟合线性关系：

```
corrected = gain * raw + offset
```

### 二次校正

使用二次多项式拟合像素响应：

```
corrected = a2 * raw² + a1 * raw + a0
```

### 明暗场校正

基于明暗两个参考点的两点校正：

```
corrected = gain * raw + offset
```
