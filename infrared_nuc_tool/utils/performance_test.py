#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试脚本：比较不同二次拟合计算方法的耗时
测试方法包括：
1. 原始方法：a2_arr * raw_arr**2 + a1_arr * raw_arr + a0_arr
2. np.polyval方法
3. 原地操作方法（当前实现）
4. 手动优化方法
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Tuple, List
import sys
import os

# 添加当前目录到路径，以便导入nuc_tool模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from nuc_tool import quadratic_apply
except ImportError:
    print("警告：无法导入nuc_tool模块，将使用本地实现进行测试")


def generate_test_data(height: int, width: int, bit_max: int = 4095) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成测试数据
    
    :param height: 图像高度
    :param width: 图像宽度
    :param bit_max: 位宽上限
    :return: (a2_arr, a1_arr, a0_arr, raw_arr)
    """
    np.random.seed(42)  # 固定随机种子，确保结果可重复
    
    # 生成系数数组
    a2_arr = np.random.uniform(-0.001, 0.001, (height, width)).astype(np.float32)
    a1_arr = np.random.uniform(0.8, 1.2, (height, width)).astype(np.float32)
    a0_arr = np.random.uniform(-100, 100, (height, width)).astype(np.float32)
    
    # 生成原始图像数据
    raw_arr = np.random.randint(0, bit_max + 1, (height, width), dtype=np.uint16)
    
    return a2_arr, a1_arr, a0_arr, raw_arr


def method_original(a2_arr: np.ndarray, a1_arr: np.ndarray, a0_arr: np.ndarray, 
                   raw_arr: np.ndarray, bit_max: int) -> np.ndarray:
    """原始方法"""
    raw_arr_float = raw_arr.astype(np.float32)
    corrected_arr = a2_arr * raw_arr_float**2 + a1_arr * raw_arr_float + a0_arr
    corrected_arr = np.clip(corrected_arr, 0, bit_max)
    return corrected_arr.astype(np.uint16)


def method_polyval(a2_arr: np.ndarray, a1_arr: np.ndarray, a0_arr: np.ndarray, 
                  raw_arr: np.ndarray, bit_max: int) -> np.ndarray:
    """np.polyval方法"""
    raw_arr_float = raw_arr.astype(np.float32)
    corrected_arr = np.polyval([a2_arr, a1_arr, a0_arr], raw_arr_float)
    corrected_arr = np.clip(corrected_arr, 0, bit_max)
    return corrected_arr.astype(np.uint16)


def method_inplace(a2_arr: np.ndarray, a1_arr: np.ndarray, a0_arr: np.ndarray, 
                  raw_arr: np.ndarray, bit_max: int) -> np.ndarray:
    """原地操作方法"""
    raw_arr_float = raw_arr.astype(np.float32)
    
    # 预分配输出数组
    corrected_arr = np.empty_like(raw_arr_float, dtype=np.float32)
    
    # 原地计算：y = a2 * x^2 + a1 * x + a0
    np.multiply(raw_arr_float, raw_arr_float, out=corrected_arr)  # corrected_arr = raw_arr^2
    np.multiply(corrected_arr, a2_arr, out=corrected_arr)  # corrected_arr = a2 * raw_arr^2
    corrected_arr += a1_arr * raw_arr_float  # corrected_arr += a1 * raw_arr
    corrected_arr += a0_arr  # corrected_arr += a0
    
    corrected_arr = np.clip(corrected_arr, 0, bit_max)
    return corrected_arr.astype(np.uint16)


def method_manual_optimized(a2_arr: np.ndarray, a1_arr: np.ndarray, a0_arr: np.ndarray, 
                           raw_arr: np.ndarray, bit_max: int) -> np.ndarray:
    """手动优化方法"""
    raw_arr_float = raw_arr.astype(np.float32)
    raw_squared = raw_arr_float * raw_arr_float  # 避免使用**2操作符
    corrected_arr = a2_arr * raw_squared + a1_arr * raw_arr_float + a0_arr
    corrected_arr = np.clip(corrected_arr, 0, bit_max)
    return corrected_arr.astype(np.uint16)


def benchmark_method(method_func, a2_arr: np.ndarray, a1_arr: np.ndarray, a0_arr: np.ndarray, 
                    raw_arr: np.ndarray, bit_max: int, num_runs: int = 10) -> Tuple[float, float]:
    """
    基准测试单个方法
    
    :param method_func: 测试方法函数
    :param a2_arr: 二次项系数
    :param a1_arr: 一次项系数
    :param a0_arr: 常数项
    :param raw_arr: 原始数据
    :param bit_max: 位宽上限
    :param num_runs: 运行次数
    :return: (平均耗时, 标准差)
    """
    times = []
    
    # 预热
    for _ in range(3):
        _ = method_func(a2_arr, a1_arr, a0_arr, raw_arr, bit_max)
    
    # 正式测试
    for _ in range(num_runs):
        start_time = time.perf_counter()
        result = method_func(a2_arr, a1_arr, a0_arr, raw_arr, bit_max)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)


def run_performance_test():
    """运行性能测试"""
    print("=" * 60)
    print("二次拟合计算方法性能测试")
    print("=" * 60)
    
    # 测试不同图像尺寸
    test_sizes = [
        (512, 512),    # 小图像
        (1024, 1024),  # 中等图像
        (2048, 2048),  # 大图像
        (4096, 4096),  # 超大图像
    ]
    
    methods = [
        ("原始方法", method_original),
        ("np.polyval方法", method_polyval),
        ("原地操作方法", method_inplace),
        ("手动优化方法", method_manual_optimized),
    ]
    
    results = {}
    
    for height, width in test_sizes:
        print(f"\n测试图像尺寸: {height} x {width}")
        print("-" * 40)
        
        # 生成测试数据
        a2_arr, a1_arr, a0_arr, raw_arr = generate_test_data(height, width)
        
        results[f"{height}x{width}"] = {}
        
        for method_name, method_func in methods:
            try:
                avg_time, std_time = benchmark_method(method_func, a2_arr, a1_arr, a0_arr, raw_arr, 4095)
                results[f"{height}x{width}"][method_name] = (avg_time, std_time)
                
                print(f"{method_name:15s}: {avg_time*1000:8.2f}ms ± {std_time*1000:5.2f}ms")
                
            except Exception as e:
                print(f"{method_name:15s}: 错误 - {str(e)}")
                results[f"{height}x{width}"][method_name] = (float('inf'), 0)
    
    return results


def plot_results(results: dict):
    """绘制性能测试结果"""
    try:
        import matplotlib.pyplot as plt
        
        # 准备数据
        sizes = list(results.keys())
        method_names = list(results[list(results.keys())[0]].keys())
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 子图1：绝对耗时
        for method_name in method_names:
            times = []
            for size in sizes:
                avg_time, _ = results[size][method_name]
                if avg_time != float('inf'):
                    times.append(avg_time * 1000)  # 转换为毫秒
                else:
                    times.append(0)
            
            ax1.plot(sizes, times, marker='o', label=method_name, linewidth=2)
        
        ax1.set_xlabel('图像尺寸')
        ax1.set_ylabel('耗时 (ms)')
        ax1.set_title('不同方法的绝对耗时对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 子图2：相对性能（以原始方法为基准）
        baseline_method = "原始方法"
        for method_name in method_names:
            if method_name == baseline_method:
                continue
                
            speedups = []
            for size in sizes:
                baseline_time, _ = results[size][baseline_method]
                method_time, _ = results[size][method_name]
                
                if baseline_time != float('inf') and method_time != float('inf') and method_time > 0:
                    speedup = baseline_time / method_time
                    speedups.append(speedup)
                else:
                    speedups.append(1)
            
            ax2.plot(sizes, speedups, marker='s', label=method_name, linewidth=2)
        
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='基准线')
        ax2.set_xlabel('图像尺寸')
        ax2.set_ylabel('加速比')
        ax2.set_title('相对性能对比（以原始方法为基准）')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('performance_test_results.png', dpi=300, bbox_inches='tight')
        print(f"\n性能测试结果图表已保存为: performance_test_results.png")
        
    except ImportError:
        print("\n警告：matplotlib未安装，跳过图表生成")


def print_summary(results: dict):
    """打印测试总结"""
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    # 找出最佳方法
    best_methods = {}
    for size, methods in results.items():
        best_time = float('inf')
        best_method = None
        
        for method_name, (avg_time, _) in methods.items():
            if avg_time < best_time:
                best_time = avg_time
                best_method = method_name
        
        best_methods[size] = (best_method, best_time)
    
    print("\n各尺寸下的最佳方法：")
    for size, (method, time) in best_methods.items():
        print(f"{size:12s}: {method:15s} ({time*1000:6.2f}ms)")
    
    # 计算平均加速比
    print("\n平均加速比（相对于原始方法）：")
    baseline_method = "原始方法"
    for method_name in ["np.polyval方法", "原地操作方法", "手动优化方法"]:
        speedups = []
        for size, methods in results.items():
            baseline_time, _ = methods[baseline_method]
            method_time, _ = methods[method_name]
            
            if baseline_time != float('inf') and method_time != float('inf') and method_time > 0:
                speedup = baseline_time / method_time
                speedups.append(speedup)
        
        if speedups:
            avg_speedup = np.mean(speedups)
            print(f"{method_name:15s}: {avg_speedup:.2f}x")


if __name__ == "__main__":
    print("开始性能测试...")
    
    # 运行测试
    results = run_performance_test()
    
    # 打印总结
    print_summary(results)
    
    # 绘制结果
    plot_results(results)
    
    print("\n性能测试完成！")
