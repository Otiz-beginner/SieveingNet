# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 16:22:02 2024

@author: user
"""

def float_to_fixed_point_binary(arr, int_bits=2, frac_bits=29):
    # 計算轉換係數
    scale = 2 ** frac_bits
    
    # 將浮點數乘以係數
    scaled_arr = np.round(arr * scale).astype(np.int64)
    
    # 計算整數的最大值和最小值
    max_val = (2 ** (int_bits + frac_bits)) - 1
    min_val = -(2 ** (int_bits + frac_bits))
    
    # 確保數值在允許的範圍內
    scaled_arr = np.clip(scaled_arr, min_val, max_val)
    
    def to_binary(val):
        if val >= 0:
            # 對於正數，直接轉成binary
            return np.unpackbits(np.array([val], dtype='>u8').view(np.uint8))[-(int_bits + frac_bits):]
            # >：Big-endian，u：Unsigned，2：two words（16bits），4: four words（32bits）
        else:
            # 對於正數，轉成binary最後再加上 - 號
            val = -val  # 先取絕對值
            bits = np.unpackbits(np.array([val], dtype='>u8').view(np.uint8))[-(int_bits + frac_bits):]
            return -bits
    
    # 將每個元素轉換為二進制
    binary_matrix = np.array([to_binary(val) for val in scaled_arr.flatten()])
    
    # 重塑為原始形狀並增加額外的維度
    result_shape = arr.shape + (int_bits + frac_bits,)
    binary_matrix = binary_matrix.reshape(result_shape)
    
    return binary_matrix.astype(np.int8)


import numpy as np

def exponent(x):
    if x.dtype == np.float32:
        packed_data = x.view(np.uint32) # 先當作無號數看待，右移才會是unsigned extend數值才不會錯
        exponent_mask = 0x7F800000
        exponent_part_bias = (packed_data & exponent_mask) >> 23
        exponent_part = exponent_part_bias.view(np.int32) - 127 # 最後再看待乘有號數，才會有負數
    elif x.dtype == np.float64:
        packed_data = x.view(np.uint64)
        exponent_mask = 0x7FF0000000000000
        exponent_part_bias = (packed_data & exponent_mask) >> 52
        exponent_part = exponent_part_bias.view(np.int64) - 1023
    else:
        raise ValueError("Only float32 and float64 are supported.")
    return exponent_part

from numba import jit

def sieve_conventional(A, W, frac_bits, im):
    A = np.asarray(A, dtype=np.float32)
    W = np.asarray(W, dtype=np.int8)
    frac_bits = np.int32(frac_bits)
    im = np.float32(im)
    n_l, n_n, B = W.shape
    samples = A.shape[0]

    # Initialize X with zeros
    X = np.zeros((samples, n_n), dtype=np.float32)

    # Pre-compute bit orders and powers
    bit_orders = np.arange(B-1, -1, -1, dtype=np.float32)
    powers = 2.0 ** (bit_orders - frac_bits)

    # Expand dimensions for broadcasting
    A_expanded = A[:, :, np.newaxis]  # shape: (samples, n_l, 1)
    W_expanded = W[np.newaxis, :, :, :]  # shape: (1, n_l, n_n, B)

    # Compute exponents of A
    exponents_A = np.frexp(A)[1].astype(np.float32)[:, :, np.newaxis]  # shape: (samples, n_l, 1)

    # Pre-allocate memory for sieving threshold
    sieving_threshold = np.zeros((samples, 1, n_n), dtype=np.float32)

    for k in range(B):
        # Compute sieving threshold
        exponents_X = np.frexp(X)[1].astype(np.float32)[:, np.newaxis, :]
        np.subtract(exponents_X, im * (exponents_X + 127) + bit_orders[k] - frac_bits, out=sieving_threshold)

        # Calculate contribution for the current bit
        mask = W_expanded[:, :, :, k].astype(np.float32)
        contribution = np.where(exponents_A >= sieving_threshold, A_expanded * mask, 0.0)

        # Sum contributions across input nodes and update X
        x_k = np.sum(contribution, axis=1)
        X += x_k * powers[k]

    return X



class SieveingNet:
    def __init__(self, samples, n_n):
        self.S_previous = np.zeros((samples, n_n), dtype=np.float32)
        
    def sieve(self, A, W, frac_bits, im):
        A = np.asarray(A, dtype=np.float32)
        W = np.asarray(W, dtype=np.int8)
        frac_bits = np.int32(frac_bits)
        im = np.float32(im)
        n_l, n_n, B = W.shape

        # Pre-compute bit orders
        bit_orders = np.arange(B-1, -1, -1, dtype=np.float32)

        # Compute exponents of A and S_previous
        exponents_A = np.frexp(A)[1].astype(np.float32)[:, :, np.newaxis, np.newaxis]
        exponents_S_p = np.frexp(self.S_previous)[1].astype(np.float32)[:, np.newaxis, :, np.newaxis]

        # Compute sieving thresholds for all bits with the new formula
        sieving_thresholds = exponents_S_p - im * (exponents_S_p - (-127)) - (bit_orders - frac_bits)

        # Calculate contributions for each bit
        contributions = np.where(
            exponents_A >= sieving_thresholds,
            A[:, :, np.newaxis, np.newaxis] * W,
            0.0
        )

        # Sum contributions and calculate final X
        X = np.einsum('ijkl,l->ik', contributions, 2.0 ** (bit_orders - frac_bits))

        self.S_previous = X
        return X


# Generate random A and W matrices
def generate_random_matrices(num_samples, input_dim, output_dim, int_bits=3, frac_bits=29):
    # A matrix range [0, 10]  ReLU 活化後的節點值可能在 [0, 10] 範圍內，具體取決於網路深度和輸入資料分佈。
    A = np.random.uniform(-1000, 1, (num_samples, input_dim)).astype(np.float32)
    
    # W matrix range [-3, 3] 在神經網路的訓練過程中，權重值通常集中在較小的範圍內（例如 [-3, 3] 或 [-1, 1]）
    W = np.random.uniform(-3, 3, (input_dim, output_dim)).astype(np.float32)
    
    # Convert W matrix to binary form
    W_fixed_B = float_to_fixed_point_binary(W, int_bits, frac_bits)
    
    return A, W, W_fixed_B

np.random.seed(4)  # For reproducibility
num_samples = 128
input_dim = 4 # input dimensions
output_dim = 4 # output dimensions
int_bits = 4
frac_bits = 28

im = 1

# Generate random matrix
A, W_float, W_fixed_B = generate_random_matrices(num_samples, input_dim, output_dim, int_bits, frac_bits)

X_exact = np.dot(A, W_float)

SieveingNet_new = SieveingNet(num_samples, output_dim)

X_approx_c = sieve_conventional(A, W_fixed_B, frac_bits, im)
X_approx_new = SieveingNet_new.sieve(A, W_fixed_B, frac_bits, im)


print(f"X_exact: \n{X_exact}")
print(f"X_approx_c: \n{X_approx_c}")
print(f"X_approx_new: \n{X_approx_new}")


import time
import pandas as pd
from typing import Callable

# 假設 SieveingNet, sieve_conventional, float_to_fixed_point_binary 和其他必要的函數已經定義

def measure_execution_time(func: Callable, *args) -> float:
    start_time = time.time()
    func(*args)
    end_time = time.time()
    return end_time - start_time

def run_performance_test(dimensions: list, num_samples: int, num_runs: int = 10):
    results = []
    
    for input_dim, output_dim in dimensions:
        print(f"Testing dimensions: input={input_dim}, output={output_dim}")
        
        # 生成隨機矩陣
        A, W_float, W_fixed_B = generate_random_matrices(num_samples, input_dim, output_dim, int_bits, frac_bits)
        
        # 初始化 SieveingNet
        sieveingnet = SieveingNet(num_samples, output_dim)
        
        # 運行多次測試並計算平均時間
        conventional_times = []
        sieveingnet_times = []
        
        for _ in range(num_runs):
            conventional_times.append(measure_execution_time(sieve_conventional, A, W_fixed_B, frac_bits, im))
            sieveingnet_times.append(measure_execution_time(sieveingnet.sieve, A, W_fixed_B, frac_bits, im))
        
        avg_conventional_time = np.mean(conventional_times)
        avg_sieveingnet_time = np.mean(sieveingnet_times)
        
        results.append({
            'Input Dim': input_dim,
            'Output Dim': output_dim,
            'Conventional Avg Time (s)': avg_conventional_time,
            'SieveingNet Avg Time (s)': avg_sieveingnet_time
        })
    
    return pd.DataFrame(results)

# 設置參數
np.random.seed(4)  # 為了可重現性
num_samples = 1
int_bits = 4
frac_bits = 28
im = 1

# 定義要測試的維度
dimensions = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512)
]

# 運行性能測試
results_df = run_performance_test(dimensions, num_samples)

# 將結果保存為 Excel 文件
excel_filename = 'sieving_methods_performance_comparison.xlsx'
results_df.to_excel(excel_filename, index=False)
print(f"Results have been saved to {excel_filename}")

# 顯示結果
print(results_df)

