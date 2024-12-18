# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:54:04 2024

@author: Otiz
"""
import numpy as np

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
            return np.unpackbits(np.array([val], dtype='>u4').view(np.uint8))[-(int_bits + frac_bits):]
        else:
            # 對於負數，轉成binary最後再加上 - 號
            val = -val  # 先取絕對值
            bits = np.unpackbits(np.array([val], dtype='>u4').view(np.uint8))[-(int_bits + frac_bits):]
            return -bits
    
    # 將每個元素轉換為二進制
    binary_matrix = np.array([to_binary(val) for val in scaled_arr.flatten()])
    
    # 重塑為原始形狀並增加額外的維度
    result_shape = arr.shape + (int_bits + frac_bits,)
    binary_matrix = binary_matrix.reshape(result_shape)
    
    return binary_matrix.astype(np.int8)

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

def Sieve(A, W, frac_bits, p, r):
    A = np.array(A, dtype=np.float32)
    W = np.array(W, dtype=np.int8)
    frac_bits = np.int32(frac_bits)
    p = np.int32(p)
    r = np.int32(r)

    n_l, n_n, B = W.shape
    samples = A.shape[0]

    # Initialize X with zeros
    X = np.zeros((samples, n_n), dtype=np.float32)

    # Pre-compute bit orders and powers
    bit_orders = np.arange(B-1, -1, -1, dtype=np.float32) # shape: (B,)
    powers = bit_orders - frac_bits # shape: (B,)

    # Expand dimensions for broadcasting
    A_expanded = A[:, :, np.newaxis, np.newaxis]  # shape: (samples, n_l, 1, 1)
    W_expanded = W[np.newaxis, :, :, :]  # shape: (1, n_l, n_n, B)
    
    # Compute exponents of A
    exponents_A = exponent(A)[:, :, np.newaxis, np.newaxis]  # shape: (samples, n_l, 1, 1)
    
    # Compute sieving threshold
    sieving_threshold = r - bit_orders - p  # shape: (B,)
    sieving_threshold = sieving_threshold[np.newaxis, np.newaxis, np.newaxis, :]  # shape: (1, 1, 1, B)
    
    # Create mask where W is non-zero for each bit
    mask = W_expanded.astype(np.float32)  # shape: (1, n_l, n_n, B)
    mask = np.tile(mask, (samples, 1, 1, 1))  # shape: (samples, n_l, n_n, B)

    # Calculate contribution for each bit
    contributions = np.where(
        exponents_A >= sieving_threshold,
        A_expanded * mask,
        0.0
    )  # shape: (samples, n_l, n_n, B)
    
    # Sum contributions across input nodes and bits
    x_k = np.sum(contributions, axis=1)  # shape: (samples, n_n, B)

    # Calculate final X
    X = np.sum(x_k * (2.0 ** powers), axis=2)  # shape: (samples, n_n)
    
    return X



# Generate random A and W matrices
def generate_random_matrices(num_samples, input_dim, output_dim, int_bits=3, frac_bits=29):
    # A matrix range [0, 10]
    data = [[5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2]]

    A = np.array(data, dtype=np.float32)
    
    # W matrix range [-3, 3]
    W = np.random.uniform(-3, 3, (input_dim, output_dim)).astype(np.float32)
    
    # Convert W matrix to binary form
    W_fixed_B = float_to_fixed_point_binary(W, int_bits, frac_bits)
    
    return A, W, W_fixed_B

np.random.seed(4)  # For reproducibility
num_samples = 2
input_dim = 4
output_dim = 2
int_bits = 4
frac_bits = 28

# Generate random matrix
A, W_float, W_fixed_B = generate_random_matrices(num_samples, input_dim, output_dim, int_bits, frac_bits)

# print A and W matrix
print("Randomly generated matrix A:")
print(A)
print("Randomly generated matrix W(in Decimal):")
print(W_float)

# Results calculated using the Sieve function
X_approx = Sieve(A, W_fixed_B, frac_bits, p=126, r=150)

# Accurate result calculated using np.dot()
X_exact = np.dot(A, W_float)

print("Sieve function approximate result:")
print(X_approx)
print("The exact result of np.dot() function:")
print(X_exact)
