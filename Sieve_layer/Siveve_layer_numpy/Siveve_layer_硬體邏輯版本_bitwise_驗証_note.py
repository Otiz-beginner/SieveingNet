# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:25:58 2024

@author: Otiz
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

def Sieve(A, W, frac_bits, p, im, r):
    A = np.array(A, dtype=np.float64)
    W = np.array(W, dtype=np.int8)
    frac_bits = np.int32(frac_bits)
    p = np.int32(p)
    im = np.float32(im)
    r = np.int32(r)

    n_l, n_n, B = W.shape
    samples = A.shape[0]

    # Initialize X with zeros
    X = np.zeros((samples, n_n), dtype=np.float64)

    # Pre-compute bit orders and powers
    bit_orders = np.arange(B-1, -1, -1, dtype=np.float32) # shape: (B,)
    powers = bit_orders - frac_bits # shape: (B,)

    # Expand dimensions for broadcasting
    A_expanded = A[:, :, np.newaxis]  # shape: (samples, n_l, 1)
    W_expanded = W[np.newaxis, :, :, :]  # shape: (1, n_l, n_n, B)
    
    # Compute exponents of A
    exponents_A = exponent(A)[:, :, np.newaxis]  # shape: (samples, n_l, 1)
    mask_non_zero_count = 0
    contribution_non_zero_count = 0
    # Iterate over bits
    for k in range(B):
        # Compute sieving threshold
        exponents_X = exponent(X)[:, np.newaxis, :]  # shape: (samples, 1, n_n)
        sieving_threshold = r + im * exponents_X - bit_orders[k] - p # shape: (samples, 1, n_n)

        # Create mask where W is non-zero for the current bit
        mask = (W_expanded[:, :, :, k]).astype(np.float32)  # shape: (1, n_l, n_n)
        mask = np.tile(mask, (samples, 1, 1))
        # print(f"mask: \n{mask}")
        mask_non_zero_count += np.count_nonzero(mask)
        # Calculate contribution for the current bit
        contribution = np.where(
            exponents_A >= sieving_threshold,
            A_expanded * mask,
            0.0
        )  # shape: (samples, n_l, n_n)
        # print(f"contribution: \n{contribution}")
        contribution_non_zero_count += np.count_nonzero(contribution)
        # Sum contributions across input nodes
        x_k = np.sum(contribution, axis=1)  # shape: (samples, n_n)

        # Update X
        X += x_k * (2.0 ** powers[k])
    print(f"mask_non_zero_count: \n{mask_non_zero_count}")
    print(f"contribution_non_zero_count: \n{contribution_non_zero_count}")
    return X



# Generate random A and W matrices
def generate_random_matrices(num_samples, input_dim, output_dim, int_bits=4, frac_bits=28):
    # A matrix range [0, 10]  ReLU 活化後的節點值可能在 [0, 10] 範圍內，具體取決於網路深度和輸入資料分佈。
    # A = np.random.uniform(0, 10, (num_samples, input_dim)).astype(np.float64)
    # 定義陣列數據
    data = [[5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2]]

    # 創建 NumPy 陣列，並指定資料型態為 float64
    A = np.array(data, dtype=np.float64)
    
    # W matrix range [-3, 3] 在神經網路的訓練過程中，權重值通常集中在較小的範圍內（例如 [-3, 3] 或 [-1, 1]）
    W = np.random.uniform(-3, 3, (input_dim, output_dim)).astype(np.float64)
    
    # Convert W matrix to binary form
    W_fixed_B = float_to_fixed_point_binary(W, int_bits, frac_bits)
    
    return A, W, W_fixed_B

np.random.seed(4)  # For reproducibility
num_samples = 2
input_dim = 4
output_dim = 2
# num_samples = 128
# input_dim = 714
# output_dim = 512
int_bits = 4
frac_bits = 28

# Generate random matrix
A, W_float, W_fixed_B = generate_random_matrices(num_samples, input_dim, output_dim, int_bits, frac_bits)

# print A and W matrix
print("Randomly generated matrix A:")
print(A)
print("Randomly generated matrix W(in Decimal):")
print(W_float)
# print("matrix W(in Binary):")
# print(W_fixed_B)

# Results calculated using the Sieve function
X_approx = Sieve(A, W_fixed_B, frac_bits, p=10, im=15, r=0)

# Accurate result calculated using np.dot()
X_exact = np.dot(A, W_float)

print("Sieve function approximate result:")
print(X_approx)
print("The exact result of np.dot() function:")
print(X_exact)
