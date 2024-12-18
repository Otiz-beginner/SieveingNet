# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:25:58 2024

@author: Otiz
"""

import numpy as np
import time

def BCB2TCB(BCB):
    state=0
    L = BCB.shape[0]
    # print(f"BCB: {BCB}")
    BCB_new = np.zeros((L+1, ))
    BCB_new[0] = 8
    BCB_new[1:] = BCB[:]
    # print(f"BCB_new: {BCB_new}")
    TCB = np.zeros((L, ))
    # print(f"TCB: {TCB}")
    for i in range(L, -1, -1): # i in [8, 7, 6, 5, 4, 3, 2, 1, 0]
        j = i-1 # j in [7, 6, 5, 4, 3, 2, 1, 0, -1]
        X=BCB_new[i]
        # print(f"處理輸入第{i}個")
        # print("-------------------")
        # print(f"state: {state}")
        # print(f"輸入為{X}")
        # print(f"j為{j}")
        if state==0:
            if X==0:
                TCB[j] = 0
            elif X==1:
                state = 1
        elif state==1:
            if X==0:
                TCB[j:j+2]= [0, 1]
                state = 0
            elif X==1:
                state = 2
        elif state==2:
            if X==0:
                state = 5
            elif X==1:
                state = 3
                TCB[j:j+3] = [0, 0, -1]
        elif state==3:
            if X==0:
                state = 4
            elif X==1:
                TCB[j] = 0
                state = 3
        elif state==4:
            if X==0:
                TCB[j:j+2] = [0, 1]
                state = 0
            elif X==1:
                state = 2
            else:
                TCB[(j+1)]= 1
        elif state==5:
            if X==0:
                TCB[j:j+4] = [0, 1, 0, -1] 
                state = 0
            elif X==1:
                TCB[j:j+2] = [0, -1]
                state = 2
            else:
                TCB[(j+1):(j+1)+3] = [1, 0, -1]
        else:
            print("Error")
        # print(f"TCB: {TCB}")
        # print()
    return TCB

def float_to_fixed_point_TCB(arr, int_bits=4, frac_bits=12):
    # 計算轉換係數
    scale = 2 ** frac_bits
    
    # 將浮點數乘以係數
    scaled_arr = np.round(arr * scale).astype(np.int32)
    
    # 計算整數的最大值和最小值
    max_val = (2 ** (int_bits + frac_bits)) - 1
    min_val = -(2 ** (int_bits + frac_bits))
    
    # 確保數值在允許的範圍內
    scaled_arr = np.clip(scaled_arr, min_val, max_val)
    
    def to_binary(val):
        if val >= 0:
            # 對於正數，直接轉成BCB
            BCB = np.unpackbits(np.array([val], dtype='>u4').view(np.uint8))[-(int_bits + frac_bits):]
            TCB = BCB2TCB(BCB)
            return TCB
            # >：Big-endian，u：Unsigned，2：two words（16bits），4: four words（32bits）
        else:
            # 對於正數，轉成binary最後再加上 - 號
            val = -val  # 先取絕對值
            BCB = np.unpackbits(np.array([val], dtype='>u4').view(np.uint8))[-(int_bits + frac_bits):]
            TCB = BCB2TCB(BCB)
            return -TCB
    
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

def Sieve(A, W, im, r, frac_bits=28, p=10):
    A = np.array(A, dtype=np.float32)
    W = np.array(W, dtype=np.int8)
    frac_bits = np.int32(frac_bits)
    p = np.int32(p)
    im = np.float32(im)
    r = np.int32(r)

    n_l, n_n, B = W.shape
    samples = A.shape[0]

    # Initialize X with zeros
    X = np.zeros((samples, n_n), dtype=np.float32)

    # Pre-compute bit orders and powers
    bit_orders = np.arange(B-1, -1, -1, dtype=np.float32) # shape: (B,)
    powers = bit_orders - frac_bits # shape: (B,)

    # Expand dimensions for broadcasting
    A_expanded = A[:, :, np.newaxis]  # shape: (samples, n_l, 1)
    W_expanded = W[np.newaxis, :, :, :]  # shape: (1, n_l, n_n, B)
    
    # Compute exponents of A
    exponents_A = exponent(A)[:, :, np.newaxis]  # shape: (samples, n_l, 1)

    # Iterate over bits
    for k in range(B):
        # Compute sieving threshold
        exponents_X = exponent(X)[:, np.newaxis, :]  # shape: (samples, 1, n_n)
        sieving_threshold = r + im * exponents_X - bit_orders[k] - p # shape: (samples, 1, n_n)

        # Create mask where W is non-zero for the current bit
        mask = (W_expanded[:, :, :, k]).astype(np.float32)  # shape: (1, n_l, n_n)

        # Calculate contribution for the current bit
        contribution = np.where(
            exponents_A >= sieving_threshold,
            A_expanded * mask,
            0.0
        )  # shape: (samples, n_l, n_n)

        # Sum contributions across input nodes
        x_k = np.sum(contribution, axis=1)  # shape: (samples, n_n)

        # Update X
        X += x_k * (2.0 ** powers[k])

    return X


# Generate random A and W matrices
def generate_random_matrices(num_samples, input_dim, output_dim, int_bits=3, frac_bits=29):
    # A matrix range [0, 10]  ReLU 活化後的節點值可能在 [0, 10] 範圍內，具體取決於網路深度和輸入資料分佈。
    A = np.random.uniform(0, 10, (num_samples, input_dim)).astype(np.float32)
    
    # W matrix range [-3, 3] 在神經網路的訓練過程中，權重值通常集中在較小的範圍內（例如 [-3, 3] 或 [-1, 1]）
    W = np.random.uniform(-3, 3, (input_dim, output_dim)).astype(np.float32)
    
    # Convert W matrix to binary form
    W_fixed_B = float_to_fixed_point_TCB(W, int_bits, frac_bits)
    
    return A, W, W_fixed_B

np.random.seed(49)  # For reproducibility
num_samples = 2
input_dim = 2
output_dim = 2
# num_samples = 128
# input_dim = 714
# output_dim = 512
int_bits = 4
frac_bits = 28

# Generate random matrix
A, W_float, W_fixed_B = generate_random_matrices(num_samples, input_dim, output_dim, int_bits, frac_bits)

# print A and W matrix
# print("Randomly generated matrix A:")
# print(A)
# print("Randomly generated matrix W(in Decimal):")
# print(W_float)
# print("matrix W(in Binary):")
# print(W_fixed_B)

# Results calculated using the Sieve function
# X_approx = Sieve(A, W_fixed_B, im=0, r=0)

# Accurate result calculated using np.dot()
X_exact = np.dot(A, W_float)

# Observe exponent of input
exponent_of_input = exponent(A)

# print("Sieve function approximate result:")
# print(X_approx)
# print(f"type(X_approx): {type(X_approx)}")
# print(f"type(X_approx[0, 0]): {type(X_approx[0, 0])}")
# print("The exact result of np.dot() function:")
print(X_exact)
print(f"type(X_exact): {type(X_exact)}")
# print(f"type(X_exact[0, 0]): {type(X_exact[0, 0])}")

for i in range(60):
    X_approx = Sieve(A, W_fixed_B, im=1, r=i)
    print(f"X_approx(im={1}, r={i}): \n{X_approx}\n")
    print("-----------------------------------------")
