# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:25:58 2024

@author: Otiz
"""
import numpy as np
import matplotlib.pyplot as plt

def count_elements(arr):
    unique_elements, counts = np.unique(arr, return_counts=True)
    result_dict = dict(zip(unique_elements, counts))
    return result_dict

def exponent_normalization(input_dict):
    max_key = max(input_dict.keys())  # 確保我們用的是正確的最大值
    normalized_dict = {}
    for element, counts in input_dict.items():
        normalized_dict[element] = counts / (2 ** (max_key - element))
    return normalized_dict

def plot_exponent_counts(count_dict):
    keys = list(count_dict.keys())
    values = list(count_dict.values())
    
    plt.bar(keys, values)
    plt.xlabel('Exponent value')
    plt.ylabel('Impact')
    plt.title('Impact of Each Exponent Number')
    plt.show()


def float_to_fixed_point_binary(arr, int_bits=4, frac_bits=28):
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
            # >：Big-endian，u：Unsigned，2：two words（16bits），4: four words（32bits）
        else:
            # 對於正數，轉成binary最後再加上 - 號
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

def Sieve(A, W, frac_bits, p, im, r):
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
    # print(f"mask_non_zero_count: \n{mask_non_zero_count}")
    # print(f"contribution_non_zero_count: \n{contribution_non_zero_count}")
    return X, mask_non_zero_count, contribution_non_zero_count



# Generate random A and W matrices
def generate_random_matrices(num_samples, input_dim, output_dim, int_bits=4, frac_bits=28):
    # A matrix range [0, 10]  ReLU 活化後的節點值可能在 [0, 10] 範圍內，具體取決於網路深度和輸入資料分佈。
    # A = np.random.uniform(0, 10, (num_samples, input_dim)).astype(np.float64)
    # 定義陣列數據
    data_A = [
    [6.1, 2.8, 4.7, 1.2],
    [5.7, 3.8, 1.7, 0.3],
    [7.7, 2.6, 6.9, 2.3],
    [6.0, 2.9, 4.5, 1.5],
    [6.8, 2.8, 4.8, 1.4],
    [5.4, 3.4, 1.5, 0.4],
    [5.6, 2.9, 3.6, 1.3],
    [6.9, 3.1, 5.1, 2.3],
    [6.2, 2.2, 4.5, 1.5],
    [5.8, 2.7, 3.9, 1.2],
    [6.5, 3.2, 5.1, 2.0],
    [4.8, 3.0, 1.4, 0.1],
    [5.5, 3.5, 1.3, 0.2],
    [4.9, 3.1, 1.5, 0.1],
    [5.1, 3.8, 1.5, 0.3],
    [6.3, 3.3, 4.7, 1.6],
    [6.5, 3.0, 5.8, 2.2],
    [5.6, 2.5, 3.9, 1.1],
    [5.7, 2.8, 4.5, 1.3],
    [6.4, 2.8, 5.6, 2.2],
    [4.7, 3.2, 1.6, 0.2],
    [6.1, 3.0, 4.9, 1.8],
    [5.0, 3.4, 1.6, 0.4],
    [6.4, 2.8, 5.6, 2.1],
    [7.9, 3.8, 6.4, 2.0],
    [6.7, 3.0, 5.2, 2.3],
    [6.7, 2.5, 5.8, 1.8],
    [6.8, 3.2, 5.9, 2.3],
    [4.8, 3.0, 1.4, 0.3],
    [4.8, 3.1, 1.6, 0.2]]

    # 創建 NumPy 陣列，並指定資料型態為 float32
    A = np.array(data_A, dtype=np.float32)
    
    # W matrix range [-3, 3] 在神經網路的訓練過程中，權重值通常集中在較小的範圍內（例如 [-3, 3] 或 [-1, 1]）
    # W = np.random.uniform(-3, 3, (input_dim, output_dim)).astype(np.float64)
    data_W1 = [
        [-7.64388488e-01,  2.48292439e-03, -2.61120316e+00, -1.61131923e+00,
         -1.46625659e+00, -1.78225946e+00,  1.85798356e+00,  9.63184393e+00,
          5.77611828e-01, -4.90204166e-01],
        [-9.67210507e-01, -1.98305913e+00, -1.38386415e+01,  6.01271036e-01,
         -9.50765936e-01,  3.19653291e-01,  1.06035464e+00, -4.10230470e-01,
          3.09208677e+00, -1.81060235e+00],
        [-1.20435325e+00,  2.05580246e+00,  1.00530702e+01, -1.41387434e+00,
         -4.81800376e-01,  1.00437007e+00, -6.97882743e-01, -1.06094424e+01,
         -4.02525109e+00, -3.65544951e-01],
        [ 3.44859836e-01,  1.34723440e+00,  8.57739932e+00,  8.24711903e-01,
          4.26505497e-01,  3.07986044e-02, -5.07618092e-01, -9.61976948e+00,
         -1.98278185e+00, -9.59742528e-01]]
    
    # 創建 NumPy 陣列，並指定資料型態為 float32
    W = np.array(data_W1, dtype=np.float32)
    
    # Convert W matrix to binary form
    W_fixed_B = float_to_fixed_point_binary(W, int_bits, frac_bits)
    
    return A, W, W_fixed_B

np.random.seed(4)  # For reproducibility
num_samples = 30
input_dim = 4
output_dim = 10
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

data_b1 = [
    [-2.27227640e-03, -3.86011263e-01, -5.19294523e+00, -1.30102034e-03,
     3.55851516e-04,  4.57156362e-03, -7.27293871e-04,  7.53348143e+00,
     4.01042719e-01, -1.88253262e-01]]

# 創建 NumPy 陣列，並指定資料型態為 float32
b1 = np.array(data_b1, dtype=np.float32)

# print(f"exponent(A): \n{exponent(A)}")
# 繪製輸入的exponent大小分佈
plot_exponent_counts(exponent_normalization(count_elements(exponent(A))))

im = 1
r = 0
p = 30

befor_count = 0
after_count = 0
# Results calculated using the Sieve function
Z1_approx, before_temp, after_temp = Sieve(A, W_fixed_B, frac_bits, p=p, im=im, r=r)
Z1_approx = Z1_approx + b1
befor_count += before_temp
after_count += after_temp
print(f"befor_count: \n{befor_count}")
print(f"after_count: \n{after_count}")

# Accurate result calculated using np.dot()
Z1_exact = np.dot(A, W_float) + b1

# print("Sieve function approximate result:")
# print(Z1_approx[0])
# print("The exact result of np.dot() function:")
# print(Z1_exact[0])

# print(f"type(Z1_approx[0, 0]): \n{type(Z1_approx[0, 0])}")
# print(f"type(Z1_exact[0, 0]): \n{type(Z1_exact[0, 0])}")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

A1_approx = sigmoid(Z1_approx)
A1_exact = sigmoid(Z1_exact)

# print("Sieve function approximate result:")
# print(A1_approx[0])
# print("The exact result of np.dot() function:")
# print(A1_exact[0])

# print(f"A1_approx[0] - A1_exact[0]: \n{A1_approx[0] - A1_exact[0]}")

# print(f"type(A1_approx[0, 0]): \n{type(A1_approx[0, 0])}")
# print(f"type(A1_exact[0, 0]): \n{type(A1_exact[0, 0])}")

data_W2 = [
    [ -0.70062102,  -1.73822194,   0.7482715 ],
    [ -5.33871641,   3.50090099,  -1.78196288],
    [ -3.39542059,  -7.40653111,   7.74764896],
    [  1.3783175,    1.05236257,   1.44275999],
    [  0.04902521,   0.09071452,  -0.48090421],
    [  0.25878724,   1.1259403,   -0.03048146],
    [ -2.78022586,  -0.190776,    -0.32172248],
    [  2.93979288,   6.98394098,  -7.65726154],
    [ 11.00255442, -13.75994926,  -2.95656856],
    [ -0.60777718,   1.2462531,    0.51947335]]

# 創建 NumPy 陣列，並指定資料型態為 float32
W2 = np.array(data_W2, dtype=np.float32)

num_samples = 30
input_dim = 10
output_dim = 3
int_bits = 4
frac_bits = 28


data_b2 = [[-2.78005376, 0.44908536, -1.91727015]]

# 創建 NumPy 陣列，並指定資料型態為 float32
b2 = np.array(data_b2, dtype=np.float32)

W2_B = float_to_fixed_point_binary(W2, int_bits, frac_bits)

# 繪製輸入的exponent大小分佈
# print(f"count_elements(exponent(A1_exact)): {count_elements(exponent(A1_exact))}")
plot_exponent_counts(exponent_normalization(count_elements(exponent(A1_exact))))

# Results calculated using the Sieve function
Z2_approx, before_temp, after_temp = Sieve(A1_approx, W2_B, frac_bits, p=p, im=im, r=r)
Z2_approx = Z2_approx + b2
befor_count += before_temp
after_count += after_temp

# Accurate result calculated using np.dot()
Z2_exact = np.dot(A1_exact, W2) + b2

# print("Sieve function approximate result:")
# print(Z2_approx[0])
# print("The exact result of np.dot() function:")
# print(Z2_exact[0])

# print(f"Z2_approx[0] - Z2_exact[0]: \n{Z2_approx[0] - Z2_exact[0]}")

A2_approx = sigmoid(Z2_approx)
A2_exact = sigmoid(Z2_exact)

# print(f"A2_approx[0]: \n{A2_approx[0]}")
# print(f"A2_exact[0]: \n{A2_exact[0]}")

# print(f"A2_approx[0] - A2_exact[0]: \n{A2_approx[0] - A2_exact[0]}")

predictions_approx = np.argmax(A2_approx, axis=1)
predictions_exact = np.argmax(A2_exact, axis=1)

# print(f"predictions_approx: \n{predictions_approx}")
# print(f"predictions_exact: \n{predictions_exact}")
accuracy = np.mean(predictions_approx == predictions_exact)
print(f"accuracy: \n{accuracy * 100}%")
befor_count += before_temp
after_count += after_temp
print(f"befor_count: \n{befor_count}")
print(f"after_count: \n{after_count}")
save_count = befor_count - after_count
save_rate = save_count / befor_count
# print(f"save_count: \n{save_count}")
print(f"save_rate: \n{save_rate * 100}%")
