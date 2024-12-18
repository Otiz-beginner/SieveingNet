# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:37:02 2024

@author: user
"""

import numpy as np
import math

def exponent(x):
    _, exponent_part = math.frexp(x)
    return exponent_part - 1

def Sieve(A, W, B, W_p, p, ip, r):
    A = np.array(A, dtype=np.float32)
    W = np.array(W, dtype=np.int8)
    
    n_l = W.shape[1]
    batch = A.shape[0]
    X = np.zeros((batch, n_l), dtype=np.float32)
    
    powers = np.array([2 ** ((B - k - 1) - W_p) for k in range(B)], dtype=np.float32)
    W_nonzero_mask = W != 0
    
    for s in range(batch):
        for i in range(n_l):
            x = 0.0
            for k in range(B):
                order = (B - k) - 1
                sivingThreshold = r + ip * exponent(x) - order - p
                
                valid_indices = np.where(W_nonzero_mask[:, i, k])[0]
                if valid_indices.size > 0:
                    A_sub = A[s, valid_indices]
                    exponents = np.array([exponent(a) for a in A_sub], dtype=np.float32)
                    valid_indices = valid_indices[exponents >= sivingThreshold]
                    if valid_indices.size > 0:
                        x_kth = np.sum(A[s, valid_indices])
                        x += x_kth * powers[k]
            X[s, i] = x
    return X

def float_to_fixed_point_binary(arr: np.ndarray, int_bits: int = 1, frac_bits: int = 15) -> np.ndarray:
    scale = 2 ** frac_bits
    scaled_arr = np.round(arr * scale).astype(np.int32)
    
    max_val = (2 ** (int_bits + frac_bits)) - 1
    min_val = -(2 ** (int_bits + frac_bits))
    scaled_arr = np.clip(scaled_arr, min_val, max_val)
    
    def to_binary(val: int) -> list[int]:
        if val < 0:
            val += 1 << (int_bits + frac_bits)
        return [int(x) for x in format(val, f'0{int_bits + frac_bits}b')]
    
    binary_matrix = np.array([to_binary(val) for val in scaled_arr.flatten()])
    result_shape = arr.shape + (int_bits + frac_bits,)
    binary_matrix = binary_matrix.reshape(result_shape)
    
    return binary_matrix.astype(np.int8)

# Example Usage
A = np.random.rand(1, 2).astype(np.float32)
W = np.random.uniform(0, 1e-1, (2, 2))
print(f"A = {A}")
print(f"W = {W}")
print("-------------------------------------------------------")
print(f"correct answer: {np.dot(A, W)}")
# print(f"correct answer type: {type(np.dot(A, W)[0, 0])}")
W = float_to_fixed_point_binary(W)

B = 16
W_p = 15
p = 10
ip = 0
r = 0

print("-------------------------------------------------------")
print(f"my answer: {Sieve(A, W, B, W_p, p, ip, r)}")
# print(f"my answer type: {type(Sieve(A, W, B, W_p, p, ip, r)[0, 0])}")
print("-------------------------------------------------------")