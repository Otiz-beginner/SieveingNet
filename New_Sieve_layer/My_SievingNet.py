# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 22:00:21 2024

@author: Otiz
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import math

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

class SieveingNet_conventional():
    def __init__(self, samples, n_n):
        self.S_privious = np.zeros((samples, n_n), dtype=np.float32)

    def sieve(self, A, W, frac_bits, im):
        A = np.array(A, dtype=np.float32)
        W = np.array(W, dtype=np.int8)
        frac_bits = np.int32(frac_bits)
        im = np.float32(im)

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
            exponents_S_p = exponent(self.S_privious)[:, np.newaxis, :]  # shape: (samples, 1, n_n)
            sieving_threshold = exponents_S_p - im * (exponents_S_p - (-127)) + (32 - bit_orders[k]) # shape: (samples, 1, n_n)

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
        self.S_privious = X
        return X, mask_non_zero_count, contribution_non_zero_count