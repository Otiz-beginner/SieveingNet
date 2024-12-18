# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:06:00 2024

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
            BCB = np.unpackbits(np.array([val], dtype='>u2').view(np.uint8))[-(int_bits + frac_bits):]
            TCB = BCB2TCB(BCB)
            return TCB
            # >：Big-endian，u：Unsigned，2：two words（16bits），4: four words（32bits）
        else:
            # 對於正數，轉成binary最後再加上 - 號
            val = -val  # 先取絕對值
            BCB = np.unpackbits(np.array([val], dtype='>u2').view(np.uint8))[-(int_bits + frac_bits):]
            TCB = BCB2TCB(BCB)
            return -TCB
    
    # 將每個元素轉換為二進制
    binary_matrix = np.array([to_binary(val) for val in scaled_arr.flatten()])
    
    # 重塑為原始形狀並增加額外的維度
    result_shape = arr.shape + (int_bits + frac_bits,)
    binary_matrix = binary_matrix.reshape(result_shape)
    
    return binary_matrix.astype(np.int8)

# 測試範例
input_array = np.array([[12.625, 0.02506763], [-0.04475388, 0.07006475]])
start_time = time.time()
for t in range(1):
    binary_result = float_to_fixed_point_TCB(input_array, int_bits=5, frac_bits=11)
end_time = time.time()

print(binary_result)

print(f"total time: {end_time - start_time}")

print(f"type(binary_result): {type(binary_result[0, 0, 0])}")





