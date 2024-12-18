### Ternary Coded Binary (TCB) Library
# TestLab, NCUE, 2019/09/02
# 

# Binary to TCB Converter
def decimal_to_TCB_string(B):
    B=bin(B)
    Q=''
    T=''
    L=len(B)
    S=0
    for i in range(L-1, 0, -1):
        X=B[i]
        if   Q=='':
            if S==0:
                if X=='0':
                    T = '0' + T
                elif X=='1':
                    Q = '1'
                else:
                    break
            else:
                if X=='0':
                    S = 0
                    Q='0'
                elif X=='1':
                    T = '0' + T
                    Q=''
                else:
                    T = '+' + T
                    break
        elif Q=='1':
            if X=='0':
                T= '0+' + T
                Q=''
            elif X=='1':
                Q='11'
            else:
                T= '+' + T 
                break
        elif Q=='11':
            if X=='0':
                Q='011'
            elif X=='1':
                Q=''
                S=1
                T = '00-' + T
            else:
                T = '+0-' + T 
                break
        elif Q=='011':
            if X=='0':
                T = '0+0-' + T 
                Q = ''
            elif X=='1':
                T = '0-' + T
                Q = '11'
            else:
                T = '+0-' + T 
                break
        elif Q=='0':
            if X=='0':
                T = '0+' + T
                Q = ''
            elif X=='1':
                Q = '11'
            else:
                T = '+' + T
                break
        else:
            print("Error")
    return(T)

def decimal_to_BCB_string(decimal_number):
    # 將十進制數字轉換為二進制字串，並去掉 '0b' 前綴
    binary_string = bin(decimal_number)[2:]
    return binary_string

# Arithmetic Weight of a Binary Number
def TCB_Weight(B):
    TCB_string = decimal_to_TCB_string(B)
    z = 0
    for c in TCB_string:
        if c == '0':
            z += 1
    return(len(TCB_string)-z)

def BCB_Weight(B):
    BCB_string = decimal_to_BCB_string(B)
    z = 0
    for c in BCB_string:
        if c == '0':
            z += 1
    return(len(BCB_string)-z)


test_number = 7

print(f"decimal_to_BCB_string(test_number): {decimal_to_BCB_string(test_number)}")
print(f"BCB_Weight(test_number): {BCB_Weight(test_number)}")
print(f"decimal_to_TCB_string(test_number): {decimal_to_TCB_string(test_number)}")
print(f"TCB_Weight(test_number): {TCB_Weight(test_number)}")

BCB_AW_rate_list = []
TCB_AW_rate_list = []
for i in range(1000000):
    TCB_string = decimal_to_TCB_string(i)
    original_L = len(TCB_string)
    new_BCB_L = BCB_Weight(i)
    new_TCB_L = TCB_Weight(i)
    # print(f"i: {i}")
    # print(f"original_L: {original_L}")
    # print(f"new_BCB_L: {new_BCB_L}")
    # print(f"new_TCB_L: {new_TCB_L}")
    BCB_AW_rate = new_BCB_L / original_L
    TCB_AW_rate = new_BCB_L / original_L
    BCB_AW_rate_list.append(BCB_AW_rate)
    TCB_AW_rate_list.append(TCB_AW_rate)
    # print(f"BCB_AW_rate: {BCB_AW_rate}")
    # print(f"TCB_AW_rate: {TCB_AW_rate}")
    # print("---------------------------")

# TCB to Binary Converter
def TCB2Bin(T):
    B=0
    L = len(T)
    for i in range(L):
        if T[i] == '0':
            t = 0
        elif T[i] == '+':
            t = 1
        else:
            t = -1
        B += t * (1 << (L-1-i))
    return(B)
    

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def plot_sorted_aw_distribution(BCB_AW_rate_list, TCB_AW_rate_list):
    # 計算每個值的出現頻率
    bcb_counter = Counter(BCB_AW_rate_list)
    tcb_counter = Counter(TCB_AW_rate_list)
    
    # 轉換為概率並排序
    total = len(BCB_AW_rate_list)
    bcb_probs = sorted([(k, v / total) for k, v in bcb_counter.items()])
    tcb_probs = sorted([(k, v / total) for k, v in tcb_counter.items()])
    
    # 創建圖表
    plt.figure(figsize=(12, 8))
    
    # 繪製 BCB 曲線
    bcb_x, bcb_y = zip(*bcb_probs)
    plt.plot(bcb_x, bcb_y, 'b-', label='BCB')
    
    # 繪製 TCB 曲線
    tcb_x, tcb_y = zip(*tcb_probs)
    plt.plot(tcb_x, tcb_y, 'r-', label='TCB')
    
    # 設置標題和標籤
    plt.title('Arithmetic-Weight Distribution', fontsize=16)
    plt.xlabel('Arithmetic-Weight rate', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    
    # 添加圖例
    plt.legend(fontsize=12)
    
    # 調整軸的範圍
    plt.xlim(0, 1)
    plt.ylim(0, max(max(bcb_y), max(tcb_y)) * 1.1)
    
    # 顯示網格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存圖表
    plt.savefig('sorted_arithmetic_weight_distribution.png', dpi=300, bbox_inches='tight')
    
    # 顯示圖表
    plt.show()

# 假設 BCB_AW_rate_list 和 TCB_AW_rate_list 已經計算完成
plot_sorted_aw_distribution(BCB_AW_rate_list, TCB_AW_rate_list)
    

