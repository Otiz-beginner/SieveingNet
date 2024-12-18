# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 22:09:15 2024

@author: Otiz
"""

import struct

def exponent(x):
    packed_data = struct.pack('f', x)
    binary_representation = struct.unpack('I', packed_data)[0]
    # 提取 exponent 部分
    exponent_mask = 0xFF
    binary_representation_no_mantissa = binary_representation // 0x7FFFFF
    exponent_part = binary_representation_no_mantissa & exponent_mask
    # if exponent_part != 0: # because float value if is 0 the exponent will be no bias
    #    exponent_part_no_bias = exponent_part - 127 
    # return exponent_part_no_bias
    return exponent_part - 127



def Sieve(A, W, B, W_p, p, ip, r):
    n = len(W)
    S = 0.0
    for k in range(B):# k in [0, 1, 2, 3]
        S_kth = 0.0
        order = ((B-k)-1)
        power = order - W_p
        sivingThreshold = r + ip * exponent(S) - order - p
        print(f"第{k+1}round")
        print(f"r = {r}")
        print(f"ip * exponent(S) = {ip * exponent(S)}")
        print(f"order = {order}")
        print(f"power = {power}")
        print(f"sivingThreshold = {sivingThreshold}")
        print(f"exponent(A[0]) = {exponent(A[0])}")
        print(f"exponent(A[1]) = {exponent(A[1])}")
        for i in range(n):
            if W[i][k] != 0:
                if exponent(A[i]) >= sivingThreshold:
                    S_kth += A[i]
        S += (S_kth * (2**power))
        print(f"S = {S}")
        print()
    return S

# A = [12.625, 4.75] # type float normal case
A = [12.625, 0.03125] # type float small case


W = [[0, 1, 0, 1], # type int
     [0, 1, 0, 1]]

# 43.4375
# 31.640625 but now answer is 31.625
B = 4
W_p = 2
p = 5 # represent how many small of input can be(the bits after point) 
ip = 0 # i is the impact of exponent(S) when sieving
r = 0 
print("-------------------------------------------------------")
print(Sieve(A, W, B, W_p, p, ip, r))















