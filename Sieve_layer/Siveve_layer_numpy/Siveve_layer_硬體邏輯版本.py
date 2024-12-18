# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:58:18 2024

@author: user
"""


import struct

def exponent(x):
    packed_data = struct.pack('f', x)
    binary_representation = struct.unpack('I', packed_data)[0]
    exponent_mask = 0xFF
    binary_representation_no_mantissa = binary_representation // 0x7FFFFF
    exponent_part = binary_representation_no_mantissa & exponent_mask
    return exponent_part - 127



def Sieve(A, W, W_p, p, im, r):
    n_l = len(W) # last layer node
    n_n = len(W[0]) # next layer node
    B   = len(W[0][0]) # How many bits for a number
    samples = len(A) # how many samples at one
    X = []
    for s in range(samples): 
        x_sample = []
        for j in range(n_n): 
            x = 0.0
            for i in range(n_l): 
                x_item = 0.0
                for k in range(B): 
                    order = ((B-k)-1)
                    power = order - W_p
                    sivingThreshold = r + im * exponent(x) - order - p
                    if W[i][j][k] != 0:
                        if exponent(A[s][i]) >= sivingThreshold:
                            x_item += (A[s][i] * (2**power))
                x += x_item
            x_sample.append(x)
        X.append(x_sample)
    return X

A = [[12.625, 4.75, 0], 
    [12.625, 0.03125, 0]] 

W = [[[1, 0, 1, 0], [0, 1, 0, 1]],  
     [[1, 0, 1, 0], [0, 1, 0, 1]], 
     [[1, 0, 1, 0], [0, 1, 0, 1]] ]

W_p = 2 
p = 5 
im = 1 
r = 0 
print(Sieve(A, W, W_p, p, im, r))