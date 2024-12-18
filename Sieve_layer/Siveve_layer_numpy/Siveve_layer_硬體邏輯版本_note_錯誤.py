# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 22:09:15 2024

@author: Otiz
"""
import time
import numpy as np

def exponent(x):
    dtype = x.dtype
    if dtype == np.float32:
        # For float32
        packed_data = x.view(np.uint32)
        exponent_mask = 0x7F800000
        exponent_shift = 23
        bias = 127
    elif dtype == np.float64:
        # For float64
        packed_data = x.view(np.uint64)
        exponent_mask = 0x7FF0000000000000
        exponent_shift = 52
        bias = 1023
    else:
        raise TypeError("Unsupported data type. Use float32 or float64.")

    # Extract the exponent part
    exponent_part = (packed_data & exponent_mask) >> exponent_shift
    return exponent_part - bias # return 1D array

def Sieve(A, W, W_p, p, im, r):
    A = np.array(A, dtype=np.float32)
    W = np.array(W, dtype=np.int8)
    W_p = np.int32(W_p)
    p = np.int32(p)
    im = np.float32(im)
    r = np.int32(r)
    
    n_l, n_n, B = W.shape
    samples = A.shape[0]
    X = np.zeros((samples, n_n), dtype=np.float32)
    
    bit_index = np.arange(B - 1, -1, -1, dtype=np.float32)
    # Calculate powers of 2 for the bit positions, ensure floating point for negative exponents
    power =  bit_index - W_p # power.shape = (B,)

    for j in range(n_n):
        for i in range(n_l):
            # Calculate exponent for A[:, i] and ensure broadcasting
            exponents = exponent(A[:, i])[:, np.newaxis] # use np.newaxis from 1D to 2D
            # Calculate sivingThreshold for each sample and bit
            sivingThreshold = r + im * exponents - bit_index - p
            # Create mask where W[i, j] is non-zero
            mask = W[i, j] != 0 # mask.shape = (B,)
            # Calculate contribution for each bit, ensure floating point arithmetic
            contribution = np.where(
                exponents >= sivingThreshold,
                A[:, i, np.newaxis] * (2.0**power),  #  A[:, i, np.newaxis].shape =(samples, 1)
                0.0
            ) # contribution.shape = (samples, B)
            # Sum contributions across the bits and apply mask
            X[:, j] += np.sum(mask * contribution, axis=1)
    
    return X

A = [[12.625, 4.75, 0], 
     [12.625, 0.03125, 0]]

W = [[[1, 0, 1, 0], [0, 1, 0, 1]],  
     [[1, 0, 1, 0], [0, 1, 0, 1]], 
     [[1, 0, 1, 0], [0, 1, 0, 1]]]

W_p = 2 
p = 5 
im = 0.0 
r = 0 
start_time = time.time()
for k in range(784*512):
    Sieve(A, W, W_p, p, im, r)
end_time = time.time()
print(end_time - start_time)
print(Sieve(A, W, W_p, p, im, r))

# S = [[43.4375, 21.71875], 
#      [31.640625, 15.8203125]]


