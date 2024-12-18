# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 21:25:44 2024

@author: Otiz
"""
import numpy as np
import time
import pandas as pd

def exponent(x):
    if isinstance(x, (float, np.float32, np.float64)):
        return int(np.frexp(x)[1]) - 1
    else:
        return np.frexp(x)[1].astype(int) - 1

def Sieve_conventional(A, W, frac_bits, im):
    B = W.shape[1]
    S = 0.0
    for k in range(B):
        S_kth = 0.0
        order = ((B-k)-1)
        power = order - frac_bits
        sivingThreshold = exponent(S) - im * (exponent(S) + 127) - power
        mask = (W[:, k] != 0) & (exponent(A) >= sivingThreshold)
        S_kth = np.sum(A[mask])
        S += (S_kth * (2**power))
    return S

class SievingNet:
    def __init__(self):
        self.S_previous = 0.0
        
    def Sieve(self, A, W, frac_bits, im):
        B = W.shape[1]
        orders = np.arange(B-1, -1, -1)
        powers = orders - frac_bits
        
        S_previous_exp = exponent(self.S_previous)
        sieving_thresholds = S_previous_exp - im * (S_previous_exp + 127) - powers

        exp_A = exponent(A)[:, np.newaxis]
        masks = (W != 0) & (exp_A >= sieving_thresholds)

        S_kth = np.sum(A[:, np.newaxis] * masks, axis=0)
        S = np.sum(S_kth * (2.0 ** powers))

        self.S_previous = S
        return S

def generate_random_data(size):
    A = np.random.uniform(-3, 3, size).astype(np.float32)
    W_fixed_B = np.random.choice([-1, 0, 1], size=(size, 32)).astype(np.int8)
    return A, W_fixed_B

def run_performance_test():
    frac_bits = 28
    im = 1
    sizes = [128, 256, 512, 1024, 2048]
    results = []

    for size in sizes:
        A, W_fixed_B = generate_random_data(size)
        
        times = 100
        
        # Test Sieve_conventional
        start_time = time.time()
        for _ in range(times):
            Sieve_conventional(A, W_fixed_B, frac_bits, im)
        conv_time = (time.time() - start_time) / times

        # Test SievingNet.Sieve
        sieving_net = SievingNet()
        start_time = time.time()
        for _ in range(times):
            sieving_net.Sieve(A, W_fixed_B, frac_bits, im)
        net_time = (time.time() - start_time) / times

        speedup = conv_time / (net_time)
        results.append({
            'Size': size,
            'Conventional Time': conv_time * 1e5,
            'SievingNet Time': net_time * 1e5,
            'Speedup': speedup
        })

    df = pd.DataFrame(results)
    print(df)
    df.to_excel('sieving_performance.xlsx', index=False)

if __name__ == "__main__":
    run_performance_test()