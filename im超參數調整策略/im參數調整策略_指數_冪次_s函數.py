# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 21:50:26 2024

@author: Otiz
"""

import numpy as np
import matplotlib.pyplot as plt

def inverse_exponential(x, im_start, im_end, k):
    return im_start + (im_end - im_start) * (1 - np.exp(-k * x))

def power_function(x, im_start, im_end, p):
    return im_start + (im_end - im_start) * np.power(x, p)

def sigmoid_function(x, im_start, im_end, k):
    return im_start + (im_end - im_start) * (1 / (1 + np.exp(-k * (x - 0.5))))

def plot_im_growth(function, im_start, im_end, param, total_epochs):
    x = np.linspace(0, total_epochs, 100)
    y = function(x / total_epochs, im_start, im_end, param)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(f"apxLvl Growth: {function.__name__}")
    plt.xlabel("Validation accuracy")
    plt.ylabel("apxLvl Value")
    plt.grid(True)
    plt.show()

def main():
    print("apxLvl 增長比較工具")
    
    while True:
        print("\n選擇函數類型:")
        print("1. 反向指數")
        print("2. 幂函數")
        print("3. Sigmoid")
        print("4. 退出")
        
        choice = input("輸入你的選擇 (1-4): ")
        
        if choice == '4':
            break
        
        im_start = float(input("輸入 im_start (0-1): "))
        im_end = float(input("輸入 im_end (0-1): "))
        total_epochs = int(input("輸入總 epoch 數: "))
        
        if choice == '1':
            k = float(input("輸入 k 值: "))
            plot_im_growth(inverse_exponential, im_start, im_end, k, total_epochs)
        elif choice == '2':
            p = float(input("輸入 p 值: "))
            plot_im_growth(power_function, im_start, im_end, p, total_epochs)
        elif choice == '3':
            k = float(input("輸入 k 值: "))
            plot_im_growth(sigmoid_function, im_start, im_end, k, total_epochs)
        else:
            print("無效的選擇，請重試。")

if __name__ == "__main__":
    main()