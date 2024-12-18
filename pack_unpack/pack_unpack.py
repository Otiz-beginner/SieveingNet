# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 22:09:15 2024

@author: user
"""

import struct

# 給inputs

# 打包數字 64.3 為單精度浮點數 (4 bytes)
packed_data = struct.pack('f', 12.625)
print(packed_data)

# 解包數據 ***注意: 出來會是一個tuple
unpacked_data = struct.unpack('f', packed_data)
print(unpacked_data)

# 將打包的二進制數據轉換為整數 ***注意: [0]可以取第一個數字
binary_representation = struct.unpack('I', packed_data)[0]
print(f"Binary representation (Decimal): {(binary_representation)}")
print(f"Binary representation (bin): {bin(binary_representation)}")

# 提取 mantissa 部分 (取最低的 23 位)
mantissa_mask = 0x7FFFFF
mantissa_part = binary_representation & mantissa_mask
print(f"mantissa part (binary): {bin(mantissa_part)}")
# print(f"mantissa part (hex): {hex(mantissa_part)}")

# 提取 exponent 部分
exponent_mask = 0xFF
binary_representation = binary_representation // 0x7FFFFF
exponent_part = binary_representation & exponent_mask
print(f"exponent part (binary): {bin(exponent_part)}")
print(f"exponent part (Decimal): {exponent_part}")
print("-------------------------------------------------------")

# # 給權重

# # 打包數字 12.625 為unsign char (1 bytes)
# packed_data_2 = struct.pack('B', 40)
# print(packed_data_2)

# # 解包數據 ***注意: 出來會是一個tuple
# unpacked_data_2 = struct.unpack('B', packed_data_2)[0]
# print(f"exponent (binary): {int(unpacked_data_2)}")

# W.append(bin(int(unpacked_data_2)))
# print(f"type of W[0]{type(W[0])}")
# print(f"W[0]: {W[0]}")
# print("-------------------------------------------------------")














