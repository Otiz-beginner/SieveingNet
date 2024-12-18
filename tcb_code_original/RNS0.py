#!/usr/bin/env python
# coding: utf-8
# Residue Number System
# TestLab, NCUE, 2019/09/02
# 
# 1. Fixed-Point Factors f=<int, mentissa> with int-bit integer and mentissa-bit mentissa in domain F are transferred to Signed integers s in domain S by s = (2^m)*f
# 2. Transfer Signed number to Unsigned number by u = (s<0) : s+M ? s;
# 3. Select dynamic range (DR) M = LCM(m1, m2, .., mk) > 2^(int+mentissa)
# 4. For sign detection in RNS, add a redundant mr = 2^(L+1) if gcd(M, mr)=2^L. The Lth bit of u will be called "parity bit for sign detection" of u, denoted as p(u).
# 5. Examples: (255, 256, 257; 512) or (255, 257; 2)
# 6. The modAdd() and modMul() for modulus 2^(L+1) can be extended from those for 2^L.
# 6. CRT to find x from (m1, m2, .., mk):
# 7. Mi = M/mi; Wi = mulinv(Mi, mi), i.e., WiMi % mi = 1; MiWi are computed in advance and then constant!
# 8. u = (u1, u2, .., uk) = U % M; wehre U = Sigma_i(ui MiWi) %M
# 9. When mr added, DR M' will be M' = 2M. u' = (u1, u2, .., uk, ur) = U' % (2M); where U' = Sigma_i (ui', Mi'Wi')
# 10. We can derive that p(u) = (p(U))

# Great Common Divisor
def gcd(m, n):
    if n == 0:
        return m
    else:
        return gcd(n, m % n)

# lowest common multiple
def lcm(a, b):
    c = a*b
    d = c/gcd(a, b)
    if c>0 :
        return d
    else:
        return -d

# Signed integers [-M/2..M/2] shifted to unsigned Binary [0..M)
def I2B(a, M):
    H = M/2
    if (a<-H) | (a>H):
        print("Error: Signed Number Overflow!")
    if a<0:
        return a + M//2
    else:
        return a

# Unsigned binary to a balanced integer ring
def B2I(a, M):
    H = M/2
    if (a<0) | (a>=M):
        print("Error: Unsigned Number Overflow!")
    if a > H:
        return a - M//2
    else:
        return a

# Binary to residue vector
def B2R(b, m):
    r = []
    for i in range(len(m)):
        r.append(b % m[i])
    return r

# Extended Eculidian Algorithm, extended gcd
# 1. Eculidian Theorem: gcd(a, b) = gcd(b%a, a) = g
# 2. Bezout Identity: g = (b%a)x + ay
# 3. Recursive Relation: g = (y-(b//a))a + xb
def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, x, y = egcd(b % a, a)
        return (g, y - (b // a) * x, x)

# modular multiplicative inverse
def mulinv(a, b):
    """return x such that (x * a) % b == 1"""
    g, x, _ = egcd(a, b)
    if g == 1:
        return x % b

# Residue to Binary
def R2B(r, m):
    M = 1
    for i in range(len(m)):
        M = M * m[i]
    Q = []
    for i in range(len(m)):
        Q.append(M//m[i])
    b = 0
    for i in range(len(m)):
        b = b + r[i] * Q[i] * mulinv(Q[i], m[i])
    return b % M

# Modular Addition
def modAdd(A, B, m):
    S = []
    for i in range(len(A)):
        S.append((A[i]+B[i])%m[i])
    return S

# Modular Subtraction
def modSub(A, B, m):
    D = []
    for i in range(len(A)):
        D.append((A[i]-B[i])%m[i])
    return D


# Modular Multiplication
def modMul(A, B, m):
    S = []
    for i in range(len(A)):
        S.append((A[i]*B[i])%m[i])
    return S

# Examples: Checking Multiply-Accumulation (MAC)
def MAC(X, Y):
    S = 0
    for i in range(len(X)):
        S += X[i] * Y[i]
    return S

# MAC for RNS
def MACr(A, B, m):
    S = B2R(0, m)
    for i in range(len(A)):
        a, b = A[i], B[i]
        S = modAdd(S, modMul(a, b, m), m)
    return S
      