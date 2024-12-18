### Ternary Coded Binary (TCB) Library
# TestLab, NCUE, 2019/09/02
# 

# Binary coded to Ternary coded Converter
def Bin2TCB(B):
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
                T= '0+' + T
                break
        elif Q=='11':
            if X=='0':
                Q='011'
            elif X=='1':
                Q=''
                S=1
                T = '00-' + T
            else:
                T = '++' + T
                break
        elif Q=='011':
            if X=='0':
                T = '00++' + T
                Q = ''
            elif X=='1':
                T = '0-' + T
                Q = '11'
            else:
                T = '0++' + T
                break
        elif Q=='0':
            S = 0
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

# Arithmetic Weight of a Binary Number
def Weight(B):
    return SWeight(Bin2TCB(B))

#### Arithmetic Weight of a Ternary/Binary String
def SWeight(S):
    z = 0
    for c in S:
        if c == '0':
            z += 1
    return(len(S)-z)

