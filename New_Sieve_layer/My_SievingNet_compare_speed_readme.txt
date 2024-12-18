My_SievingNet_compare_speed: 
是直接使用np.frexp沒有修改(浮點數為0的話exponent為0)
，跑速度方面有差距，但不明顯。
My_SievingNet_compare_speed_2: 
np.frexp變成跟我自己寫的的exponent效果一樣(浮點數為0的話exponent為-127)，並且比較速度方面使用了numba函式庫中的jit編譯。

我現在有兩個問題: 
1. 是不是np.frexp對於輸入浮點數如果為0返回的exponent為0，但如果我利用自己寫的exponent函式會是返回-127，如果是的話我可以怎麼修改才能使得np.frexp也返回-127呢?並且如果可修改的話sieve_conventional跟sieveingnet.sieve要怎麼修改呢

2. 我跑了比較速度的函式，但發現conventional_times沒有比sieveingnet_times慢多少剩摯友時候還比較快，這是為什麼呢，因為我期望的是sieve_conventional需要跑迴圈所以會比較慢，sieveingnet.sieve因為平行化會比較快，請問我哪裡搞錯了嗎?或是其實我沒有搞錯，只是sieveingnet.sieve哪邊還可以再優化呢?

時間是最大的限制
鬧鐘提醒就像中斷
我不用一直看to do list


對於下面的程式碼，我最終的目標是想要比較Sieve_conventional跟類別SievingNet中的Sieve的運算速度差別，但在這之前我有幾件事情想先做做。
第1: Sieve_conventional跟SievingNet請幫我改使用numpy來撰寫。接著對於Sieve函式還有類別SievingNet中的Sieve中的分別的
""
for i in range(n):
                if W[i][k] != 0:
                    if exponent(A[i]) >= sivingThreshold:
                        S_kth += A[i]
""
這一層迴圈都幫我使用numpy撰寫成失量化操作。
另外SievingNet中的Sieve請幫我將"for k in range(B):"這一層for迴圈使用numpy失量化操作，至於Sieve_conventional則是幫我保留這一層對應的for迴圈。

第2: 請幫我撰寫一個可以隨機生成A和W_fixed_B的陣列，A的長度幫我由4, 8, 16, 32一直上升到512做測試，W_fixed_B第二層(1-axis)固定為32(例如目前就為4)，W_fixed_B第一層(0-axis)需要跟A陣列長度相同。

第3: 最後幫我做一個測試Sieve_conventional跟類別SievingNet中的Sieve的運算速度的程式碼。frac_bits、im三者數值都固定，結果的部分print出來並輸出到excel檔案最做成表格，感謝。

如果有任何設計細節問題歡迎隨時問我，直到清楚任何設計細節後再開始回答我。


1. A為float32，W_fixed_B應該可以是int8就行，忘記講隨機生成時裡面的數據只有可能為0或1或-1(W_fixed_B)。
2. 改為32
3. A介於3~-3之間，W_fixed_B只可能01或-1或1。
4. 10次
5. 大小、運行時間、speedup都包括。


對於下面的程式碼，功能為比較Sieve_conventional跟類別SievingNet中的Sieve的運算速度差別，一個在決定Siving Threshold時是使用本次總和輸出S的exponent，另一個是使用前一次總和輸出S_previous的exponent，所以一個需要循環分次做累加做B次，另一個可以平行化一次做。最後在各種不同輸入nodes數量下做測試，看需要花多久時間。這是這一個程式碼的用意，現在我想要把這一個程式碼寫成類似下面範例風格的的pseudo code，盡量好看簡明清楚一點，不需用非常像原本程式碼那麼詳細，至少只需要把我上面成數這一個程式瑪用途的重點有描寫道就好。

如果有任何設計細節問題歡迎隨時問我，直到清楚任何設計細節後再開始回答我。

程式碼: 

pseudo code範例: 

1. 希望，但希望簡單易懂就好。
2. 希望納入，但是簡短帶過。
3. 不用，直接利用exponent()的樣子來使用就行。
4. 您是否有推薦的方法，不然也可以使用文字。