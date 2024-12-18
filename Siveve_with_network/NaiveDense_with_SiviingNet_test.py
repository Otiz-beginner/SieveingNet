import numpy as np
import math
import time
import matplotlib.pyplot as plt
from keras.datasets import mnist


def BCB2TCB(BCB):
    state=0
    L = BCB.shape[0]
    # print(f"BCB: {BCB}")
    BCB_new = np.zeros((L+1, ))
    BCB_new[0] = 8
    BCB_new[1:] = BCB[:]
    # print(f"BCB_new: {BCB_new}")
    TCB = np.zeros((L, ))
    # print(f"TCB: {TCB}")
    for i in range(L, -1, -1): # i in [8, 7, 6, 5, 4, 3, 2, 1, 0]
        j = i-1 # j in [7, 6, 5, 4, 3, 2, 1, 0, -1]
        X=BCB_new[i]
        # print(f"處理輸入第{i}個")
        # print("-------------------")
        # print(f"state: {state}")
        # print(f"輸入為{X}")
        # print(f"j為{j}")
        if state==0:
            if X==0:
                TCB[j] = 0
            elif X==1:
                state = 1
        elif state==1:
            if X==0:
                TCB[j:j+2]= [0, 1]
                state = 0
            elif X==1:
                state = 2
        elif state==2:
            if X==0:
                state = 5
            elif X==1:
                state = 3
                TCB[j:j+3] = [0, 0, -1]
        elif state==3:
            if X==0:
                state = 4
            elif X==1:
                TCB[j] = 0
                state = 3
        elif state==4:
            if X==0:
                TCB[j:j+2] = [0, 1]
                state = 0
            elif X==1:
                state = 2
            else:
                TCB[(j+1)]= 1
        elif state==5:
            if X==0:
                TCB[j:j+4] = [0, 1, 0, -1] 
                state = 0
            elif X==1:
                TCB[j:j+2] = [0, -1]
                state = 2
            else:
                TCB[(j+1):(j+1)+3] = [1, 0, -1]
        else:
            print("Error")
        # print(f"TCB: {TCB}")
        # print()
    return TCB

# int_bits和frac_bits是W的位元寬度，一開始就必須設定好
def float_to_fixed_point_TCB(arr, int_bits=4, frac_bits=28):
    # 計算轉換係數
    scale = 2 ** frac_bits
    
    # 將浮點數乘以係數
    scaled_arr = np.round(arr * scale).astype(np.int32)
    
    # 計算整數的最大值和最小值
    max_val = (2 ** (int_bits + frac_bits)) - 1
    min_val = -(2 ** (int_bits + frac_bits))
    
    # 確保數值在允許的範圍內
    scaled_arr = np.clip(scaled_arr, min_val, max_val)
    
    def to_binary(val):
        if val >= 0:
            # 對於正數，直接轉成BCB
            BCB = np.unpackbits(np.array([val], dtype='>u4').view(np.uint8))[-(int_bits + frac_bits):]
            TCB = BCB2TCB(BCB)
            return TCB
            # >：Big-endian，u：Unsigned，2：two words（16bits），4: four words（32bits）
        else:
            # 對於正數，轉成binary最後再加上 - 號
            val = -val  # 先取絕對值
            BCB = np.unpackbits(np.array([val], dtype='>u4').view(np.uint8))[-(int_bits + frac_bits):]
            TCB = BCB2TCB(BCB)
            return -TCB
    
    # 將每個元素轉換為二進制
    binary_matrix = np.array([to_binary(val) for val in scaled_arr.flatten()])
    
    # 重塑為原始形狀並增加額外的維度
    result_shape = arr.shape + (int_bits + frac_bits,)
    binary_matrix = binary_matrix.reshape(result_shape)
    
    return binary_matrix.astype(np.int8)

def exponent(x):
    if x.dtype == np.float32:
        packed_data = x.view(np.uint32) # 先當作無號數看待，右移才會是unsigned extend數值才不會錯
        exponent_mask = 0x7F800000
        exponent_part_bias = (packed_data & exponent_mask) >> 23
        exponent_part = exponent_part_bias.view(np.int32) - 127 # 最後再看待乘有號數，才會有負數
    elif x.dtype == np.float64:
        packed_data = x.view(np.uint64)
        exponent_mask = 0x7FF0000000000000
        exponent_part_bias = (packed_data & exponent_mask) >> 52
        exponent_part = exponent_part_bias.view(np.int64) - 1023
    else:
        raise ValueError("Only float32 and float64 are supported.")

    return exponent_part

def Sieve(A, W, im, r, frac_bits=28, p=10):
    A = np.array(A, dtype=np.float32)
    W = np.array(W, dtype=np.int8)
    frac_bits = np.int32(frac_bits)
    p = np.int32(p)
    im = np.float32(im)
    r = np.int32(r)

    n_l, n_n, B = W.shape
    samples = A.shape[0]

    # Initialize X with zeros
    X = np.zeros((samples, n_n), dtype=np.float32)

    # Pre-compute bit orders and powers
    bit_orders = np.arange(B-1, -1, -1, dtype=np.float32) # shape: (B,)
    powers = bit_orders - frac_bits # shape: (B,)

    # Expand dimensions for broadcasting
    A_expanded = A[:, :, np.newaxis]  # shape: (samples, n_l, 1)
    W_expanded = W[np.newaxis, :, :, :]  # shape: (1, n_l, n_n, B)
    
    # Compute exponents of A
    exponents_A = exponent(A)[:, :, np.newaxis]  # shape: (samples, n_l, 1)

    # Iterate over bits
    for k in range(B):
        # Compute sieving threshold
        exponents_X = exponent(X)[:, np.newaxis, :]  # shape: (samples, 1, n_n)
        sieving_threshold = r + im * exponents_X - bit_orders[k] - p # shape: (samples, 1, n_n)

        # Create mask where W is non-zero for the current bit
        mask = (W_expanded[:, :, :, k]).astype(np.float32)  # shape: (1, n_l, n_n)

        # Calculate contribution for the current bit
        contribution = np.where(
            exponents_A >= sieving_threshold,
            A_expanded * mask,
            0.0
        )  # shape: (samples, n_l, n_n)

        # Sum contributions across input nodes
        x_k = np.sum(contribution, axis=1)  # shape: (samples, n_n)

        # Update X
        X += x_k * (2.0 ** powers[k])

    return X



class NaiveDense:
    def __init__(self, input_size, output_size, activation): # ****待測試正確性****
        self.activation = activation
        self.W = np.random.uniform(0, 1e-1, (input_size, output_size))
        self.b = np.zeros(output_size)
        self.W_binary = np.random.uniform(0, 1e-1, (input_size, output_size, 32)) # 32是W的位元寬度寬度
        
    def __call__(self, inputs, im=0, r=0): # ****待測試正確性****
        Sieve_result = Sieve(input, self.W_binary, im, r)
        return self.activation(Sieve_result + self.b)

    @property
    def weights(self):
        return [self.W, self.b]


class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs, ip=0, r=0):
        x = inputs
        for layer in self.layers:
            x = layer(x, ip, r)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights

class BatchGenerator:
    def __init__(self, images, labels, batch_size=4):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return images, labels

def one_training_step(model, images_batch, labels_batch, learning_rate=1e-2, im=0, r=0): # *new*
    predictions = model(images_batch, im, r) # 前向傳播1(為了算loss)
    per_sample_losses = np.mean(-np.log(predictions[np.arange(len(labels_batch)), labels_batch]))
    gradients = compute_gradients(model, images_batch, labels_batch, im, r)# 計算梯度
    update_weights(model, gradients, learning_rate)
    return per_sample_losses

def compute_gradients(model, inputs, targets, im, r):
    activations = [inputs]
    for layer in model.layers:
        activations.append(layer(activations[-1], im, r))# 前向傳播2(為了算與正確label差距)

    predictions = activations[-1]
    one_hot_targets = np.eye(predictions.shape[1])[targets]# 正確label
    delta = (predictions - one_hot_targets) / predictions.shape[0]# 計算與正確label差距

    gradients = []
    for i in reversed(range(len(model.layers))):# 反向傳播
        layer = model.layers[i]
        grad_w = np.dot(activations[i].T, delta)
        grad_b = np.sum(delta, axis=0)
        gradients.append((grad_w, grad_b))

        if i > 0:
            delta = np.dot(delta, layer.W.T) * (activations[i] > 0)

    gradients.reverse()
    return gradients

def update_weights(model, gradients, learning_rate):
    for i, layer in enumerate(model.layers):
        grad_w, grad_b = gradients[i]
        layer.W -= learning_rate * grad_w
        layer.b -= learning_rate * grad_b

def binary_ternary_converter(model): # ****待測試正確性****
   for layer in model.layers:
       layer.W_binary = float_to_fixed_point_TCB(layer.W)

def evaluate_model(model, test_images, test_labels):
    predictions = model(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_labels == test_labels)
    return accuracy

def fit(model, images, labels, test_images, test_labels, epochs, batch_size=128):
    accuracy_list = [] # 紀錄每次正確率的空list
    for epoch_counter in range(epochs): # 進行epochs次訓練
        start_time = time.time() # 一個epoch開始訓練時間點
        print(f"Epoch {epoch_counter}")
        batch_generator = BatchGenerator(images, labels) # 把dataset分批成num_batches等分
        for batch_counter in range(batch_generator.num_batches): # 進行num_batches次訓練
            images_batch, labels_batch = batch_generator.next() # 一次一小batche
            loss = one_training_step(model, images_batch, labels_batch) # 一小batche完整訓練一次
            if batch_counter % 5 == 0: # 每5個batches才print一次現在的loss多少
                print(f"loss at batch {batch_counter}: {loss:.2f}") # 所以這邊會print的次數等於訓練dataset的大小除與batch_size
        accuracy = evaluate_model(model, test_images, test_labels) # 紀錄這一個epoch的正確率
        accuracy_list.append(accuracy)
        end_time = time.time() # 一個epoch結束訓練時間點
        elapsed_time = end_time - start_time # 一個epoch訓練時間
        print(f"Epoch {epoch_counter} accuracy: {accuracy:.2f}, elapsed time: {elapsed_time:.2f}s") # 說明現在是第幾個epoch，正確率多少，花多久時間
    return accuracy_list

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255

sample_count = 600 # 記得batch_size也要改batch_counter幾次要print一次batch loss也要改

train_images = train_images[:sample_count, :]
train_labels = train_labels[:sample_count]
test_images = test_images[:sample_count, :]
test_labels = test_labels[:sample_count]

model = NaiveSequential([
    NaiveDense(input_size=28*28, output_size=256, activation=relu),
    NaiveDense(input_size=256, output_size=10, activation=softmax)
])

epochs = 3
batch_size = 32
accuracy_list = fit(model, train_images, train_labels, test_images, test_labels, epochs, batch_size=4)

# Plotting accuracy vs epochs
plt.plot(range(epochs), accuracy_list)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.show()

# Final accuracy
predictions = model(test_images)
predicted_labels = np.argmax(predictions, axis=1)
print(f"Final accuracy: {np.mean(predicted_labels == test_labels):.2f}")
