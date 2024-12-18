import numpy as np
import math
import time
import matplotlib.pyplot as plt
from keras.datasets import mnist

class NaiveDense:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation
        self.W = np.random.uniform(0, 1e-1, (input_size, output_size))
        self.b = np.zeros(output_size)

    def __call__(self, inputs):
        return self.activation(np.dot(inputs, self.W) + self.b)
    @property
    def weights(self):
        return [self.W, self.b]


class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights

class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
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

def one_training_step(model, images_batch, labels_batch, learning_rate=1e-3):
    predictions = model(images_batch)# 替換的地方
    per_sample_losses = np.mean(-np.log(predictions[np.arange(len(labels_batch)), labels_batch]))
    gradients = compute_gradients(model, images_batch, labels_batch)
    update_weights(model, gradients, learning_rate)
    return per_sample_losses

def compute_gradients(model, inputs, targets):
    activations = [inputs]
    for layer in model.layers:
        activations.append(layer(activations[-1]))

    predictions = activations[-1]
    one_hot_targets = np.eye(predictions.shape[1])[targets]
    delta = (predictions - one_hot_targets) / predictions.shape[0]

    gradients = []
    for i in reversed(range(len(model.layers))):
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
            if batch_counter % 100 == 0: # 每100個batches才print一次現在的loss多少
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


model = NaiveSequential([
    NaiveDense(input_size=28*28, output_size=512, activation=relu),
    NaiveDense(input_size=512, output_size=10, activation=softmax)
])


accuracy_list = fit(model, train_images, train_labels, test_images, test_labels, epochs=5, batch_size=128)

# Plotting accuracy vs epochs
plt.plot(range(5), accuracy_list)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.show()

# Final accuracy
predictions = model(test_images)
predicted_labels = np.argmax(predictions, axis=1)
print(f"Final accuracy: {np.mean(predicted_labels == test_labels):.2f}")
