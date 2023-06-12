import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def leaky_relu(x, p):
    return np.maximum(x, p * x)

fig, ax = plt.subplots(2, 2, figsize=(10, 6), dpi=150)  # 创建画板

x = np.arange(-5.0, 5.0, 0.1)

sigmoidy = sigmoid(x)
ax[0][0].plot(x, sigmoidy, label='Sigmoid')
ax[0][0].legend()  # 添加图例

reluy = relu(x)
ax[0][1].plot(x, reluy, label='ReLU')
ax[0][1].legend()

tanhy = tanh(x)
ax[1][0].plot(x, tanhy, label='Tanh')
ax[1][0].legend()

LeakyReLUy = leaky_relu(x, 0.1)
ax[1][1].plot(x, LeakyReLUy, label='LeakyReLU')
ax[1][1].legend()

fig.show()
# fig.savefig('./images/Function_image.svg', format='svg', dpi=150)
