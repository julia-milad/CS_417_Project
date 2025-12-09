import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("Train shape:", x_train.shape, y_train.shape)
print("Test shape:", x_test.shape, y_test.shape)




plt.imshow(x_train[1], cmap='gray')
plt.title(f'Label: {y_train[0]}')
plt.show()



x_train= (x_train / 255.0)
x_test= (x_test / 255.0)


x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("Train shape after reshape:", x_train.shape)
print("Test shape after reshape:", x_test.shape)
