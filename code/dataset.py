import numpy as np
from tensorflow.keras.datasets import fashion_mnist

def load_and_preprocess(val_ratio=0.1):
 (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
 print("Train shape:", x_train.shape, y_train.shape)
 print("Test shape:", x_test.shape, y_test.shape)

 x_train= (x_train / 255.0)
 x_test= (x_test / 255.0)


 x_train = x_train.reshape(-1, 28, 28, 1)
 x_test = x_test.reshape(-1, 28, 28, 1)

 print("Train shape after reshape:", x_train.shape)
 print("Test shape after reshape:", x_test.shape)


 indices = np.arange(len(x_train))
 np.random.shuffle(indices)
 x_train = x_train[indices]
 y_train = y_train[indices]

 val_size = int(len(x_train) * val_ratio)
 x_val = x_train[:val_size]
 y_val = y_train[:val_size]

 x_train = x_train[val_size:]
 y_train = y_train[val_size:]

 return (x_train, y_train), (x_val, y_val),(x_test, y_test)

