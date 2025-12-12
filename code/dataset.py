import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from model import build_cnn

def load_and_preprocess(val_ratio=0.1):
 (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
 print("Train shape:", x_train.shape, y_train.shape)
 print("Test shape:", x_test.shape, y_test.shape)


 # plt.imshow(x_train[3], cmap='gray')
 # plt.title(f'Label: {y_train[3]}')
 # plt.show()



 x_train= (x_train / 255.0)
 x_test= (x_test / 255.0)


 x_train = x_train.reshape(-1, 28, 28, 1)
 x_test = x_test.reshape(-1, 28, 28, 1)

 print("Train shape after reshape:", x_train.shape)
 print("Test shape after reshape:", x_test.shape)

 class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                'Ankle boot']

 for i in range(11):
     label_index = y_train[i]
     print(f"Sample {i}: Label {label_index} -> {class_names[label_index]}")
 model = build_cnn(input_shape=(28, 28, 1), num_classes=10)

 val_size = int(len(x_train) * val_ratio)
 x_val = x_train[:val_size]
 y_val = y_train[:val_size]

 x_train = x_train[val_size:]
 y_train = y_train[val_size:]

 return (x_train, y_train), (x_val, y_val),(x_test, y_test)

