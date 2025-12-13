import tensorflow as tf
from dataset import load_and_preprocess
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CONF_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
SAMPLE_PRED_PATH = RESULTS_DIR / "sample_prediction.png"

MODEL_PATH = "../saved_model/best_model.h5"

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag','Ankle boot']
img_num = 7

def show_prediction(i, predictions, labels, images):
    plt.figure()
    plt.imshow(images[i], cmap="gray")
    plt.axis("off")

    pred_label = np.argmax(predictions[i])
    true_label = labels[i]

    color = "green" if pred_label == true_label else "red"
    plt.title(
        f"Predicted: {class_names[pred_label]} "
        f"({100*np.max(predictions[i]):.1f}%)\n"
        f"Actual: {class_names[true_label]}",
        color=color
    )
    plt.savefig(SAMPLE_PRED_PATH)
    plt.show()

def evaluate_model():
    (_, _), (_, _), (x_test, y_test) = load_and_preprocess()

    model = load_model(MODEL_PATH)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Results:")
    print(f"\nTest Accuracy : {test_accuracy:.4f}")
    print(f"\nTest Loss     : {test_loss:.4f}")

    predictions = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(predictions, axis=1)

    print("\nClassification Report")
    print(classification_report(y_test, y_pred_classes))

    cm = confusion_matrix(y_test, y_pred_classes)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(CONF_MATRIX_PATH)
    plt.show()

    show_prediction(img_num, predictions, y_test, x_test)

evaluate_model()

