# evaluate.py
import tensorflow as tf
from dataset import load_and_preprocess
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = "../saved_model/best_model.h5"

def evaluate_model():
    (_, _), (_, _), (x_test, y_test) = load_and_preprocess()

    model = load_model(MODEL_PATH)

    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"\nTest Results:")
    print(f"\nTest Accuracy : {test_accuracy:.4f}")
    print(f"\nTest Loss     : {test_loss:.4f}")

    predictions = model.predict(x_test)
    y_pred_classes = np.argmax(predictions, axis=1)

    print("\nClassification Report")
    print(classification_report(y_test, y_pred_classes))

    cm = confusion_matrix(y_test, y_pred_classes)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
evaluate_model()
