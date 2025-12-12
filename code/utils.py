import matplotlib.pyplot as plt

def plot_history(history, out_path_acc=None, out_path_loss=None):

    plt.figure()
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(out_path_acc)
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(out_path_loss)
    plt.show()