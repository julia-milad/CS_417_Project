from model import build_cnn
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from dataset import load_and_preprocess
from utils import plot_history

OUTPUT_DIR = "../results"
def main():
    (x_train, y_train), (x_val, y_val), _ = load_and_preprocess()
    model = build_cnn(input_shape=(28, 28, 1), num_classes=10)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False
    )
    datagen.fit(x_train)

    callbacks = [
        EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
        ModelCheckpoint("../saved_model/best_model.h5", save_best_only=True),
        ReduceLROnPlateau(patience=3, factor=0.3, monitor="val_loss"),
    ]

    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        epochs=30,
        validation_data=(x_val, y_val),
        callbacks=callbacks
    )

    plot_history(history,
                 out_path_acc=os.path.join(OUTPUT_DIR, "accuracy_curve.png"),
                 out_path_loss=os.path.join(OUTPUT_DIR, "loss_curve.png")
                 )

    model.summary()
    print("Model saved successfully!")

main()