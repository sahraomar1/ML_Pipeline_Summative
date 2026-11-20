"""Build, train, test, save and retrain the model."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from src.preprocessing import CLASS_NAMES, IMAGE_SIZE, MODEL_PATH


def build_model(base_weights="imagenet"):
    """Make the model."""
    base = keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,), include_top=False, weights=base_weights
    )
    base.trainable = False  

    model = keras.Sequential([
        keras.Input(shape=IMAGE_SIZE + (3,)),
        layers.Rescaling(1 / 127.5, offset=-1),     
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),                        
        layers.Dense(
            len(CLASS_NAMES),
            activation="softmax",
            kernel_regularizer=regularizers.l2(0.001),  
        ),
    ])
    compile_model(model, learning_rate=0.001)
    return model


def compile_model(model, learning_rate):
    """Choose how the model learns."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def train_model(model, train_data, validation_data, epochs=15):
    """Train the model. Stop early if it stops getting better."""
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    return model.fit(train_data, validation_data=validation_data, epochs=epochs, callbacks=callbacks)


def evaluate_model(model, test_data):
    """Test the model. Give loss, accuracy, precision, recall and F1."""
    true_labels = np.concatenate([labels.numpy() for _, labels in test_data])
    probabilities = model.predict(test_data, verbose=0)
    predicted_labels = probabilities.argmax(axis=1)

    loss = keras.losses.sparse_categorical_crossentropy(true_labels, probabilities).numpy().mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="weighted", zero_division=0
    )
    all_labels = list(range(len(CLASS_NAMES)))
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels, labels=all_labels).tolist(),
        "class_names": CLASS_NAMES,
    }


def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)


def load_saved_model(path=MODEL_PATH):
    return keras.models.load_model(path)


def retrain_model(images, labels, epochs=5):
    """Train the saved model again on new photos."""
    model = load_saved_model()
    compile_model(model, learning_rate=0.0001)
    model.fit(images, labels, epochs=epochs, batch_size=16, shuffle=True)
    save_model(model)
    return model