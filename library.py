# Standard library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, subplots, show
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomContrast,
    Flatten,
    Dropout,
    Dense
)
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

# Keras add-ons
from keras.optimizers import AdamW
from keras.metrics import (
    Precision,
    Recall,
    AUC,
    TopKCategoricalAccuracy,
    F1Score
)

# Scikit-learn
from sklearn.metrics import f1_score

# ----------------------------------------------------------- #
# ------------------ D.A PIPELINE VIZUALIZATION ------------- #
# ----------------------------------------------------------- #
def show_image(array):
    """
    Prints image encoded as a numpy array (uint8)
    """

    figure, axis = subplots(frameon=False)
    axis.imshow(array, aspect="equal")
    axis.set_axis_off()
    show()

# ----------------------------------------------------------- #
# ---------------- VISUALIZATION FUNCTION ------------------- #
# ----------------------------------------------------------- #
def vis_images(train):
    plt.figure(figsize=(10, 10))
    for images, labels in train.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(int(np.argmax(labels[i])))
            plt.axis("off")
    plt.show()
    
# ----------------------------------------------------------- #
# ---------------- MODEL SUMMARY FUNCTION ------------------- #
# ----------------------------------------------------------- #
def model_summary(model_class,input_shape=(224,224,3)):
    """
    Prints the summary of the model.
    """
    model = model_class()
    inputs = Input(shape=input_shape)
    _ = model.call(inputs)
    model.summary()
    return model

# ----------------------------------------------------------- #
# -------------- CHECK IMAGE SIZE FUNCTION ------------------ #
# ----------------------------------------------------------- #
def check_sizes(dataset):
    import numpy as np
    """
    Check the sizes of the images and labels in the dataset.
    """

    for images, labels in dataset.take(1):
        print("Image Batch Shape:", images.shape)
        print("Min Pixel Value:", np.min(images.numpy()))
        print("Max Pixel Value:", np.max(images.numpy()))
        print("Label Batch Shape:", labels.shape)


# ----------------------------------------------------------- #
# ------------------ DATA LOADING FUNCTION ------------------ #
# ----------------------------------------------------------- #
def data_loading(
    directory,
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(224, 224),
    color_mode="rgb",
    shuffle=True,
    interpolation="bilinear",
    seed=42
):
    return keras.utils.image_dataset_from_directory(
        directory=directory,
        labels=labels,
        label_mode=label_mode,
        batch_size=batch_size,
        image_size=image_size,
        color_mode=color_mode,
        shuffle=shuffle,
        interpolation=interpolation,
        seed=seed
    )

# ----------------------------------------------------------- #
# ------------------ FINE TUNE FUNCTION --------------------- #
# ----------------------------------------------------------- #

def fine_tune(model, 
              fine_tune_at=50, 
              lr=1e-5, 
              base_model=None, 
              base_layer_name="convnext_base", 
              checkpoint_path="best_finetuned.keras", 
              callbacks=None):
    if base_model is None:
        base_model = model.get_layer(base_layer_name)

    base_model.trainable = True

    if fine_tune_at > 0:
        for layer in base_model.layers[:-fine_tune_at]:
            layer.trainable = False

    model.compile(
        optimizer=AdamW(learning_rate=lr, weight_decay=3e-4),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy',
                TopKCategoricalAccuracy(k=5, name='top_5_accuracy'),
                F1Score(average='macro', name='f1_macro'),
                F1Score(average='weighted', name='f1_weighted')]
    )

    new_checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True)
    if callbacks:
        callbacks = [cb for cb in callbacks if not isinstance(cb, ModelCheckpoint)]
        callbacks.append(new_checkpoint)
    else:
        callbacks = [new_checkpoint]

    return model, callbacks

# ----------------------------------------------------------- #
# ---------------- PLOT ALL METRICS FUNCTION --------------- #
# ----------------------------------------------------------- #

def plot_all_metrics(history, title="Training Metrics"):
    metrics = [m for m in history.history.keys() if not m.startswith('val_')]

    epochs = range(1, len(history.history[metrics[0]]) + 1)
    num_metrics = len(metrics)

    cols = 2
    rows = (num_metrics + 1) // cols

    plt.figure(figsize=(cols * 6, rows * 4))
    plt.suptitle(title, fontsize=16)

    for i, metric in enumerate(metrics):
        plt.subplot(rows, cols, i + 1)
        plt.plot(epochs, history.history[metric], label=f'Train {metric}')

        val_metric = 'val_' + metric
        if val_metric in history.history:
            plt.plot(epochs, history.history[val_metric], label=f'Val {metric}')

        plt.title(metric.replace("_", " ").title())
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# ----------------------------------------------------------- #
# ----------------- METRICS TO EVALUATE TEST ----------------- #
# ----------------------------------------------------------- #
def get_classes(y_true_one_hot, y_pred_probs):
    y_true = np.argmax(y_true_one_hot, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    return y_true, y_pred

def compute_accuracy(y_true_one_hot, y_pred_probs):
    y_true, y_pred = get_classes(y_true_one_hot, y_pred_probs)
    return accuracy_score(y_true, y_pred)

def compute_f1_macro(y_true_one_hot, y_pred_probs):
    y_true, y_pred = get_classes(y_true_one_hot, y_pred_probs)
    return f1_score(y_true, y_pred, average='macro')

def compute_f1_weighted(y_true_one_hot, y_pred_probs):
    y_true, y_pred = get_classes(y_true_one_hot, y_pred_probs)
    return f1_score(y_true, y_pred, average='weighted')

def compute_top5_accuracy(y_true_one_hot, y_pred_probs):
    y_true = np.argmax(y_true_one_hot, axis=1)
    return top_k_accuracy_score(y_true, y_pred_probs, k=5)



# MODELS
# ----------------------------------------------------------- #
# ------------------ INCEPTIONRESNETV2 ---------------------- #
# ----------------------------------------------------------- #
class InResCustom(Model):
    def __init__(
        self,
        n_classes=202,
        use_augmentation=False,
        dropout_rate=None,
        regularizer=None,
        train_base=False,
        input_shape=(224, 224, 3),
        custom_aug_layers=None
    ):
        super().__init__()
        self.n_classes = n_classes

        ####### AUGMENTATION
        if use_augmentation:
            if custom_aug_layers is not None:
                self.augmentation = custom_aug_layers
            else:
                self.augmentation = Sequential([
                    RandomFlip("horizontal"),
                    RandomRotation(0.25), ## BEFORE 0.4
                    RandomZoom(0.15), ### BEFORE 0.2
                    RandomContrast(0.15, value_range=(0,1)), ## BEFORE 0.25
                    #RandomSharpness(0.25, value_range=(0,1)),
                    #RandomSaturation(0.2, value_range=(0,1))
                ])
        else:
            self.augmentation = None

        ######## PRETRAINED BASE
        self.pretrained_model = InceptionResNetV2(
            include_top=False,
            input_shape=input_shape
        )
        self.pretrained_model.trainable = train_base

        ######### FLATTEN AND DENSE
        self.flatten = Flatten()
        self.dropout = Dropout(dropout_rate) if dropout_rate else None
        self.dense = Dense(n_classes, activation="softmax", kernel_regularizer=regularizer)

    ######### CALL METHOD
    def call(self, inputs, training=False):
        x = self.augmentation(inputs, training=training) if self.augmentation else inputs
        x = self.pretrained_model(x)
        x = self.flatten(x)
        if self.dropout:
            x = self.dropout(x, training=training)
        return self.dense(x)
    