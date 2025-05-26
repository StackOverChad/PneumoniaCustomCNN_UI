import kagglehub
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to dataset files:", path)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns
from IPython.display import clear_output
import glob
import datetime

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

import kagglehub
dataset_path_object = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print("Path to dataset files:", dataset_path_object)

base_data_dir = os.path.join(dataset_path_object, 'chest_xray')
train_dir = os.path.join(base_data_dir, 'train')
val_dir = os.path.join(base_data_dir, 'val')
test_dir = os.path.join(base_data_dir, 'test')

if not os.path.exists(train_dir):
    train_dir = os.path.join(dataset_path_object, 'train')
    val_dir = os.path.join(dataset_path_object, 'val')
    test_dir = os.path.join(dataset_path_object, 'test')
    if not os.path.exists(train_dir):
        raise FileNotFoundError("Could not locate dataset directories. Structure might have changed.")

print(f"Using Training directory: {train_dir}")
print(f"Using Validation directory: {val_dir}")
print(f"Using Testing directory: {test_dir}")

IMG_WIDTH = 150
IMG_HEIGHT = 150
IMAGE_SIZE = (IMG_WIDTH, IMG_HEIGHT)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',
    image_size=IMAGE_SIZE,
    interpolation='bilinear',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='binary',
    image_size=IMAGE_SIZE,
    interpolation='bilinear',
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=42
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='binary',
    image_size=IMAGE_SIZE,
    interpolation='bilinear',
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names
print("Class names:", class_names)

data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1),
], name="data_augmentation")

rescale = layers.Rescaling(1./255)

def preprocess_data(image, label):
    image = rescale(image)
    return image, label

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset_for_eval = test_dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset_for_eval = test_dataset_for_eval.prefetch(buffer_size=tf.data.AUTOTUNE)

def build_custom_cnn_model(input_shape):
    model = Sequential([
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        layers.Dense(1, activation='sigmoid')
    ], name="custom_cnn_pneumonia")
    return model

model_input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)
model = build_custom_cnn_model(model_input_shape)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.AUC(name='auc')])

normal_count = len(glob.glob(os.path.join(train_dir, 'NORMAL', '*.[jp][pn]g')))
pneumonia_count = len(glob.glob(os.path.join(train_dir, 'PNEUMONIA', '*.[jp][pn]g')))
total_train_samples = normal_count + pneumonia_count

print(f"Normal images in training: {normal_count}")
print(f"Pneumonia images in training: {pneumonia_count}")

if total_train_samples > 0 and normal_count > 0 and pneumonia_count > 0:
    weight_for_0 = (1 / normal_count) * (total_train_samples / 2.0)
    weight_for_1 = (1 / pneumonia_count) * (total_train_samples / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"Calculated Class weights: {class_weight} (0: {class_names[0]}, 1: {class_names[1]})")
else:
    class_weight = None
    print("Could not calculate class weights. Check class counts or dataset structure.")

log_dir = "logs/fit_custom_cnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_filepath = '/kaggle/working/best_custom_cnn_model.keras'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_auc',
    mode='max',
    save_best_only=True)

early_stopping_callback = EarlyStopping(
    monitor='val_auc',
    patience=10,
    mode='max',
    restore_best_weights=True)

reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1)

print("\n--- Training Custom CNN Model ---")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[model_checkpoint_callback, early_stopping_callback, reduce_lr_callback, tensorboard_callback],
    class_weight=class_weight
)

best_model = None
if os.path.exists(checkpoint_filepath):
    print(f"Loading best model from: {checkpoint_filepath}")
    best_model = tf.keras.models.load_model(checkpoint_filepath)
else:
    print("No best model checkpoint found. Using model from end of training.")
    best_model = model

if best_model is None:
    print("ERROR: Model is None. Cannot proceed to evaluation.")
else:
    print("\n--- Evaluating on Test Data (Custom CNN) ---")
    eval_results = best_model.evaluate(test_dataset_for_eval)
    print(f"Test Loss: {eval_results[0]:.4f}")
    for i, metric_name in enumerate(best_model.metrics_names[1:]):
        print(f"Test {metric_name}: {eval_results[i+1]:.4f}")

    y_pred_probs_test = best_model.predict(test_dataset_for_eval).ravel()
    y_pred_classes_test = (y_pred_probs_test > 0.5).astype(int)

    y_true_test = []
    for _, labels_batch in test_dataset_for_eval:
        y_true_test.extend(labels_batch.numpy().flatten())
    y_true_test = np.array(y_true_test)

    print("\n--- Detailed Classification Report (Test Data) ---")
    print(classification_report(y_true_test, y_pred_classes_test, target_names=class_names, digits=4, zero_division=0))

    print("\n--- Individual Metrics (Test Data) ---")
    print(f"Sklearn Accuracy: {accuracy_score(y_true_test, y_pred_classes_test):.4f}")
    if 'PNEUMONIA' in class_names:
        pneumonia_idx = class_names.index('PNEUMONIA')
        print(f"Sklearn Precision (Pneumonia): {precision_score(y_true_test, y_pred_classes_test, pos_label=pneumonia_idx, zero_division=0):.4f}")
        print(f"Sklearn Recall (Pneumonia): {recall_score(y_true_test, y_pred_classes_test, pos_label=pneumonia_idx, zero_division=0):.4f}")
        print(f"Sklearn F1-Score (Pneumonia): {f1_score(y_true_test, y_pred_classes_test, pos_label=pneumonia_idx, zero_division=0):.4f}")
    else:
        print(f"Sklearn Precision (Positive Class): {precision_score(y_true_test, y_pred_classes_test, zero_division=0):.4f}")
        print(f"Sklearn Recall (Positive Class): {recall_score(y_true_test, y_pred_classes_test, zero_division=0):.4f}")
        print(f"Sklearn F1-Score (Positive Class): {f1_score(y_true_test, y_pred_classes_test, zero_division=0):.4f}")

    try:
        print(f"Sklearn ROC AUC: {roc_auc_score(y_true_test, y_pred_probs_test):.4f}")
    except ValueError as e_auc:
        print(f"Could not calculate ROC AUC: {e_auc}")

    cm = confusion_matrix(y_true_test, y_pred_classes_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Test Data - Custom CNN)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    def plot_training_history(history_obj, model_name="Custom CNN"):
        metrics_to_plot = ['accuracy', 'loss', 'precision', 'recall', 'auc']
        plt.figure(figsize=(18, 12))

        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(2, 3, i + 1)

            if metric in history_obj.history:
                plt.plot(history_obj.history[metric], label=f'Training {metric.capitalize()}')
            if f'val_{metric}' in history_obj.history:
                plt.plot(history_obj.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}', linestyle='--')

            plt.title(f'{metric.capitalize()} vs. Epochs')
            plt.xlabel('Epochs')
            plt.ylabel(metric.capitalize())
            plt.legend(loc='best')
            plt.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.suptitle(f'{model_name} Training History', fontsize=16)
        plt.show()

    if history:
        plot_training_history(history, model_name="Custom CNN")
    else:
        print("Training history not available to plot.")

print("\n--- Custom CNN Model Training and Evaluation Complete ---")

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

if 'best_model' in locals() and best_model is not None:
    drive_base_folder = '/content/drive/My Drive/'
    project_model_folder = 'ColabNotebooks/Models/PneumoniaCustomCNN'

    drive_save_path_dir = os.path.join(drive_base_folder, project_model_folder)

    model_filename_on_drive = 'custom_cnn_pneumonia_chest_xray.keras'
    full_drive_save_path = os.path.join(drive_save_path_dir, model_filename_on_drive)

    try:
        os.makedirs(drive_save_path_dir, exist_ok=True)
        print(f"Ensured directory exists: {drive_save_path_dir}")

        print(f"Attempting to save model to Google Drive at: {full_drive_save_path}")
        best_model.save(full_drive_save_path)
        print(f"Model successfully saved to: {full_drive_save_path}")

        print("\nFiles in the Drive save directory:")
        for item in os.listdir(drive_save_path_dir):
            print(os.path.join(drive_save_path_dir, item))

    except Exception as e:
        print(f"An error occurred during model saving or directory creation: {e}")
        print("Please ensure Google Drive is mounted correctly and you have write permissions.")

else:
    print("The 'best_model' object was not found or is None. Cannot save to Google Drive.")
    print("Make sure your model training and selection process has completed successfully.")

custom_model_save_name = "custom_cnn_pneumonia_model.keras"
best_model.save(custom_model_save_name)

from google.colab import files
files.download(custom_model_save_name)

import json
class_names_pneumonia = ['NORMAL', 'PNEUMONIA']
with open("class_names_pneumonia.json", "w") as f:
    json.dump(class_names_pneumonia, f)
files.download("class_names_pneumonia.json")