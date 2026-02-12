import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.model import build_model

# =========================
# Constants
# =========================
BATCH_SIZE = 32
EPOCHS = 40
TEST_SIZE = 0.2
SEED = 42

CACHE_DIR = os.path.join("data", "cache")
OUTPUT_DIR = os.path.join("output", "plots")
MODEL_DIR = "models"


# =========================
# Plot Training History
# =========================
def plot_history(history):
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_plot.png'))
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.legend(['Train', 'Validation'])
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_plot.png'))
    plt.close()


# =========================
# Training Pipeline
# =========================
def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading cached data...")
    X = np.load(os.path.join(CACHE_DIR, 'X.npy'))
    y = np.load(os.path.join(CACHE_DIR, 'y.npy'))

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=SEED
    )

    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")

    # Class weights
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights_array))
    print("Class Weights:", class_weights)

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=25,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode='reflect'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_gen = val_datagen.flow(
        X_val, y_val,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Build model
    model = build_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    callbacks = [
        EarlyStopping(monitor='val_auc', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-6),
        ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_model.h5'),
            monitor='val_auc',
            save_best_only=True
        )
    ]

    print(f"Starting Binary DR Training ({EPOCHS} epochs)...")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    plot_history(history)

    # =========================
    # Evaluation
    # =========================
    print("Evaluating on Validation Set...")

    X_val_norm = X_val / 255.0
    y_probs = model.predict(X_val_norm).ravel()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_val, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.title(f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'))
    plt.close()

    # Precision-Recall Curve
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_val, y_probs)
    ap_score = average_precision_score(y_val, y_probs)

    plt.figure()
    plt.plot(recall_vals, precision_vals)
    plt.title(f"Precision-Recall Curve (AP = {ap_score:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(os.path.join(OUTPUT_DIR, 'pr_curve.png'))
    plt.close()

    # Optimal threshold (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    with open(os.path.join(OUTPUT_DIR, "optimal_threshold.txt"), "w") as f:
        f.write(f"Optimal Threshold: {optimal_threshold:.4f}")

    # Sensitivity & Specificity vs Threshold
    sensitivity = tpr
    specificity = 1 - fpr

    plt.figure()
    plt.plot(thresholds, sensitivity)
    plt.plot(thresholds, specificity)
    plt.title("Sensitivity & Specificity vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend(["Sensitivity", "Specificity"])
    plt.savefig(os.path.join(OUTPUT_DIR, 'threshold_curve.png'))
    plt.close()

    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_val, y_probs, n_bins=10)

    plt.figure()
    plt.plot(prob_pred, prob_true)
    plt.plot([0, 1], [0, 1])
    plt.title("Calibration Curve")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.savefig(os.path.join(OUTPUT_DIR, 'calibration_curve.png'))
    plt.close()

    # Confusion Matrix (using optimal threshold)
    y_pred = (y_probs >= optimal_threshold).astype(int)

    cm = confusion_matrix(y_val, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix (Raw)")
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_raw.png'))
    plt.close()

    cm_norm = confusion_matrix(y_val, y_pred, normalize='true')

    plt.figure()
    sns.heatmap(cm_norm, annot=True)
    plt.title("Confusion Matrix (Normalized)")
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_normalized.png'))
    plt.close()

    report = classification_report(y_val, y_pred)

    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)

    print(report)
    print(f"AUC Score: {roc_auc:.4f}")
    print("All evaluation plots saved.")


if __name__ == "__main__":
    train()
