import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from src.model import build_model

# Constants
BATCH_SIZE = 32
EPOCHS = 25 # Increased for convergence
IMG_SIZE = (224, 224)
test_size = 0.2
seed = 42
CACHE_DIR = os.path.join("data", "cache")
OUTPUT_DIR = os.path.join("output", "plots")
MODEL_DIR = "models"

class Epoch5Check(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 4: 
            val_acc = logs.get('val_accuracy')
            if val_acc < 0.45:
                print(f"\nEpoch 5 Check: Val Acc {val_acc:.4f} < 45%. Reducing LR.")
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, 1e-5)

def focal_loss(gamma=2., alpha=4.):
    gamma = float(gamma)
    alpha = float(alpha)
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1.e-9
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(loss)
    return focal_loss_fixed

def plot_history(history):
    plt.figure(figsize=(10,6))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_plot.png'))
    plt.close()
    
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_plot.png'))
    plt.close()

def train():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

    # 1. Load Cached Data
    print("Loading cached data...")
    X = np.load(os.path.join(CACHE_DIR, 'X.npy'))
    y = np.load(os.path.join(CACHE_DIR, 'y.npy'))
    
    # One-hot
    y_cat = tf.keras.utils.to_categorical(y, num_classes=5)
    
    # 2. Split (Stratified)
    X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=test_size, stratify=y, random_state=seed)
    
    # 3. Oversampling (Manual)
    # We need to recombine X_train and y_train indices to oversample
    print(f"Original Train Size: {len(X_train)}")
    
    y_train_indices = np.argmax(y_train, axis=1)
    unique_classes, counts = np.unique(y_train_indices, return_counts=True)
    max_count = np.max(counts)
    print(f"Max class count: {max_count}")
    
    X_train_resampled = []
    y_train_resampled = []
    
    for cls in unique_classes:
        # Get indices of this class
        cls_indices = np.where(y_train_indices == cls)[0]
        
        # Get samples
        X_cls = X_train[cls_indices]
        y_cls = y_train[cls_indices]
        
        # Resample
        if len(X_cls) < max_count:
            # Oversample
            X_cls_res, y_cls_res = resample(X_cls, y_cls, 
                                            replace=True, 
                                            n_samples=max_count, 
                                            random_state=seed)
        else:
            # Keep as is (or downsample if we wanted, but we oversample)
            X_cls_res, y_cls_res = X_cls, y_cls
        
        X_train_resampled.append(X_cls_res)
        y_train_resampled.append(y_cls_res)
        
    X_train_bal = np.concatenate(X_train_resampled)
    y_train_bal = np.concatenate(y_train_resampled)
    
    # Shuffle
    idx = np.arange(len(X_train_bal))
    np.random.shuffle(idx)
    X_train_bal = X_train_bal[idx]
    y_train_bal = y_train_bal[idx]
    
    print(f"Balanced Train Size: {len(X_train_bal)}")

    # 4. Heavy Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=45, 
        brightness_range=[0.7, 1.3],
        zoom_range=0.2,
        fill_mode='reflect'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = train_datagen.flow(X_train_bal, y_train_bal, batch_size=BATCH_SIZE)
    val_gen = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)
    
    # 5. Model
    model = build_model(learning_rate=1e-3)
    
    # Compile
    # We must use an optimizer instance to be able to set learning rate in callback
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=focal_loss(gamma=2.0), metrics=['accuracy'])
    
    # 6. Training
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True), # Increased patience slightly
        ModelCheckpoint(os.path.join(MODEL_DIR, 'best_model.h5'), monitor='val_accuracy', save_best_only=True),
        Epoch5Check()
    ]
    
    print(f"Starting Minority Class Boosting Training ({EPOCHS} Epochs)...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        # class_weight=None, # Removed class weights as we explicitly oversampled
        callbacks=callbacks
    )
    
    # 7. Evaluation
    print("Evaluating...")
    plot_history(history)
    
    print("Generating report on Validation Set...")
    X_val_norm = X_val / 255.0
    y_p = model.predict(X_val_norm)
    y_pred = np.argmax(y_p, axis=1)
    y_true = np.argmax(y_val, axis=1)
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()
    
    report = classification_report(y_true, y_pred)
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
        
    print(report)
    print("Training Complete.")

if __name__ == "__main__":
    train()
