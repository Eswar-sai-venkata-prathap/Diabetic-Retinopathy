import os
import cv2
import numpy as np
import tensorflow as tf
import glob
from src.model import build_model
from src.preprocess import load_ben_graham

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # EfficientNetB3 last conv layer is usually 'top_activation' or similar.
    # Let's try to find it dynamically or use a known one.
    # For EfficientNetB3, 'top_conv' or 'top_activation' before GlobalAveragePooling.
    # If using include_top=False, the last layer of base_model is the one.
    # We can search for the last logic layer that is 4D.
    
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def focal_loss(gamma=2., alpha=4.):
    # Just a dummy loader if model needs it (though we use CategoricalCrossentropy now)
    # But if model was saved with focal_loss object? 
    # Current train uses CategoricalCrossentropy(label_smoothing=0.1).
    # So we don't strictly need focal_loss unless we load old model.
    # But let's keep it just in case or remove if clean.
    pass

def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    # Retrieve heatmap and resize
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, colormap)
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))
    
    # Superimpose
    superimposed_img = jet * alpha + img
    return superimposed_img

def save_gradcam(img_path, model, save_path):
    # Image is already in data/processed, so it is Ben Graham processed?
    # task.md says: Refine src/preprocess.py (Ben Graham with Radius Scaling).
    # And we ran preprocess.
    # So valid/test images in data/processed are BG processed.
    
    img = cv2.imread(img_path)
    if img is None: return
    
    # Model expects 224x224, RGB, normalized [0,1]
    img = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_rgb / 255.0, axis=0)
    
    # MobileNetV2 last conv layer: 'Conv_1'
    # Even with SpatialDropout added after base_model.output, base_model structure inside remains matching.
    target_layer = "Conv_1" 

    try:
        heatmap = make_gradcam_heatmap(img_array, model, target_layer)
        # Resize heatmap to match image size (224x224) done in overlay_heatmap?
        # overlay_heatmap resizes heatmap to img.shape
        superimposed = overlay_heatmap(img, heatmap) # img is BGR, suitable for overlay/imwrite
        cv2.imwrite(save_path, superimposed)
    except Exception as e:
        print(f"Grad-CAM failed for {target_layer}: {e}")
        # Try finding last 4D layer
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:
                print(f"Fallback to {layer.name}")
                heatmap = make_gradcam_heatmap(img_array, model, layer.name)
                superimposed = overlay_heatmap(img, heatmap)
                cv2.imwrite(save_path, superimposed)
                break

def generate_explanations():
    model_path = os.path.join("models", "best_model.h5")
    if not os.path.exists(model_path):
        print("Model not found. Cannot generate explanations.")
        return

    # Load with custom_objects if needed
    # We use standard loss now, so no custom objects?
    # If we saved QWKMetrics? Callbacks aren't saved in model file usually.
    # Load with compile=False to avoid custom loss deserialization issues
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Find some images
    images = glob.glob(os.path.join("data", "processed", "**", "*.jpg"), recursive=True)
    # Only take a few unique classes if possible
    # Just take first 3 for valid
    if not images:
        images = glob.glob(os.path.join("data", "raw", "**", "*.jpg"), recursive=True)
        
    print(f"Generating explanations for {min(3, len(images))} images...")
    
    output_dir = os.path.join("output", "plots")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i, img_path in enumerate(images[:3]):
        save_path = os.path.join(output_dir, f"gradcam_{i}.png")
        save_gradcam(img_path, model, save_path)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    generate_explanations()
