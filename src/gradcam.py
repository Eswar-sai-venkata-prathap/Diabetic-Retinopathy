import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out", pred_index=None):
    """
    Generates Grad-CAM heatmap for a specific input image.
    """
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
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

def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Superimposes heatmap onto original image.
    """
    heatmap = np.uint8(255 * heatmap)
    
    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, colormap)
    
    # Resize heatmap to image size
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))
    
    # Superimpose
    superimposed_img = jet * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    return superimposed_img

def save_gradcam(img_path, model, save_path, last_conv_layer_name="conv5_block3_out"):
    from src.preprocess import preprocess_image
    
    # Load and preprocess
    params = cv2.imread(img_path)
    img = preprocess_image(img_path)
    img_array = np.expand_dims(img, axis=0)
    
    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    # Overlay - we need the original (but resized) image for display
    # preprocess_image does resizing, so 'img' is 224x224x3 (normalized)
    # We need to denormalize for visualization (x255)
    img_display = np.uint8(img * 255)
    
    result = overlay_heatmap(img_display, heatmap)
    result.save(save_path)
    
    return result

if __name__ == "__main__":
    pass
