import numpy as np
import tensorflow as tf
import cv2


# =========================
# Automatically Find Last Conv Layer
# =========================
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            return layer.name
    raise ValueError("No 4D convolutional layer found in model.")


# =========================
# Generate Grad-CAM Heatmap
# =========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generates Grad-CAM heatmap dynamically.
    """

    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Safe normalization
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)

    if max_val == 0:
        return np.zeros_like(heatmap.numpy())

    heatmap /= max_val

    return heatmap.numpy()


# =========================
# Overlay Heatmap
# =========================
def overlay_heatmap(img, heatmap, alpha=0.4):
    """
    Superimposes heatmap onto original image.
    img: uint8 RGB image
    heatmap: normalized 2D array
    """

    heatmap = np.uint8(255 * heatmap)

    # Apply colormap
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert BGR to RGB for correct blending
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    colored = cv2.resize(colored, (img.shape[1], img.shape[0]))

    overlay = cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)

    return overlay


# =========================
# Save Grad-CAM
# =========================
def save_gradcam(img_path, model, save_path):
    from src.preprocess import preprocess_image

    # Preprocess
    img = preprocess_image(img_path)
    img_array = np.expand_dims(img, axis=0)

    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array, model)

    # Convert for display
    img_display = np.uint8(img * 255)

    result = overlay_heatmap(img_display, heatmap)

    cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    return result


if __name__ == "__main__":
    pass
