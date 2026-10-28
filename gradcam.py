# Explainable AI Component: Grad-CAM (Gradient-weighted Class Activation Mapping)
# Visualizes the leaf regions that influenced the model's prediction

import numpy as np
import tensorflow as tf
import matplotlib
from PIL import Image

def generate_gradcam(img_array, model, last_conv_layer_name):
    # Create model mapping input to target conv layer activations and predictions
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradient of the top score with respect to feature map activations
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to keep only positive contributions and normalize
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val != 0:
        heatmap = heatmap / max_val
    
    confidence = float(predictions[0][pred_index])
    return heatmap.numpy(), int(pred_index), confidence

def overlay_gradcam(img_pil, heatmap, alpha=0.4):
    # Resize heatmap to match original image dimensions
    heatmap_pil = Image.fromarray(np.uint8(255 * heatmap)).resize(img_pil.size, Image.Resampling.BICUBIC)
    heatmap_arr = np.array(heatmap_pil) / 255.0

    # Apply Jet color palette
    colormap = matplotlib.colormaps["jet"]
    colored_heatmap = colormap(heatmap_arr)[:, :, :3]
    colored_heatmap = np.uint8(colored_heatmap * 255)

    # Superimpose heatmap onto original image
    original_arr = np.array(img_pil.convert("RGB"))
    blended = np.uint8(colored_heatmap * alpha + original_arr * (1 - alpha))
    return Image.fromarray(blended), heatmap_pil
