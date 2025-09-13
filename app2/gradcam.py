import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image as keras_image
import os

def generate_gradcam_overlay(image_path, model, last_conv_layer_name="mixed10", class_names=None):
    img = keras_image.load_img(image_path, target_size=(299, 299))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    preds = model.predict(x)
    pred_class = np.argmax(preds[0])
    pred_label = class_names[pred_class] if class_names else str(pred_class)
    confidence = preds[0][pred_class]

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        loss = predictions[:, pred_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    orig = cv2.imread(image_path)
    orig = cv2.resize(orig, (299, 299))
    heatmap_resized = cv2.resize(heatmap, (299, 299))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(orig, 0.6, heatmap_colored, 0.4, 0)

    # Bounding boxes for cancerous region
    binary_map = np.uint8(heatmap_resized > 0.6)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(overlayed, (x, y), (x + w, y + h), (0, 255, 0), 2)

    gradcam_dir = os.path.join("media", "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)
    overlay_path = os.path.join(gradcam_dir, os.path.basename(image_path))
    cv2.imwrite(overlay_path, overlayed)

    return overlay_path, pred_label, confidence
