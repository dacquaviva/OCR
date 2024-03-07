import cv2
import numpy as np
import onnxruntime as ort
import os

CHARS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-']

def preprocess_image(img_path, img_size):
    img = cv2.imread(img_path)
        
    img_resized = cv2.resize(img, img_size)
    img_float = img_resized.astype('float32')
    img_float -= 127.5
    img_float *= 0.0078125
    img_transposed = np.transpose(img_float, (2, 0, 1))
    return img_transposed

def decode_label(preb):
    no_repeat_blank_label = []
    pre_c = preb[0]
    if pre_c != len(CHARS) - 1:
        no_repeat_blank_label.append(pre_c)
    for c in preb:
        if (pre_c == c) or (c == len(CHARS) - 1):
            if c == len(CHARS) - 1:
                pre_c = c
            continue
        no_repeat_blank_label.append(c)
        pre_c = c
    return ''.join([CHARS[i] for i in no_repeat_blank_label])

def main():
    # Set up parameters
    img_size = (240, 75)
    onnx_model_path = './OCRNet.onnx'  # Replace with your ONNX model path
    
    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_model_path)

    # Load and preprocess the image
    img_path = './test_image.png'  # Replace with your image path
    img = preprocess_image(img_path, img_size)
    
    # Prepare input for ONNX model
    input_name = ort_session.get_inputs()[0].name
    input_data = {input_name: img[np.newaxis, ...]}

    # Extract ground truth from filename
    gt = os.path.splitext(os.path.basename(img_path))[0]

    # Perform inference
    preds = ort_session.run(None, input_data)[0]

    # Decode the prediction
    pred = preds[0, :, :]  # Take the first (and only) prediction
    pred_label = np.argmax(pred, axis=0)
    pred_str = decode_label(pred_label)

    # Print the ground truth and prediction
    print(f"Ground Truth: {gt}")
    print(f"Prediction: {pred_str}")

if __name__ == "__main__":
    main()