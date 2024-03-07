import cv2
import numpy as np
import onnxruntime as ort
import os

class YOLOv8TextDetector:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def prepare_input(self, image, save_path=None):
        self.img_height, self.img_width = image.shape[:2]

        # Letterbox
        input_img, dw, dh = self.resize_and_pad(image, (self.input_height, self.input_width))

        # Save padded image if a save path is provided
        if save_path:
            cv2.imwrite(save_path, input_img)

        # Normalize and transpose
        input_img = input_img.astype(np.float32) / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :]

        return input_tensor, (dw, dh)

    def resize_and_pad(self, image, new_shape):
        old_size = image.shape[:2] 
        ratio = float(new_shape[-1]/max(old_size))
        new_size = tuple([int(x*ratio) for x in old_size])
        image = cv2.resize(image, (new_size[1], new_size[0]))
        
        delta_w = new_shape[1] - new_size[1]
        delta_h = new_shape[0] - new_size[0]
        
        color = [100, 100, 100]
        new_im = cv2.copyMakeBorder(image, 0, delta_h, 0, delta_w, cv2.BORDER_CONSTANT, value=color)
        
        return new_im, delta_w, delta_h

    def process_output(self, output, pad):
        predictions = np.squeeze(output[0]).T
        
        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        
        if len(scores) == 0:
            return [], []

        # Get the class with the highest score
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        
        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)
        
        # Rescale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes, pad)
        
        # Apply non-maximum suppression to suppress weak, overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold).flatten()

        return [boxes[i] for i in indices], [scores[i] for i in indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]
        
        # Convert boxes to xyxy format
        boxes = self.xywh2xyxy(boxes)
        
        return boxes

    def rescale_boxes(self, boxes, pad):
        # Rescale boxes to original image dimensions
        boxes = np.array(boxes)
        dw, dh = pad
        input_shape = (self.input_height - dh, self.input_width - dw)
        
        boxes[:, [0, 2]] *= self.img_width / input_shape[1]
        boxes[:, [1, 3]] *= self.img_height / input_shape[0]
        
        return boxes.astype(np.int32).tolist()

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def detect(self, image, save_padded_path=None):
        input_tensor, pad = self.prepare_input(image, save_padded_path)
        
        # Perform inference on the image
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # Process output data
        self.boxes, self.scores = self.process_output(outputs, pad)
        
        return self.boxes, self.scores

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return self.draw_boxes(image, self.boxes, self.scores, draw_scores, mask_alpha)

    def draw_boxes(self, image, boxes, scores, draw_scores=True, mask_alpha=0.4):
        mask = np.zeros(image.shape, dtype=np.uint8)
        
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Draw filled bounding box on mask
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), -1)
        
        # Apply mask to image
        masked_image = cv2.addWeighted(image, 1, mask, mask_alpha, 0)
        
        # Draw bounding boxes on the image
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            
            # Draw bounding box on image
            cv2.rectangle(masked_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            if draw_scores:
                # Draw label and score on image
                label = f'Text: {scores[i]:.2f}'
                cv2.putText(masked_image, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return masked_image
        
# Example usage
if __name__ == "__main__":
    # Initialize YOLOv8 object detector
    yolov8_detector = YOLOv8TextDetector(model_path="./weights/best.onnx",
                                         conf_threshold=0.5,
                                         iou_threshold=0.5)
    
    # Read image
    image = cv2.imread("./test_image.png")
    print(image.shape)

    # Perform inference
    boxes, scores = yolov8_detector.detect(image)

    # Draw detections
    img_with_detections = yolov8_detector.draw_detections(image)

    # Display the image
    cv2.imshow("Text Detection", img_with_detections)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
