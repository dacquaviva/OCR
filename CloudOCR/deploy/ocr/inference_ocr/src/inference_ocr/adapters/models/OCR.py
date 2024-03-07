from .base import Model 
from .utils import strLabelConverter

import numpy as np
import cv2



class TextDetector(Model):
    def __init__(self, model_name, model_input_name, model_output_name, model_datatype, triton_client):
        
        super().__init__(model_name, model_input_name, model_output_name, model_datatype, triton_client)
        
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.d_size = (640,640)
                
                        
    def _preprocessing(self, image):
        """Preprocess the image before inference."""
        
        self.image = image

        h, w = image.shape[:2]

        # Resize while preserving aspect ratio
        if h != self.d_size[0] or w != self.d_size[1]:

            # Pad back up to input size where necessary
            padding_delta_h = self.d_size[0] - h
            padding_delta_w = self.d_size[1] - w

            if padding_delta_h < 0 or padding_delta_w < 0:            
                ratio = min(self.d_size[0] / h, self.d_size[0] / w)
                
                new_unpadded_h = round(h * ratio)
                new_unpadded_w = round(w * ratio)

                image = cv2.resize( image, (new_unpadded_w, new_unpadded_h), interpolation=cv2.INTER_AREA,)

                # Pad back up to input size where necessary
                padding_delta_h = self.d_size[0] - new_unpadded_h 
                padding_delta_w = self.d_size[1] - new_unpadded_w

            self.padding_delta_h = padding_delta_h
            self.padding_delta_w = padding_delta_w

            image = cv2.copyMakeBorder( image, 0, self.padding_delta_h, 0, self.padding_delta_w, cv2.BORDER_CONSTANT, value=0,)
            
        image_input = image[..., ::-1]  # bgr -> rgb
        image_input = (image_input / 255.0 - self.mean) / self.std
        image_input = image_input.astype(np.float32)
        image_input = image_input.transpose(2, 0, 1)
        image_input = np.ascontiguousarray(image_input)
        
        return image_input
    
    
    def _postprocessing(self, preds):
        """Postprocess the image after inference."""
        preds = preds[0][0]
        preds = np.where(preds>0.5, 255,0)
        preds = preds.astype(np.uint8)
        preds = preds[0:self.d_size[1]-self.padding_delta_h, 0:self.d_size[0]-self.padding_delta_w]
        preds = cv2.resize(preds,(self.image.shape[1], self.image.shape[0]))
        
        ratio_prime = 4.0
        min_area = 100
        cropped_images = []
        contours, hierarchy = cv2.findContours(preds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            enlarge = area * ratio_prime / cv2.arcLength(cnt, True)
            # Get the bounding box for the contour
            (cx, cy), (width, height), angle = cv2.minAreaRect(cnt)
            
            # Crop the contour horizontally using the angle obtained from cv2.minAreaRect
            
            cx, cy = int(cx), int(cy)
            width = int(width + enlarge)
            height = int(height + enlarge)
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            
            rotated = cv2.warpAffine(self.image, M, (self.image.shape[1], self.image.shape[0]))
            
            # Crop the image
            cropped = rotated[cy - height//2:cy + height//2, cx - width//2:cx + width//2]
            
            if cropped.shape[0] == 0 or cropped.shape[1] == 0:
                continue
            if cropped.shape[0] > cropped.shape[1]:
                cropped_images.append( { "crop1": np.rot90(cropped, k=1), "crop2": np.rot90(cropped, k=-1), "coordinates": [cx, cy, width, height, angle]})
            else:
                cropped_images.append( {"crop1": cropped, "crop2": np.rot90(np.rot90(cropped, k=1), k=1), "coordinates": [cx, cy, width, height, angle]})
            
        return cropped_images    
        
        
    
class TextRecognirez(Model):
    def __init__(self, model_name, model_input_name, model_output_name, model_datatype, triton_client):
        super().__init__(model_name, model_input_name, model_output_name, model_datatype, triton_client)
        self.alphabet = """0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~!"#$%&'()*+,-./ """
        self.converter = strLabelConverter(self.alphabet)   
        self.image_d_size = (320, 48)
        self.imagenet_mean = [0.5]
        self.imagenet_std = [0.5]
            
    def _preprocessing(self, image):
        """Preprocess the image before inference."""
        image_input = cv2.resize(image, self.image_d_size)  # resize
        image_input = image_input[..., ::-1]  # bgr -> rgb
        image_input = (image_input / 255.0 - self.imagenet_mean) / self.imagenet_std  # normalize
        image_input = image_input.astype(np.float32)  # float64 -> float32
        image_input = np.ascontiguousarray(image_input)  # contiguous array memory
        image_input = image_input.transpose(2, 0, 1)
        image_input = np.ascontiguousarray(image_input)
        
        return image_input
    
    def _postprocessing(self, preds):
        """Postprocess the image after inference."""
        # Model postprocessing, CTC decoding
        preds = np.array(preds)

        values = np.max(preds, axis=2)
        preds = np.argmax(preds, axis=2)

        confidence = np.prod(values, axis=1).item()
        # confidence = np.mean(values, axis=1).item()

        preds = np.transpose(preds, (1, 0)).ravel()
        preds_size = np.array([preds.size])

        # raw_pred = converter.decode(preds, preds_size, raw=True)
        ocr = self.converter.decode(preds, preds_size, raw=False)

        return ocr, confidence
