import logging
from abc import ABC, abstractmethod

import numpy as np

from ...utils.dataclasses import ModelTritonInterface

class Model(ABC):
    """Base class for all models."""

    def __init__(self,  model_name, model_input_name, model_output_name, model_datatype, triton_client):
        
        self.log = logging.getLogger(f"Model: {model_name}")
        
        self.triton_client = triton_client
        self.model_triton_interface = ModelTritonInterface(
            model_name=model_name,
            model_input_name=model_input_name,
            model_output_name=model_output_name,
            model_datatype=model_datatype,
        )
        
        self.log.info(f"Initialised model: {model_name}")
    
    @abstractmethod
    def _preprocessing(self, image):
        """Preprocess the image before inference."""
        
    @abstractmethod
    def _postprocessing(self, preds):
        """Postprocess the image after inference."""
            
    def __call__(self, image):
        image = self._preprocessing(image)
        preds = self._infer(image)
        results = self._postprocessing(preds)
        return results
       
    def _infer(self, image):
        return self.triton_client(image, self.model_triton_interface)
        