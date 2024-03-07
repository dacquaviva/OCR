from . import config

from .adapters.triton_client import TritonClient
from .adapters.models.OCR import TextDetector, TextRecognirez

from .service_layer.message_handler import MessageHandler


def bootstrap():

    # Triton
    triton_client_config = config.get_triton_client_config_dict()
    triton_client = TritonClient.from_dict(triton_client_config)
    
    #Text Detector
    text_detector_config = config.get_text_detector_config()
    text_detector = TextDetector(triton_client=triton_client, **text_detector_config)
    
    #Text Recognizer
    text_recognizer_config = config.get_text_recognizer_config()
    text_recognizer = TextRecognirez(triton_client=triton_client, **text_recognizer_config)
    # MessageHandler
    message_handler = MessageHandler(
             text_detector=text_detector, text_recognizer=text_recognizer)

    return {
        "message_handler": message_handler,
    }


