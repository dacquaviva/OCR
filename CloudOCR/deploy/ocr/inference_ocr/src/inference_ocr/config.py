import os
import logging

logging.basicConfig(level="INFO")


def get_text_detector_config():
    return {
        "model_name": os.environ.get("TEXT_DETECTOR_MODEL_NAME"),
        "model_input_name": os.environ.get("TEXT_DETECTOR_MODEL_INPUT_NAME"),
        "model_output_name": os.environ.get("TEXT_DETECTOR_MODEL_OUTPUT_NAME"),
        "model_datatype": os.environ.get("TEXT_DETECTOR_MODEL_DATATYPE"),
    }


def get_text_recognizer_config():
    return {
        "model_name": os.environ.get("TEXT_RECOGNIZER_MODEL_NAME"),
        "model_input_name": os.environ.get("TEXT_RECOGNIZER_MODEL_INPUT_NAME"),
        "model_output_name": os.environ.get("TEXT_RECOGNIZER_MODEL_OUTPUT_NAME"),
        "model_datatype": os.environ.get("TEXT_RECOGNIZER_MODEL_DATATYPE"),
    }


def get_triton_client_config_dict():
    return {
        "host": os.environ.get("TRITON_HOST"),
        "port": int(os.environ.get("TRITON_PORT")),
        "connection_attempts": int(os.environ.get("TRITON_ATTEMPTS")),
    }
