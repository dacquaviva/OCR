from dataclasses import dataclass

@dataclass
class ModelTritonInterface:
    model_name: str 
    model_input_name: str
    model_output_name: str 
    model_datatype: str
