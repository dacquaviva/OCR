import tensorrt as trt
import os

def build_engine(onnx_file_path, engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config.set_flag(trt.BuilderFlag.TF32)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print(f'ERROR: Failed to parse the ONNX file: {onnx_file_path}')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print(f'ONNX file parsed successfully: {onnx_file_path}')

    engine = builder.build_engine(network, config)
    
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f'TensorRT engine file saved successfully: {engine_file_path}')
    return engine

def convert_models():
    onnx_files = [f for f in os.listdir('/workspace') if f.endswith('.onnx')]
    
    for onnx_file in onnx_files:
        onnx_file_path = os.path.join('/workspace', onnx_file)
        engine_file_path = os.path.join('/workspace', onnx_file.replace('.onnx', '.trt'))
        
        print(f"Converting {onnx_file} to TensorRT...")
        build_engine(onnx_file_path, engine_file_path)

if __name__ == '__main__':
    convert_models()