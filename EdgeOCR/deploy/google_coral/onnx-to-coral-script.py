import os
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import subprocess

def convert_models():
    onnx_files = [f for f in os.listdir('/workspace') if f.endswith('.onnx')]
    
    if not onnx_files:
        print("No ONNX files found in the workspace directory.")
        return

    for onnx_file in onnx_files:
        onnx_file_path = os.path.join('/workspace', onnx_file)
        base_name = onnx_file[:-5]
        
        print(f"Converting {onnx_file} to TensorFlow Lite format...")
        
        try:
            # Load ONNX model
            onnx_model = onnx.load(onnx_file_path)
            
            # Convert ONNX model to TensorFlow
            tf_rep = prepare(onnx_model)
            
            # Save TensorFlow model
            tf_model_path = f'/workspace/{base_name}_tf'
            tf_rep.export_graph(tf_model_path)
            
            # Convert TensorFlow model to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            converter.representative_dataset = representative_dataset_gen
            tflite_model = converter.convert()
            
            # Save TFLite model
            tflite_path = f'/workspace/{base_name}.tflite'
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"TensorFlow Lite model saved as: {tflite_path}")
            
            # Compile for Edge TPU
            edge_tpu_path = f'/workspace/{base_name}_edgetpu.tflite'
            subprocess.run(['edgetpu_compiler', '-s', tflite_path], check=True)
            
            print(f"Edge TPU model compiled and saved as: {edge_tpu_path}")
            
        except Exception as e:
            print(f"Error converting {onnx_file}: {str(e)}")
    
    print("Conversion process finished. Please check the output above for any errors.")

def representative_dataset_gen():
    # This is a placeholder function. You should replace this with actual
    # representative data from your dataset.
    for _ in range(100):
        yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]

if __name__ == '__main__':
    convert_models()
    
    # Print the contents of the workspace after conversion
    print("\nContents of /workspace after conversion:")
    for item in os.listdir('/workspace'):
        print(item)
