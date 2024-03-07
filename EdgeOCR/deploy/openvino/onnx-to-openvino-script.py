import os
import sys
from openvino.runtime import Core
from openvino.tools import mo
from openvino.runtime import serialize

def convert_models():
    onnx_files = [f for f in os.listdir('/workspace') if f.endswith('.onnx')]
    
    if not onnx_files:
        print("No ONNX files found in the workspace directory.")
        return

    for onnx_file in onnx_files:
        onnx_file_path = os.path.join('/workspace', onnx_file)
        output_dir = os.path.join('/workspace', onnx_file[:-5] + '_openvino')
        
        print(f"Converting {onnx_file} to OpenVINO IR...")
        
        try:
            # Convert ONNX to OpenVINO IR
            model = mo.convert_model(onnx_file_path, output_dir=output_dir)
            
            # Serialize the model
            xml_path = os.path.join(output_dir, onnx_file[:-5] + '.xml')
            bin_path = os.path.join(output_dir, onnx_file[:-5] + '.bin')
            serialize(model, xml_path, bin_path)
            
            print(f"OpenVINO IR files saved in: {output_dir}")
            
            # Verify that files were created
            if os.path.exists(xml_path) and os.path.exists(bin_path):
                print(f"Successfully created {xml_path} and {bin_path}")
            else:
                print(f"Error: Expected files were not created in {output_dir}")
            
            # Optionally, you can compile the model for specific hardware
            # Uncomment the following lines if you want to compile for a specific device
            # ie = Core()
            # compiled_model = ie.compile_model(model, device_name="CPU")
            # print(f"Model compiled for CPU")
            
            print(f"Conversion completed for {onnx_file}")
        
        except Exception as e:
            print(f"Error converting {onnx_file}: {str(e)}")
            
    print("Conversion process finished. Please check the output above for any errors.")

if __name__ == '__main__':
    convert_models()
    
    # Print the contents of the workspace after conversion
    print("\nContents of /workspace after conversion:")
    for item in os.listdir('/workspace'):
        print(item)
        if os.path.isdir(os.path.join('/workspace', item)):
            subdir = os.path.join('/workspace', item)
            print(f"  Contents of {item}:")
            for subitem in os.listdir(subdir):
                print(f"    {subitem}")