import numpy as np
import onnxruntime as ort

model_path = 'model_w_15_quant.onnx'

def load_input_from_file(filename):
    """Load input data from example.txt file"""
    with open(filename, 'r') as f:
        content = f.read().strip()
    
    # Split the content into separate grids
    lines = content.split('\n')
    lines = [line.strip() for line in lines if line.strip()]  # Remove empty lines
    
    grids = []
    current_grid = []
    
    for line in lines:
        if line:
            # Parse the line into integers
            row = [int(x) for x in line.split()]
            current_grid.append(row)
            
            # If we have 15 rows, we have a complete grid
            if len(current_grid) == 15:
                grids.append(np.array(current_grid))
                current_grid = []
    
    # For 3-channel input, we need to stack the 3 grids as channels
    if len(grids) >= 3:
        # Stack first 3 grids as channels: shape (3, 15, 15)
        input_tensor = np.stack(grids[:3], axis=0).astype(np.float32)
        return input_tensor
    else:
        print(f"Warning: Expected 3 grids but found {len(grids)}")
        return grids

def run_inference():
    """Load ONNX model and run inference on example data"""
    print("Loading ONNX model...")
    
    # Create inference session
    session = ort.InferenceSession(model_path)
    
    # Get model input/output info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()
    
    print(f"Model input name: {input_info.name}")
    print(f"Model input shape: {input_info.shape}")
    print(f"Model input type: {input_info.type}")
    
    print("Model outputs:")
    for i, output in enumerate(output_info):
        print(f"  Output {i}: {output.name}, shape: {output.shape}, type: {output.type}")
      # Load input data
    print("\nLoading input data from example.txt...")
    input_file = 'C:/Users/StrangeMP/OneDrive/Desktop/15_by_15_AlphaGomoku-demo/quant/example.txt'
    input_tensor = load_input_from_file(input_file)
    
    if isinstance(input_tensor, np.ndarray):
        print(f"Loaded input tensor with shape: {input_tensor.shape}")
        print(f"Input tensor dtype: {input_tensor.dtype}")
        
        # Add batch dimension: (1, 3, 15, 15)
        input_data = np.expand_dims(input_tensor, axis=0)
        print(f"Input data shape for inference: {input_data.shape}")
        
        # Run inference
        try:
            outputs = session.run(None, {input_info.name: input_data})
            
            print(f"\nInference successful!")
            print(f"Number of outputs: {len(outputs)}")
            
            for j, output in enumerate(outputs):
                print(f"\nOutput {j}:")
                print(f"  Shape: {output.shape}")
                print(f"  Type: {output.dtype}")
                print(f"  Sample values: {output.flatten()[:10]}")  # Show first 10 values
                
                # If this looks like policy/value output (common in AlphaGo-style models)
                if output.shape[-1] == 225:  # 15x15 = 225 positions
                    print(f"  -> This appears to be policy output (move probabilities)")
                    # Reshape to 15x15 for visualization
                    policy = output.reshape(15, 15)
                    max_pos = np.unravel_index(np.argmax(policy), policy.shape)
                    print(f"  -> Best move position: row {max_pos[0]}, col {max_pos[1]}")
                    print(f"  -> Max probability value: {np.max(policy):.6f}")
                    print(f"  -> Min probability value: {np.min(policy):.6f}")
                    print(f"  -> Mean probability value: {np.mean(policy):.6f}")
                elif output.shape[-1] == 1:
                    print(f"  -> This appears to be value output: {output[0][0]:.6f}")
                    print(f"  -> Value interpretation: {'Favorable' if output[0][0] > 0 else 'Unfavorable' if output[0][0] < 0 else 'Neutral'}")
                else:
                    print(f"  -> Output interpretation: Unknown format")
            
            return outputs
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return None
    else:
        print("Failed to load input tensor properly")
        return None

if __name__ == "__main__":
    try:
        results = run_inference()
        print(f"\nInference completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure onnxruntime is installed: pip install onnxruntime")
