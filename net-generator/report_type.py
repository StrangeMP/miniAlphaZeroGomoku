"""
Script to load ONNX models and report data types used for different node types.
"""

import onnx
import os
from collections import defaultdict, Counter
import argparse


def get_onnx_data_type_name(data_type):
    """Convert ONNX data type integer to human-readable string."""
    type_map = {
        1: "FLOAT",
        2: "UINT8", 
        3: "INT8",
        4: "UINT16",
        5: "INT16",
        6: "INT32",
        7: "INT64",
        8: "STRING",
        9: "BOOL",
        10: "FLOAT16",
        11: "DOUBLE",
        12: "UINT32",
        13: "UINT64",
        14: "COMPLEX64",
        15: "COMPLEX128",
        16: "BFLOAT16"
    }
    return type_map.get(data_type, f"UNKNOWN({data_type})")


def analyze_onnx_model(model_path):
    """Analyze ONNX model and report data types for different node types."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    try:
        # Load the ONNX model
        model = onnx.load(model_path)
        
        print(f"Analyzing ONNX model: {os.path.basename(model_path)}")
        print("=" * 60)
        
        # Get model info
        print(f"IR Version: {model.ir_version}")
        print(f"Producer Name: {model.producer_name}")
        print(f"Producer Version: {model.producer_version}")
        print(f"Domain: {model.domain}")
        print(f"Model Version: {model.model_version}")
        print(f"Opset Version: {model.opset_import[0].version if model.opset_import else 'N/A'}")
        print()
        
        # Analyze inputs
        print("Model Inputs:")
        print("-" * 30)
        for input_tensor in model.graph.input:
            tensor_type = input_tensor.type.tensor_type
            data_type = get_onnx_data_type_name(tensor_type.elem_type)
            shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in tensor_type.shape.dim]
            print(f"  Name: {input_tensor.name}")
            print(f"  Data Type: {data_type}")
            print(f"  Shape: {shape}")
            print()
        
        # Analyze outputs
        print("Model Outputs:")
        print("-" * 30)
        for output_tensor in model.graph.output:
            tensor_type = output_tensor.type.tensor_type
            data_type = get_onnx_data_type_name(tensor_type.elem_type)
            shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param for dim in tensor_type.shape.dim]
            print(f"  Name: {output_tensor.name}")
            print(f"  Data Type: {data_type}")
            print(f"  Shape: {shape}")
            print()
        
        # Analyze nodes and their data types
        node_type_data_types = defaultdict(set)
        node_type_counts = Counter()
        
        # Create a mapping of tensor names to their data types
        tensor_data_types = {}
        
        # Add input tensor data types
        for input_tensor in model.graph.input:
            tensor_type = input_tensor.type.tensor_type
            data_type = get_onnx_data_type_name(tensor_type.elem_type)
            tensor_data_types[input_tensor.name] = data_type
        
        # Add initializer (weight/bias) data types
        for initializer in model.graph.initializer:
            data_type = get_onnx_data_type_name(initializer.data_type)
            tensor_data_types[initializer.name] = data_type
        
        # Add value_info data types (intermediate tensors)
        for value_info in model.graph.value_info:
            tensor_type = value_info.type.tensor_type
            data_type = get_onnx_data_type_name(tensor_type.elem_type)
            tensor_data_types[value_info.name] = data_type
        
        # Analyze each node
        for node in model.graph.node:
            op_type = node.op_type
            node_type_counts[op_type] += 1
            
            # Collect data types from input tensors
            for input_name in node.input:
                if input_name in tensor_data_types:
                    node_type_data_types[op_type].add(tensor_data_types[input_name])
            
            # Collect data types from output tensors
            for output_name in node.output:
                if output_name in tensor_data_types:
                    node_type_data_types[op_type].add(tensor_data_types[output_name])
        
        # Report node types and their data types
        print("Node Types and Data Types:")
        print("-" * 30)
        for op_type in sorted(node_type_counts.keys()):
            count = node_type_counts[op_type]
            data_types = sorted(node_type_data_types[op_type])
            print(f"  {op_type}: {count} nodes")
            if data_types:
                print(f"    Data types used: {', '.join(data_types)}")
            else:
                print(f"    Data types used: Not determined")
            print()
        
        # Summary statistics
        print("Summary:")
        print("-" * 30)
        print(f"Total nodes: {len(model.graph.node)}")
        print(f"Unique node types: {len(node_type_counts)}")
        print(f"Total initializers: {len(model.graph.initializer)}")
        
        all_data_types = set()
        for data_types in node_type_data_types.values():
            all_data_types.update(data_types)
        print(f"Data types used in model: {', '.join(sorted(all_data_types))}")
        
    except Exception as e:
        print(f"Error analyzing model: {str(e)}")


def main():
    """Main function to handle command line arguments and run analysis."""
    parser = argparse.ArgumentParser(description="Analyze ONNX model data types")
    parser.add_argument("--model", type=str, help="Path to ONNX model file")
    parser.add_argument("--all", action="store_true", help="Analyze all ONNX models in the model directory")
    
    args = parser.parse_args()
    
    # Default model directory
    model_dir = 'C:/Users/StrangeMP/OneDrive/Desktop/15_by_15_AlphaGomoku-demo/quant/model/'
    
    if args.all:
        # Analyze all ONNX models in the directory
        if os.path.exists(model_dir):
            onnx_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
            if not onnx_files:
                print(f"No ONNX files found in {model_dir}")
                return
            
            for onnx_file in sorted(onnx_files):
                model_path = os.path.join(model_dir, onnx_file)
                analyze_onnx_model(model_path)
                print("\n" + "="*80 + "\n")
        else:
            print(f"Model directory not found: {model_dir}")
    
    elif args.model:
        # Analyze specific model
        analyze_onnx_model(args.model)
    
    else:
        # Default: analyze the quantized model
        default_model = os.path.join(model_dir, 'model_b_15_quant.onnx')
        if os.path.exists(default_model):
            analyze_onnx_model(default_model)
        else:
            print(f"Default model not found: {default_model}")
            print("Use --model <path> to specify a model or --all to analyze all models")


if __name__ == "__main__":
    main()
