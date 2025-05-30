#!/usr/bin/env python3
"""
Convert exported model parameters from .npz files to C++ header format.
Handles different data types with appropriate formatting:
- Integers: decimal format
- Floating point: hexadecimal format (no precision loss)
"""

import numpy as np
import os
from pathlib import Path

def get_cpp_type(dtype):
    """Map numpy dtype to C++ type."""
    dtype_map = {
        'int8': 'int8_t',
        'int16': 'int16_t', 
        'int32': 'int32_t',
        'int64': 'int64_t',
        'uint8': 'uint8_t',
        'uint16': 'uint16_t',
        'uint32': 'uint32_t', 
        'uint64': 'uint64_t',
        'float16': 'float',  # C++ doesn't have native float16, use float
        'float32': 'float',
        'float64': 'double'
    }
    return dtype_map.get(str(dtype), 'float')

def format_value(value, dtype):
    """Format a single value based on its data type."""
    if np.issubdtype(dtype, np.integer):
        # Integer types: use decimal format
        return str(int(value))
    elif np.issubdtype(dtype, np.floating):
        # Floating point types: use hex format for precision
        if dtype == np.float16:
            # Convert to float32 first for hex representation
            return float(value).hex()
        elif dtype == np.float32:
            return float(value).hex()
        elif dtype == np.float64:
            return float(value).hex()
        else:
            return float(value).hex()
    else:
        # Default to string representation
        return str(value)

def format_initializer_list(array):
    """Format array values as initializer list for custom C++ data types."""
    dtype = array.dtype
    
    if array.size == 0:
        return "{}"
    
    def format_nested_array(arr, depth=0):
        """Recursively format multi-dimensional arrays with proper nesting."""
        indent = "    " * (depth + 1)
        
        if arr.ndim == 1:
            # Base case: 1D array - format values in a single line or wrapped lines
            values = [format_value(val, dtype) for val in arr]
            
            # Group values for readability
            if np.issubdtype(dtype, np.integer):
                values_per_line = 8
            else:
                values_per_line = 4
            
            if len(values) <= values_per_line:
                # Single line
                return "{" + ", ".join(values) + "}"
            else:
                # Multiple lines
                lines = []
                for i in range(0, len(values), values_per_line):
                    line_values = values[i:i + values_per_line]
                    if i == 0:
                        lines.append("{" + ", ".join(line_values))
                    else:
                        lines.append(indent + " " + ", ".join(line_values))
                
                # Close the last line
                lines[-1] += "}"
                return ",\n".join(lines)
        else:
            # Recursive case: multi-dimensional array
            sub_arrays = []
            for i in range(arr.shape[0]):
                sub_array_str = format_nested_array(arr[i], depth + 1)
                # Add proper indentation for sub-arrays
                if '\n' in sub_array_str:
                    # Multi-line sub-array
                    lines = sub_array_str.split('\n')
                    indented_lines = [indent + lines[0]]
                    for line in lines[1:]:
                        indented_lines.append(indent + line)
                    sub_arrays.append('\n'.join(indented_lines))
                else:
                    # Single-line sub-array
                    sub_arrays.append(indent + sub_array_str)
            
            return "{\n" + ",\n".join(sub_arrays) + "\n" + "    " * depth + "}"
    
    return format_nested_array(array)

def get_custom_cpp_type(param_data):
    """Determine the custom C++ type based on array dimensions."""
    shape = param_data.shape
    cpp_type = get_cpp_type(param_data.dtype)
    
    if len(shape) == 0 or param_data.size == 1:
        # Scalar
        return cpp_type, False
    elif len(shape) == 1:
        # 1D array -> Vec<T, L>
        return f"Vec<{cpp_type}, {shape[0]}>", True
    elif len(shape) == 3:
        # 3D array -> Tensor<T, Channels, Height, Width>
        # Assuming shape is (C, H, W)
        channels, height, width = shape
        return f"Tensor<{cpp_type}, {channels}, {height}, {width}>", True
    elif len(shape) == 4:
        # 4D array -> Vec<Tensor<T, C, H, W>, L>
        # Assuming shape is (N, C, H, W)
        n, channels, height, width = shape
        return f"Vec<Tensor<{cpp_type}, {channels}, {height}, {width}>, {n}>", True
    else:
        # Fallback to standard array for other dimensions
        dims = "][".join(str(dim) for dim in shape)
        return f"{cpp_type}[{dims}]", False

def sanitize_name(name):
    """Sanitize parameter name for C++ variable naming."""
    # Replace invalid characters with underscores
    sanitized = ""
    for char in name:
        if char.isalnum() or char == '_':
            sanitized += char
        else:
            sanitized += '_'
    
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = "param_" + sanitized
    
    return sanitized

def generate_header(npz_file_path, output_header_path, model_name):
    """Generate C++ header from .npz file."""
    
    # Load the .npz file
    data = np.load(npz_file_path)
    
    # Start building the header content
    header_content = []
    
    # Header guard
    guard_name = f"{model_name.upper()}_PARAMS_H"
    header_content.append(f"#ifndef {guard_name}")
    header_content.append(f"#define {guard_name}")
    header_content.append("")
    header_content.append("#include <cstdint>")
    header_content.append("#include <initializer_list>")
    header_content.append("")
    header_content.append("// Custom C++ data types")
    header_content.append("template<typename T, size_t L>")
    header_content.append("struct Vec {")
    header_content.append("    T data[L];")
    header_content.append("    constexpr Vec(std::initializer_list<T> init) {")
    header_content.append("        size_t i = 0;")
    header_content.append("        for (const auto& val : init) {")
    header_content.append("            if (i < L) data[i++] = val;")
    header_content.append("        }")
    header_content.append("    }")
    header_content.append("};")
    header_content.append("")
    header_content.append("template<typename T, size_t C, size_t H, size_t W>")
    header_content.append("struct Tensor {")
    header_content.append("    T data[C][H][W];")
    header_content.append("    constexpr Tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> init) {")
    header_content.append("        size_t c = 0;")
    header_content.append("        for (const auto& channel : init) {")
    header_content.append("            if (c >= C) break;")
    header_content.append("            size_t h = 0;")
    header_content.append("            for (const auto& row : channel) {")
    header_content.append("                if (h >= H) break;")
    header_content.append("                size_t w = 0;")
    header_content.append("                for (const auto& val : row) {")
    header_content.append("                    if (w >= W) break;")
    header_content.append("                    data[c][h][w] = val;")
    header_content.append("                    w++;")
    header_content.append("                }")
    header_content.append("                h++;")
    header_content.append("            }")
    header_content.append("            c++;")
    header_content.append("        }")
    header_content.append("    }")
    header_content.append("};")
    header_content.append("")
    header_content.append(f"namespace {model_name}_params {{")
    header_content.append("")
    
    # Sort parameters by name for consistent output
    param_names = sorted(data.files)
    
    # Generate constants for each parameter
    for param_name in param_names:
        param_data = data[param_name]
        sanitized_name = sanitize_name(param_name)
        
        # Add comment with original name and shape info
        header_content.append(f"// Original name: {param_name}")
        header_content.append(f"// Shape: {param_data.shape}, dtype: {param_data.dtype}")
        
        # Determine the appropriate C++ type and declaration
        custom_type, use_custom = get_custom_cpp_type(param_data)
        
        if param_data.size == 1:
            # Scalar value
            value = format_value(param_data.item(), param_data.dtype)
            header_content.append(f"constexpr {custom_type} {sanitized_name} = {value};")
        else:
            # Array value using custom types or initializer lists
            formatted_values = format_initializer_list(param_data)
            header_content.append(f"constexpr {custom_type} {sanitized_name} = {formatted_values};")
        
        header_content.append("")
    
    # Add array size constants
    header_content.append("// Array sizes")
    for param_name in param_names:
        param_data = data[param_name]
        if param_data.size > 1:
            sanitized_name = sanitize_name(param_name)
            header_content.append(f"constexpr size_t {sanitized_name}_size = {param_data.size};")
    
    header_content.append("")
    header_content.append(f"}} // namespace {model_name}_params")
    header_content.append("")
    header_content.append(f"#endif // {guard_name}")
    
    # Write to file
    with open(output_header_path, 'w') as f:
        f.write('\n'.join(header_content))
    
    print(f"Generated C++ header: {output_header_path}")
    print(f"Total parameters: {len(param_names)}")
    
    # Print statistics
    scalar_count = sum(1 for name in param_names if data[name].size == 1)
    array_count = len(param_names) - scalar_count
    print(f"Scalars: {scalar_count}, Arrays: {array_count}")

def main():
    """Main function to process both model files."""
    
    # Define file paths
    quant_dir = Path(__file__).parent
    
    models = [
        {
            'npz_file': quant_dir / 'model_w_15_params.npz',
            'header_file': quant_dir / 'model_w_15_params.h',
            'model_name': 'model_w_15'
        },
        {
            'npz_file': quant_dir / 'model_b_15_params.npz', 
            'header_file': quant_dir / 'model_b_15_params.h',
            'model_name': 'model_b_15'
        }
    ]
    
    for model in models:
        npz_file = model['npz_file']
        header_file = model['header_file']
        model_name = model['model_name']
        
        if npz_file.exists():
            print(f"\n{'='*60}")
            print(f"Processing {model_name}")
            print(f"{'='*60}")
            generate_header(npz_file, header_file, model_name)
        else:
            print(f"Warning: {npz_file} not found, skipping...")

if __name__ == "__main__":
    main()