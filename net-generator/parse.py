import onnx
from onnx import numpy_helper
import json
import numpy as np

def tensor_to_list(tensor):
    np_array = numpy_helper.to_array(tensor)
    return np_array.tolist() if np_array.size > 1 else np_array.item()

# def remove_redundant_quant_dequant(model_info):
#     """Remove redundant QuantizeLinear and DequantizeLinear nodes."""
#     nodes = model_info["nodes"]
#     updated_nodes = []
#     quant_to_dequant_map = {}

#     # Identify QuantizeLinear and DequantizeLinear pairs
#     for node in nodes:
#         if node["type"] == "QuantizeLinear":
#             quant_to_dequant_map[node["outputs"][0]] = node
#             continue  # Skip adding this QuantizeLinear node
#         elif node["type"] == "DequantizeLinear":
#             input_tensor = node["inputs"][0]
#             if input_tensor in quant_to_dequant_map:
#                 quant_node = quant_to_dequant_map[input_tensor]
#                 # Check if they share the same scale and zero-point
#                 if quant_node["inputs"][1:] == node["inputs"][1:]:
#                     # Update connections to bypass these nodes
#                     for downstream_node in nodes:
#                         downstream_node["inputs"] = [
#                             quant_node["inputs"][0] if inp == node["outputs"][0] else inp
#                             for inp in downstream_node["inputs"]
#                         ]
#                     continue  # Skip adding this DequantizeLinear node
#         updated_nodes.append(node)

#     model_info["nodes"] = updated_nodes

def extract_model_info(onnx_path):
    model = onnx.load(onnx_path)
    graph = model.graph

    model_info = {
        "inputs": [],
        "outputs": [],
        "initializers": {},
        "nodes": []
    }

    initializer_names = {init.name for init in graph.initializer}

    # Inputs (exclude initializers)
    for input_tensor in graph.input:
        if input_tensor.name not in initializer_names:
            tensor_type = input_tensor.type.tensor_type
            dims = [dim.dim_value for dim in tensor_type.shape.dim]
            model_info["inputs"].append({
                "name": input_tensor.name,
                "dims": dims,
                "data_type": tensor_type.elem_type
            })

    # Outputs
    for output_tensor in graph.output:
        tensor_type = output_tensor.type.tensor_type
        dims = [dim.dim_value for dim in tensor_type.shape.dim]
        model_info["outputs"].append({
            "name": output_tensor.name,
            "dims": dims,
            "data_type": tensor_type.elem_type
        })

    # Initializers
    for init in graph.initializer:
        model_info["initializers"][init.name] = {
            "dims": list(init.dims),
            # "data": tensor_to_list(init),
            "data_type": init.data_type
        }

    # Nodes
    for node in graph.node:
        model_info["nodes"].append({
            "type": node.op_type,
            "name": node.name,
            "inputs": list(node.input),
            "outputs": list(node.output),
            # "attributes": {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        })

    # Remove redundant QuantizeLinear and DequantizeLinear nodes
    # remove_redundant_quant_dequant(model_info)

    return model_info


model_dir = "C:/Users/StrangeMP/OneDrive/Desktop/15_by_15_AlphaGomoku-demo/quant/model/"
model_files = [
    "model_w_15_quant.onnx",
    "model_b_15_quant.onnx",
]
for model_file in model_files:
    model_path = model_dir + model_file
    model_info = extract_model_info(model_path)
    output_file = model_dir + model_file.replace(".onnx", "_info.json")
    
    with open(output_file, 'w') as f:
        json.dump(model_info, f, indent=4)

    print(f"Extracted info for {model_file} and saved to {output_file}")

