import onnx
import numpy as np
import json
from onnx import numpy_helper
from collections import defaultdict
import os

def export_initializers_to_npz(model, output_path):
    """
    Export all initializers from the ONNX model to an NPZ file.
    Each initializer is saved with its name as the key.
    """
    initializers_dict = {}
    
    print(f"Exporting {len(model.graph.initializer)} initializers...")
    
    for initializer in model.graph.initializer:
        # Convert ONNX tensor to numpy array
        np_array = numpy_helper.to_array(initializer)
        initializers_dict[initializer.name] = np_array
        print(f"  {initializer.name}: shape {np_array.shape}, dtype {np_array.dtype}")
    
    # Save to NPZ file
    np.savez_compressed(output_path, **initializers_dict)
    print(f"Saved {len(initializers_dict)} initializers to: {output_path}")
    
    return initializers_dict

def generate_execution_plan(model, output_path):
    """
    Generate an execution plan JSON file showing node connections and dependencies.
    """
    graph = model.graph
    
    # Build mappings
    output_to_node = {}  # Maps output name to the node that produces it
    input_to_nodes = defaultdict(list)  # Maps input name to nodes that consume it
    
    # Process all nodes
    for i, node in enumerate(graph.node):
        # Map outputs to this node
        for output in node.output:
            output_to_node[output] = {
                'node_index': i,
                'node_name': node.name,
                'node_type': node.op_type
            }
        
        # Map inputs to this node
        for input_name in node.input:
            input_to_nodes[input_name].append({
                'node_index': i,
                'node_name': node.name,
                'node_type': node.op_type
            })
    
    # Build execution plan
    execution_plan = {
        'model_info': {
            'ir_version': model.ir_version,
            'total_nodes': len(graph.node),
            'total_initializers': len(graph.initializer),
            'model_inputs': [inp.name for inp in graph.input],
            'model_outputs': [out.name for out in graph.output]
        },
        'nodes': [],
        'connections': [],
        'initializers': [init.name for init in graph.initializer]
    }
    
    # Process each node
    for i, node in enumerate(graph.node):
        node_info = {
            'index': i,
            'name': node.name,
            'op_type': node.op_type,
            'inputs': list(node.input),
            'outputs': list(node.output),
            'input_count': len(node.input),
            'output_count': len(node.output),
            'attributes': {}
        }
        
        # Extract attributes
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INT:
                node_info['attributes'][attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                node_info['attributes'][attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.STRING:
                node_info['attributes'][attr.name] = attr.s.decode('utf-8')
            elif attr.type == onnx.AttributeProto.INTS:
                node_info['attributes'][attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOATS:
                node_info['attributes'][attr.name] = list(attr.floats)
            else:
                node_info['attributes'][attr.name] = f"<{attr.type}>"
        
        execution_plan['nodes'].append(node_info)
        
        # Build connections for this node
        for input_name in node.input:
            if input_name in output_to_node:
                # This input comes from another node
                producer = output_to_node[input_name]
                connection = {
                    'from_node': producer['node_index'],
                    'from_node_name': producer['node_name'],
                    'from_node_type': producer['node_type'],
                    'to_node': i,
                    'to_node_name': node.name,
                    'to_node_type': node.op_type,
                    'tensor_name': input_name
                }
                execution_plan['connections'].append(connection)
            elif input_name in [init.name for init in graph.initializer]:
                # This input is an initializer
                connection = {
                    'from_node': 'INITIALIZER',
                    'from_node_name': input_name,
                    'from_node_type': 'Constant',
                    'to_node': i,
                    'to_node_name': node.name,
                    'to_node_type': node.op_type,
                    'tensor_name': input_name
                }
                execution_plan['connections'].append(connection)
            elif input_name in [inp.name for inp in graph.input]:
                # This input is a model input
                connection = {
                    'from_node': 'MODEL_INPUT',
                    'from_node_name': input_name,
                    'from_node_type': 'Input',
                    'to_node': i,
                    'to_node_name': node.name,
                    'to_node_type': node.op_type,
                    'tensor_name': input_name
                }
                execution_plan['connections'].append(connection)
    
    # Add dependency analysis
    execution_plan['dependency_analysis'] = analyze_dependencies(graph)
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(execution_plan, f, indent=2)
    
    print(f"Execution plan saved to: {output_path}")
    print(f"  Nodes: {len(execution_plan['nodes'])}")
    print(f"  Connections: {len(execution_plan['connections'])}")
    
    return execution_plan

def analyze_dependencies(graph):
    """
    Analyze node dependencies and execution order.
    """
    output_to_node = {}
    node_dependencies = defaultdict(list)
    
    # Build output to node mapping
    for i, node in enumerate(graph.node):
        for output in node.output:
            output_to_node[output] = i
    
    # Build dependency graph
    for i, node in enumerate(graph.node):
        for input_name in node.input:
            if input_name in output_to_node:
                producer_idx = output_to_node[input_name]
                node_dependencies[i].append(producer_idx)
    
    # Calculate execution levels (topological sort-like)
    levels = {}
    unprocessed = set(range(len(graph.node)))
    level = 0
    
    while unprocessed:
        current_level_nodes = []
        for node_idx in list(unprocessed):
            # Check if all dependencies are satisfied
            deps_satisfied = all(dep in levels for dep in node_dependencies[node_idx])
            if deps_satisfied:
                current_level_nodes.append(node_idx)
        
        if not current_level_nodes:
            # Handle circular dependencies or other issues
            current_level_nodes = list(unprocessed)
        
        for node_idx in current_level_nodes:
            levels[node_idx] = level
            unprocessed.remove(node_idx)
        
        level += 1
    
    # Build dependency analysis
    dependency_info = {
        'execution_levels': {},
        'node_dependencies': {},
        'max_level': max(levels.values()) if levels else 0
    }
    
    # Group nodes by level
    for node_idx, level in levels.items():
        if level not in dependency_info['execution_levels']:
            dependency_info['execution_levels'][level] = []
        dependency_info['execution_levels'][level].append({
            'node_index': node_idx,
            'node_name': graph.node[node_idx].name,
            'node_type': graph.node[node_idx].op_type
        })
    
    # Add dependency info for each node
    for node_idx, deps in node_dependencies.items():
        dependency_info['node_dependencies'][node_idx] = {
            'depends_on': deps,
            'dependency_count': len(deps),
            'execution_level': levels[node_idx]
        }
    
    return dependency_info

def finalize_models():
    """
    Main function to finalize both optimized models.
    """
    model_dir = "C:/Users/StrangeMP/OneDrive/Desktop/15_by_15_AlphaGomoku-demo/quant/model/"
    output_dir = "C:/Users/StrangeMP/OneDrive/Desktop/15_by_15_AlphaGomoku-demo/quant/"
    
    for color in ['w', 'b']:
        model_path = f"{model_dir}model_{color}_15_quant.onnx"
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Finalizing model: {color.upper()}")
        print(f"{'='*60}")
        
        # Load the optimized model
        model = onnx.load(model_path)
        
        # Export initializers to NPZ
        npz_path = f"{output_dir}model_{color}_15_params.npz"
        initializers = export_initializers_to_npz(model, npz_path)
        
        # Generate execution plan
        json_path = f"{output_dir}model_{color}_15_execution_plan.json"
        execution_plan = generate_execution_plan(model, json_path)
        
        print(f"\nFinalization complete for model {color.upper()}:")
        print(f"  Parameters: {npz_path}")
        print(f"  Execution plan: {json_path}")
        print(f"  Total parameters: {len(initializers)}")
        print(f"  Total nodes: {len(execution_plan['nodes'])}")

if __name__ == "__main__":
    finalize_models()