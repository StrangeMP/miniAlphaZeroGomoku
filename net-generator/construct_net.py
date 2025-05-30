"""
input format of different nodes in the quantized graph:

For QLinearConv nodes,  [InputTensor, InputScale, InputZP, WeightTensor, WeightScale, WeightZP, OutputScale, OutputZP, BiasTensor(optional, if specified)]

For QLinearMul, QLinearAdd,  [InputTensor, InputScale, InputZP, BTensor, BScale, BZP, OutputScale, OutputZP] where B refers to the second input tensor.

For QGemm, [InputTensor, InputScale, InputZP, BTensor, BScale, BZP, CTensor, CScale, CZP] where B and C refers to the another two input tensors.

For Reshape nodes, [InputTensor, Shape]

For Softmax and Tanh, [InputTensor]

"""

import json
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

working_color = 'b'

@dataclass
class TensorInfo:
    """Information about a tensor in the network"""

    name: str
    scale: Optional[float] = None
    zero_point: Optional[int] = None
    dtype: str = "int8"  # Default for quantized tensors
    shape: Optional[List[int]] = None

    def __post_init__(self):
        # Extract scale and zero_point from tensor name if they follow the pattern
        if self.name.endswith("_scale") and self.scale is None:
            # This is a scale tensor, we'll populate it later
            pass
        elif self.name.endswith("_zero_point") and self.zero_point is None:
            # This is a zero_point tensor, we'll populate it later
            pass


class Node(ABC):
    """Abstract base class for all network nodes"""

    def __init__(
        self,
        index: int,
        name: str,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        attributes: Dict[str, Any] = None,
    ):
        self.index = index
        self.name = name
        self.op_type = op_type
        self.input_names = inputs
        self.output_names = outputs
        self.attributes = attributes or {}

        # Will be populated later during graph construction
        self.input_tensors: List[TensorInfo] = []
        self.output_tensors: List[TensorInfo] = []
        self.producer_nodes: List["Node"] = []
        self.consumer_nodes: List["Node"] = []

    @abstractmethod
    def get_cpp_template_args(self) -> str:
        """Generate C++ template arguments for this node type"""
        pass

    @abstractmethod
    def get_cpp_constructor_args(self) -> str:
        """Generate C++ constructor arguments for this node type"""
        pass

    @abstractmethod
    def get_cpp_member_declaration(self, var_name: str) -> str:
        """Generate C++ member variable declaration"""
        pass

    @abstractmethod
    def get_cpp_forward_call(
        self, var_name: str, input_vars: List[str], output_var: str = ""
    ) -> str:
        """Generate C++ forward pass call"""
        pass

    def get_sanitized_name(self) -> str:
        """Get a sanitized C++ variable name from node name"""
        # Replace problematic characters with underscores
        sanitized = self.name.replace("/", "_").replace(":", "_").replace("-", "_")
        # Remove leading numbers if any
        if sanitized[0].isdigit():
            sanitized = "node_" + sanitized
        return sanitized

    @staticmethod
    def tensor_name_to_cpp_var(tensor_name: str) -> str:
        """Convert tensor name to C++ variable name by replacing / and : with _"""
        return tensor_name.replace("/", "_").replace(":", "_")


class ConvNode(Node):
    """Convolution node implementation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Extract conv-specific attributes
        self.kernel_shape = self.attributes.get("kernel_shape", [3, 3])
        self.strides = self.attributes.get("strides", [1, 1])
        self.pads = self.attributes.get("pads", [1, 1, 1, 1])
        self.dilations = self.attributes.get("dilations", [1, 1])

        # Parse input format: [InputTensor, InputScale, InputZP, WeightTensor, WeightScale, WeightZP, OutputScale, OutputZP, BiasTensor(optional)]
        if len(self.input_names) >= 8:
            self.input_tensor = self.input_names[0]
            self.input_scale_name = self.input_names[1]
            self.input_zp_name = self.input_names[2]
            self.weight_tensor_name = self.input_names[3]
            self.weight_scale_name = self.input_names[4]
            self.weight_zp_name = self.input_names[5]
            self.output_scale_name = self.input_names[6]
            self.output_zp_name = self.input_names[7]
            # Optional bias tensor
            if len(self.input_names) > 8:
                self.bias_tensor_name = self.input_names[8]
            else:
                self.bias_tensor_name = None

    def get_cpp_template_args(self) -> str:
        # Template: <InputScale, InputZP, WeightScale, WeightZP, OutputScale, OutputZP, Filters, InputChannels, InputHeight, InputWidth, KernelHeight, KernelWidth>

        # Convert tensor names to C++ variable names
        input_scale_var = (
            self.tensor_name_to_cpp_var(self.input_scale_name)
            if hasattr(self, "input_scale_name")
            else "1.0f"
        )
        input_zp_var = (
            self.tensor_name_to_cpp_var(self.input_zp_name)
            if hasattr(self, "input_zp_name")
            else "0"
        )
        weight_scale_var = (
            self.tensor_name_to_cpp_var(self.weight_scale_name)
            if hasattr(self, "weight_scale_name")
            else "1.0f"
        )
        weight_zp_var = (
            self.tensor_name_to_cpp_var(self.weight_zp_name)
            if hasattr(self, "weight_zp_name")
            else "0"
        )
        output_scale_var = (
            self.tensor_name_to_cpp_var(self.output_scale_name)
            if hasattr(self, "output_scale_name")
            else "1.0f"
        )
        output_zp_var = (
            self.tensor_name_to_cpp_var(self.output_zp_name)
            if hasattr(self, "output_zp_name")
            else "0"
        )

        # Determine filters and input channels based on node index
        # Determine filters and input channels based on node index
        black_config = {
            1: {"filters": 32, "input_channels": 3},
            25: {"filters": 2, "input_channels": 32},
            26: {"filters": 1, "input_channels": 32}
        }
        
        white_config = {
            1: {"filters": 32, "input_channels": 3},
            26: {"filters": 2, "input_channels": 32},
            25: {"filters": 1, "input_channels": 32}
        }
        
        conv_config = white_config if working_color == 'w' else black_config
        
        config = conv_config.get(self.index, {"filters": 32, "input_channels": 32})
        filters = config["filters"]
        input_channels = config["input_channels"]

        return f"{input_scale_var}, {input_zp_var}, {weight_scale_var}, {weight_zp_var}, {output_scale_var}, {output_zp_var}, /* Filters */ {filters}, /* InputChannels */ {input_channels}, /* InputHeight */ 15, /* InputWidth */ 15, /* KernelHeight */ {self.kernel_shape[0]}, /* KernelWidth */ {self.kernel_shape[1]}"

    def get_cpp_constructor_args(self) -> str:
        weight_var = (
            self.tensor_name_to_cpp_var(self.weight_tensor_name)
            if hasattr(self, "weight_tensor_name")
            else "default_weights"
        )
        if hasattr(self, "bias_tensor_name") and self.bias_tensor_name:
            bias_var = self.tensor_name_to_cpp_var(self.bias_tensor_name)
            return f"{weight_var}, {bias_var}"
        else:
            return f"{weight_var}"

    def get_cpp_member_declaration(self, var_name: str) -> str:
        return f"QLinearConv<{self.get_cpp_template_args()}> {var_name};"

    def get_cpp_forward_call(
        self, var_name: str, input_vars: List[str], output_var: str = ""
    ) -> str:
        input_tensor = input_vars[0] if input_vars else "input"
        return f"{var_name}.feed({input_tensor});\n"


class ElemWiseNode(Node):
    """Element-wise operation (Add/Mul) node implementation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Parse input format: [InputTensor, InputScale, InputZP, BTensor, BScale, BZP, OutputScale, OutputZP]
        if len(self.input_names) >= 8:
            self.input_tensor = self.input_names[0]
            self.input_scale_name = self.input_names[1]
            self.input_zp_name = self.input_names[2]
            self.b_tensor_name = self.input_names[3]
            self.b_scale_name = self.input_names[4]
            self.b_zp_name = self.input_names[5]
            self.output_scale_name = self.input_names[6]
            self.output_zp_name = self.input_names[7]

    def get_cpp_template_args(self) -> str:
        # Template: <a_scale, a_zp, b_scale, b_zp, c_scale, c_zp, C, H, W>
        input_scale_var = (
            self.tensor_name_to_cpp_var(self.input_scale_name)
            if hasattr(self, "input_scale_name")
            else "1.0f"
        )
        input_zp_var = (
            self.tensor_name_to_cpp_var(self.input_zp_name)
            if hasattr(self, "input_zp_name")
            else "0"
        )
        b_scale_var = (
            self.tensor_name_to_cpp_var(self.b_scale_name)
            if hasattr(self, "b_scale_name")
            else "1.0f"
        )
        b_zp_var = (
            self.tensor_name_to_cpp_var(self.b_zp_name)
            if hasattr(self, "b_zp_name")
            else "0"
        )
        output_scale_var = (
            self.tensor_name_to_cpp_var(self.output_scale_name)
            if hasattr(self, "output_scale_name")
            else "1.0f"
        )
        output_zp_var = (
            self.tensor_name_to_cpp_var(self.output_zp_name)
            if hasattr(self, "output_zp_name")
            else "0"
        )

        return f"/* a_scale */ {input_scale_var}, /* a_zp */ {input_zp_var}, /* b_scale */ {b_scale_var}, /* b_zp */ {b_zp_var}, /* c_scale */ {output_scale_var}, /* c_zp */ {output_zp_var}, /* C */ 32, /* H */ 15, /* W */ 15"

    def get_cpp_constructor_args(self) -> str:
        return ""  # QElemWise is a static class

    def get_cpp_member_declaration(self, var_name: str) -> str:
        return f"// QElemWise<{self.get_cpp_template_args()}> - static methods only"

    def get_cpp_forward_call(
        self, var_name: str, input_vars: List[str], output_var: str = ""
    ) -> str:
        # ElemWise performs in-place operations on the input tensor
        input_tensor = input_vars[0]
        b_tensor = input_vars[1]

        if self.op_type in ["QLinearAdd"]:
            return f"QElemWise<{self.get_cpp_template_args()}>::add({input_tensor}, {b_tensor});"
        elif self.op_type in ["QLinearMul"]:
            return f"QElemWise<{self.get_cpp_template_args()}>::mul({input_tensor}, {b_tensor});"
        return "// Error: unknown element-wise operation"


class BNNode(Node):
    """Batch Normalization operation node implementation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_mul = self.op_type in ["QLinearMul"]  # True for mul, False for add
        # Parse input format similar to ElemWise: [InputTensor, InputScale, InputZP, BTensor, BScale, BZP, OutputScale, OutputZP]
        if len(self.input_names) >= 8:
            self.input_tensor = self.input_names[0]
            self.input_scale_name = self.input_names[1]
            self.input_zp_name = self.input_names[2]
            self.b_tensor_name = self.input_names[3]
            self.b_scale_name = self.input_names[4]
            self.b_zp_name = self.input_names[5]
            self.output_scale_name = self.input_names[6]
            self.output_zp_name = self.input_names[7]

    def get_cpp_template_args(self) -> str:
        # Template: <a_scale, a_zp, b_scale, b_zp, c_scale, c_zp, C, H, W, IsMul>
        input_scale_var = (
            self.tensor_name_to_cpp_var(self.input_scale_name)
            if hasattr(self, "input_scale_name")
            else "1.0f"
        )
        input_zp_var = (
            self.tensor_name_to_cpp_var(self.input_zp_name)
            if hasattr(self, "input_zp_name")
            else "0"
        )
        b_scale_var = (
            self.tensor_name_to_cpp_var(self.b_scale_name)
            if hasattr(self, "b_scale_name")
            else "1.0f"
        )
        b_zp_var = (
            self.tensor_name_to_cpp_var(self.b_zp_name)
            if hasattr(self, "b_zp_name")
            else "0"
        )
        output_scale_var = (
            self.tensor_name_to_cpp_var(self.output_scale_name)
            if hasattr(self, "output_scale_name")
            else "1.0f"
        )
        output_zp_var = (
            self.tensor_name_to_cpp_var(self.output_zp_name)
            if hasattr(self, "output_zp_name")
            else "0"
        )

        # Determine channel count based on corresponding conv layer filter count
        # BN nodes typically follow Conv nodes, so we use similar logic
        black_channels_map = {
            28: 1,
            30: 1,
            27: 2,
            29: 2,
        }
        
        white_channels_map = {
            27: 1,
            29: 1,
            28: 2,
            30: 2,
        }
        
        channels_map = black_channels_map if working_color == 'b' else white_channels_map
        
        channels = channels_map.get(self.index, 32)  # Default to 32 if not found

        return f"/* a_scale */ {input_scale_var}, /* a_zp */ {input_zp_var}, /* b_scale */ {b_scale_var}, /* b_zp */ {b_zp_var}, /* c_scale */ {output_scale_var}, /* c_zp */ {output_zp_var}, /* C */ {channels}, /* H */ 15, /* W */ 15, /* IsMul */ {'true' if self.is_mul else 'false'}"

    def get_cpp_constructor_args(self) -> str:
        const_var = (
            self.tensor_name_to_cpp_var(self.b_tensor_name)
            if hasattr(self, "b_tensor_name")
            else "constants"
        )
        return f"{const_var}"

    def get_cpp_member_declaration(self, var_name: str) -> str:
        return f"BNop<{self.get_cpp_template_args()}> {var_name};"

    def get_cpp_forward_call(
        self, var_name: str, input_vars: List[str], output_var: str = ""
    ) -> str:
        # BNop performs in-place operations on the input tensor
        input_tensor = input_vars[0] if input_vars else "input"
        if self.is_mul:
            return f"{var_name}.mul({input_tensor});"
        else:
            return f"{var_name}.add({input_tensor});"


class QGemmNode(Node):
    """QGemm (matrix multiplication) node implementation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Parse input format: [InputTensor, InputScale, InputZP, BTensor, BScale, BZP, CTensor, CScale, CZP]
        if len(self.input_names) >= 9:
            self.input_tensor = self.input_names[0]
            self.input_scale_name = self.input_names[1]
            self.input_zp_name = self.input_names[2]
            self.b_tensor_name = self.input_names[3]  # Weight matrix
            self.b_scale_name = self.input_names[4]
            self.b_zp_name = self.input_names[5]
            self.c_tensor_name = self.input_names[6]  # Bias vector
            self.c_scale_name = self.input_names[7]
            self.c_zp_name = self.input_names[8]

    def get_cpp_template_args(self) -> str:
        # Template: <a_scale, a_zp, b_scale, b_zp, c_scale, c_zp, FlattenLen, Filters>
        input_scale_var = (
            self.tensor_name_to_cpp_var(self.input_scale_name)
            if hasattr(self, "input_scale_name")
            else "1.0f"
        )
        input_zp_var = (
            self.tensor_name_to_cpp_var(self.input_zp_name)
            if hasattr(self, "input_zp_name")
            else "0"
        )
        b_scale_var = (
            self.tensor_name_to_cpp_var(self.b_scale_name)
            if hasattr(self, "b_scale_name")
            else "1.0f"
        )
        b_zp_var = (
            self.tensor_name_to_cpp_var(self.b_zp_name)
            if hasattr(self, "b_zp_name")
            else "0"
        )
        c_scale_var = (
            self.tensor_name_to_cpp_var(self.c_scale_name)
            if hasattr(self, "c_scale_name")
            else "1.0f"
        )
        c_zp_var = (
            self.tensor_name_to_cpp_var(self.c_zp_name)
            if hasattr(self, "c_zp_name")
            else "0"
        )

        # Determine FlattenLen and Filters based on node index
        # Determine FlattenLen and Filters based on node index
        black_config = {
            33: {"flatten_len": 450, "filters": 225, "requant": "false"}, # Softmax 
            34: {"flatten_len": 225, "filters": 32, "requant": "true"}, # Tanh
            36: {"flatten_len": 32, "filters": 1, "requant": "false"} # Tanh
        }
        
        white_config = {
            34: {"flatten_len": 450, "filters": 225, "requant": "false"}, # Softmax 
            33: {"flatten_len": 225, "filters": 32, "requant": "true"}, # Tanh
            35: {"flatten_len": 32, "filters": 1, "requant": "false"} # Tanh
        }
        
        node_config = white_config if working_color == 'w' else black_config
        
        config = node_config.get(self.index, {"flatten_len": 960, "filters": 225, "requant": "false"})
        flatten_len = config["flatten_len"]
        filters = config["filters"]
        requant = config["requant"]
        
        return f"/* a_scale */ {input_scale_var}, /* a_zp */ {input_zp_var}, /* b_scale */ {b_scale_var}, /* b_zp */ {b_zp_var}, /* c_scale */ {c_scale_var}, /* c_zp */ {c_zp_var}, /* FlattenLen */ {flatten_len}, /* Filters */ {filters}, /* Requant */ {requant}"

    def get_cpp_constructor_args(self) -> str:
        b_var = (
            self.tensor_name_to_cpp_var(self.b_tensor_name)
            if hasattr(self, "b_tensor_name")
            else "weights"
        )
        c_var = (
            self.tensor_name_to_cpp_var(self.c_tensor_name)
            if hasattr(self, "c_tensor_name")
            else "biases"
        )
        return f"{b_var}, {c_var}"

    def get_cpp_member_declaration(self, var_name: str) -> str:
        return f"QGemm<{self.get_cpp_template_args()}> {var_name};"

    def get_cpp_forward_call(
        self, var_name: str, input_vars: List[str], output_var: str = ""
    ) -> str:
        input_tensor = input_vars[0] if input_vars else "input"
        return f"{var_name}.feed({input_tensor});\n"


class ReshapeNode(Node):
    """Reshape node implementation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Parse input format: [InputTensor, InputScale, InputZP, OutputScale, OutputZP]
        if len(self.input_names) >= 5:
            self.input_tensor = self.input_names[0]
            self.input_scale_name = self.input_names[1]
            self.input_zp_name = self.input_names[2]
            self.output_scale_name = self.input_names[3]
            self.output_zp_name = self.input_names[4]

    def get_cpp_template_args(self) -> str:
        return ""  # Reshape doesn't need template args in our implementation

    def get_cpp_constructor_args(self) -> str:
        return ""

    def get_cpp_member_declaration(self, var_name: str) -> str:
        return f"// Reshape operation - handled inline"

    def get_cpp_forward_call(
        self, var_name: str, input_vars: List[str], output_var: str = ""
    ) -> str:
        input_tensor = input_vars[0] if input_vars else "input"
        return f"{input_tensor}.flatten(); // Reshape: {self.name}"


class ActivationNode(Node):
    """Activation function node (Softmax, Tanh) implementation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Parse input format: [InputTensor, InputScale, InputZP, OutputScale, OutputZP]
        if len(self.input_names) >= 5:
            self.input_tensor = self.input_names[0]
            self.input_scale_name = self.input_names[1]
            self.input_zp_name = self.input_names[2]
            self.output_scale_name = self.input_names[3]
            self.output_zp_name = self.input_names[4]

    def get_cpp_template_args(self) -> str:
        # Template: <InputScale, InputZP, OutputScale, OutputZP>
        input_scale_var = (
            self.tensor_name_to_cpp_var(self.input_scale_name)
            if hasattr(self, "input_scale_name")
            else "1.0f"
        )
        input_zp_var = (
            self.tensor_name_to_cpp_var(self.input_zp_name)
            if hasattr(self, "input_zp_name")
            else "0"
        )
        output_scale_var = (
            self.tensor_name_to_cpp_var(self.output_scale_name)
            if hasattr(self, "output_scale_name")
            else "1.0f"
        )
        output_zp_var = (
            self.tensor_name_to_cpp_var(self.output_zp_name)
            if hasattr(self, "output_zp_name")
            else "0"
        )

        return f"/* InputScale */ {input_scale_var}, /* InputZP */ {input_zp_var}, /* OutputScale */ {output_scale_var}, /* OutputZP */ {output_zp_var}"

    def get_cpp_constructor_args(self) -> str:
        return ""

    def get_cpp_member_declaration(self, var_name: str) -> str:
        return f"// {self.op_type} activation - handled inline"

    def get_cpp_forward_call(
        self, var_name: str, input_vars: List[str], output_var: str = ""
    ) -> str:
        input_tensor = input_vars[0] if input_vars else "input"
        if self.op_type == "Softmax":
            return f"pi = softmax({input_tensor}); // Softmax activation"
        elif self.op_type == "Tanh":
          return f"v = Tanh::call({input_tensor}[0]); // Tanh activation"
        return f"// Unknown activation: {self.op_type}"


class NetworkGraph:
    """Container for the entire network structure"""

    def __init__(self):
        self.nodes: List[Node] = []
        self.tensors: Dict[str, TensorInfo] = {}
        self.initializers: List[str] = []
        self.model_inputs: List[str] = []
        self.model_outputs: List[str] = []
        self.dependency_analysis: Dict[str, Any] = {}
        self.execution_order: List[int] = []

    def add_node(self, node: Node):
        """Add a node to the graph"""
        self.nodes.append(node)

    def add_tensor(self, tensor: TensorInfo):
        """Add a tensor to the graph"""
        self.tensors[tensor.name] = tensor

    def get_node_by_index(self, index: int) -> Optional[Node]:
        """Get node by its index"""
        for node in self.nodes:
            if node.index == index:
                return node
        return None

    def build_execution_order(self):
        """Build execution order based on dependency analysis"""
        if not self.dependency_analysis:
            return

        # Sort nodes by execution level
        level_to_nodes = {}
        for node_idx, info in self.dependency_analysis.get(
            "node_dependencies", {}
        ).items():
            level = info.get("execution_level", 0)
            if level not in level_to_nodes:
                level_to_nodes[level] = []
            level_to_nodes[level].append(int(node_idx))

        # Flatten to execution order
        self.execution_order = []
        for level in sorted(level_to_nodes.keys()):
            self.execution_order.extend(level_to_nodes[level])


def create_node_from_json(node_data: Dict[str, Any]) -> Node:
    """Factory function to create appropriate node type from JSON data"""
    op_type = node_data.get("op_type", "")
    index = node_data.get("index", 0)
    name = node_data.get("name", "")
    inputs = node_data.get("inputs", [])
    outputs = node_data.get("outputs", [])
    attributes = node_data.get("attributes", {})

    if op_type == "QLinearConv":
        return ConvNode(index, name, op_type, inputs, outputs, attributes)
    elif op_type in ["QLinearAdd", "QLinearMul"]:
        # Check if this is a batch norm operation based on name
        if "batchnorm" in name.lower():
            return BNNode(index, name, op_type, inputs, outputs, attributes)
        else:
            return ElemWiseNode(index, name, op_type, inputs, outputs, attributes)
    elif op_type == "QGemm":
        return QGemmNode(index, name, op_type, inputs, outputs, attributes)
    elif op_type == "Reshape":
        return ReshapeNode(index, name, op_type, inputs, outputs, attributes)
    elif op_type in ["Softmax", "Tanh"]:
        return ActivationNode(index, name, op_type, inputs, outputs, attributes)
    else:
        # Default to base Node class for unknown types
        class UnknownNode(Node):
            def get_cpp_template_args(self) -> str:
                return ""

            def get_cpp_constructor_args(self) -> str:
                return ""

            def get_cpp_member_declaration(self, var_name: str) -> str:
                return f"// Unknown node type: {self.op_type}"

            def get_cpp_forward_call(
                self, var_name: str, input_vars: List[str], output_var: str = ""
            ) -> str:
                return f"// Unknown operation: {self.op_type}"

        return UnknownNode(index, name, op_type, inputs, outputs, attributes)


def parse_execution_plan(json_file_path: str) -> NetworkGraph:
    """Parse the execution plan JSON file and return a NetworkGraph"""
    with open(json_file_path, "r") as f:
        data = json.load(f)

    graph = NetworkGraph()

    # Parse model info
    model_info = data.get("model_info", {})
    graph.model_inputs = model_info.get("model_inputs", [])
    graph.model_outputs = model_info.get("model_outputs", [])

    # Parse initializers
    graph.initializers = data.get("initializers", [])

    # Parse nodes
    nodes_data = data.get("nodes", [])
    for node_data in nodes_data:
        if node_data:  # Skip empty node entries
            node = create_node_from_json(node_data)
            graph.add_node(node)

    # Parse connections to build tensor information
    connections = data.get("connections", [])
    for conn in connections:
        tensor_name = conn.get("tensor_name", "")
        if tensor_name:
            tensor = TensorInfo(name=tensor_name)
            graph.add_tensor(tensor)

    # Parse dependency analysis
    graph.dependency_analysis = data.get("dependency_analysis", {})

    # Build execution order
    graph.build_execution_order()

    return graph


def populate_tensor_values(graph: NetworkGraph, json_data: Dict[str, Any]):
    """Populate tensor scale and zero-point values from connections"""
    connections = json_data.get("connections", [])

    for conn in connections:
        tensor_name = conn.get("tensor_name", "")
        if not tensor_name:
            continue

        # Create or update tensor info
        if tensor_name not in graph.tensors:
            graph.tensors[tensor_name] = TensorInfo(name=tensor_name)

        tensor = graph.tensors[tensor_name]

        # Try to extract scale/zero_point from connection if available
        # This would need to be enhanced based on actual connection format
        if "_scale" in tensor_name:
            # This is a scale tensor - extract base tensor name
            base_name = tensor_name.replace("_scale", "")
            if base_name in graph.tensors:
                # Would populate actual scale value here if available in JSON
                pass
        elif "_zero_point" in tensor_name:
            # This is a zero point tensor - extract base tensor name
            base_name = tensor_name.replace("_zero_point", "")
            if base_name in graph.tensors:
                # Would populate actual zero point value here if available in JSON
                pass


def build_node_relationships(graph: NetworkGraph):
    """Build producer/consumer relationships between nodes"""
    # Create mapping of output tensor name to producing node
    output_to_node = {}
    for node in graph.nodes:
        for output_name in node.output_names:
            output_to_node[output_name] = node

    # For each node, find its producers based on input tensors
    for node in graph.nodes:
        for input_name in node.input_names:
            # Skip scale/zero_point tensors and initializers
            if (
                "_scale" in input_name
                or "_zero_point" in input_name
                or input_name in graph.initializers
            ):
                continue

            producer_node = output_to_node.get(input_name)
            if producer_node and producer_node != node:
                node.producer_nodes.append(producer_node)
                producer_node.consumer_nodes.append(node)


def infer_tensor_dimensions(graph: NetworkGraph):
    """Infer tensor dimensions where possible"""
    # Start with known input dimensions (typically 1x3x15x15 for board input)
    if graph.model_inputs:
        input_name = graph.model_inputs[0]
        if input_name in graph.tensors:
            graph.tensors[input_name].shape = [
                1,
                3,
                15,
                15,
            ]  # Batch, Channels, Height, Width

    # Propagate dimensions through the network based on node types
    for node_idx in graph.execution_order:
        node = graph.get_node_by_index(node_idx)
        if not node:
            continue

        # Infer output dimensions based on node type and input dimensions
        if isinstance(node, ConvNode):
            # For conv layers, output shape depends on filters, input shape, kernel, stride, padding
            # This would need more sophisticated dimension tracking
            pass
        elif isinstance(node, ReshapeNode):
            # Reshape operations would have target shape in attributes
            pass
        elif isinstance(node, QGemmNode):
            # QGemm output dimensions depend on weight matrix shape
            pass


def extract_parameter_info(graph: NetworkGraph) -> Dict[str, Any]:
    """Extract parameter information (weights, biases) needed for C++ generation"""
    params = {
        "conv_weights": [],
        "conv_biases": [],
        "bn_weights": [],
        "matmul_weights": [],
        "scales": {},
        "zero_points": {},
    }

    for node in graph.nodes:
        if isinstance(node, ConvNode):
            if hasattr(node, "weight_tensor_name"):
                params["conv_weights"].append(
                    {
                        "name": node.weight_tensor_name,
                        "cpp_name": node.weight_tensor_name.replace("/", "_").replace(
                            ":", "_"
                        ),
                        "node_name": node.name,
                    }
                )
            if hasattr(node, "bias_tensor_name"):
                params["conv_biases"].append(
                    {
                        "name": node.bias_tensor_name,
                        "cpp_name": node.bias_tensor_name.replace("/", "_").replace(
                            ":", "_"
                        ),
                        "node_name": node.name,
                    }
                )
        elif isinstance(node, BNNode):
            # Batch norm parameters
            if len(node.input_names) > 3:
                params["bn_weights"].append(
                    {
                        "name": node.input_names[3],
                        "cpp_name": node.input_names[3]
                        .replace("/", "_")
                        .replace(":", "_"),
                        "node_name": node.name,
                    }
                )
        elif isinstance(node, QGemmNode):
            if hasattr(node, "b_tensor_name"):
                params["matmul_weights"].append(
                    {
                        "name": node.b_tensor_name,
                        "cpp_name": node.b_tensor_name.replace("/", "_").replace(
                            ":", "_"
                        ),
                        "node_name": node.name,
                    }
                )

    # Extract scale and zero point information
    for tensor_name, tensor in graph.tensors.items():
        if "_scale" in tensor_name:
            base_name = tensor_name.replace("_scale", "")
            params["scales"][base_name] = tensor_name
        elif "_zero_point" in tensor_name:
            base_name = tensor_name.replace("_zero_point", "")
            params["zero_points"][base_name] = tensor_name

    return params


def parse_execution_plan_complete(json_file_path: str) -> NetworkGraph:
    """Complete parsing with all relationship building and inference"""
    # Load JSON data
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Parse basic structure
    graph = parse_execution_plan(json_file_path)

    # Populate tensor values from connections
    populate_tensor_values(graph, data)

    # Build node relationships
    build_node_relationships(graph)

    # Infer tensor dimensions where possible
    infer_tensor_dimensions(graph)

    return graph


# Test function to verify the parsing works correctly
def test_parsing():
    """Test function to verify the execution plan parsing"""
    try:
        # Test with the b model
        json_file = "C:/Users/StrangeMP/OneDrive/Desktop/15_by_15_AlphaGomoku-demo/quant/model_b_15_execution_plan.json"
        graph = parse_execution_plan_complete(json_file)

        print(f"Successfully parsed execution plan:")
        print(f"  Model inputs: {graph.model_inputs}")
        print(f"  Model outputs: {graph.model_outputs}")
        print(f"  Number of nodes: {len(graph.nodes)}")
        print(f"  Number of tensors: {len(graph.tensors)}")
        print(f"  Number of initializers: {len(graph.initializers)}")
        print(f"  Execution order length: {len(graph.execution_order)}")

        # Print node type distribution
        node_types = {}
        for node in graph.nodes:
            node_type = type(node).__name__
            node_types[node_type] = node_types.get(node_type, 0) + 1

        print(f"  Node type distribution:")
        for node_type, count in sorted(node_types.items()):
            print(f"    {node_type}: {count}")

        # Print first few nodes as examples
        print(f"\n  First 5 nodes:")
        for i, node in enumerate(graph.nodes[:5]):
            print(f"    [{i}] {node.name} ({type(node).__name__}) - {node.op_type}")

        # Test parameter extraction
        params = extract_parameter_info(graph)
        print(f"\n  Parameter extraction:")
        print(f"    Conv weights: {len(params['conv_weights'])}")
        print(f"    Conv biases: {len(params['conv_biases'])}")
        print(f"    BN weights: {len(params['bn_weights'])}")
        print(f"    MatMul weights: {len(params['matmul_weights'])}")
        print(f"    Scales: {len(params['scales'])}")
        print(f"    Zero points: {len(params['zero_points'])}")

        # Show a few examples of each parameter type
        if params["conv_weights"]:
            print(f"    Example conv weight: {params['conv_weights'][0]}")
        if params["matmul_weights"]:
            print(f"    Example matmul weight: {params['matmul_weights'][0]}")

        # Test C++ code generation for a few nodes
        print(f"\n  C++ code generation examples:")
        for i, node in enumerate(graph.nodes[:3]):
            var_name = f"node_{i}"
            print(f"    {node.name}:")
            print(f"      Member: {node.get_cpp_member_declaration(var_name)}")
            print(
                f"      Forward: {node.get_cpp_forward_call(var_name, ['input_tensor'])}"
            )

        return True

    except Exception as e:
        print(f"Error during parsing: {e}")
        import traceback

        traceback.print_exc()
        return False


# Test function to verify the parsing works correctly
def test_parsing_enhanced():
    """Enhanced test function to verify the execution plan parsing"""
    try:
        # Test with the b model
        json_file = "C:/Users/StrangeMP/OneDrive/Desktop/15_by_15_AlphaGomoku-demo/quant/model_b_15_execution_plan.json"
        graph = parse_execution_plan_complete(json_file)

        print(f"Successfully parsed execution plan:")
        print(f"  Model inputs: {graph.model_inputs}")
        print(f"  Model outputs: {graph.model_outputs}")
        print(f"  Number of nodes: {len(graph.nodes)}")
        print(f"  Number of tensors: {len(graph.tensors)}")
        print(f"  Number of initializers: {len(graph.initializers)}")
        print(f"  Execution order length: {len(graph.execution_order)}")

        # Print node type distribution
        node_types = {}
        for node in graph.nodes:
            node_type = type(node).__name__
            node_types[node_type] = node_types.get(node_type, 0) + 1

        print(f"  Node type distribution:")
        for node_type, count in sorted(node_types.items()):
            print(f"    {node_type}: {count}")

        # Print first few nodes as examples
        print(f"\n  First 5 nodes:")
        for i, node in enumerate(graph.nodes[:5]):
            print(f"    [{i}] {node.name} ({type(node).__name__}) - {node.op_type}")

        # Test parameter extraction
        params = extract_parameter_info(graph)
        print(f"\n  Parameter extraction:")
        print(f"    Conv weights: {len(params['conv_weights'])}")
        print(f"    Conv biases: {len(params['conv_biases'])}")
        print(f"    BN weights: {len(params['bn_weights'])}")
        print(f"    MatMul weights: {len(params['matmul_weights'])}")
        print(f"    Scales: {len(params['scales'])}")
        print(f"    Zero points: {len(params['zero_points'])}")

        # Show a few examples of each parameter type
        if params["conv_weights"]:
            print(f"    Example conv weight: {params['conv_weights'][0]}")
        if params["matmul_weights"]:
            print(f"    Example matmul weight: {params['matmul_weights'][0]}")

        # Test C++ code generation for a few nodes
        print(f"\n  C++ code generation examples:")
        for i, node in enumerate(graph.nodes[:3]):
            var_name = f"node_{i}"
            print(f"    {node.name}:")
            print(f"      Member: {node.get_cpp_member_declaration(var_name)}")
            print(
                f"      Forward: {node.get_cpp_forward_call(var_name, ['input_tensor'])}"
            )

        return True

    except Exception as e:
        print(f"Error during parsing: {e}")
        import traceback

        traceback.print_exc()
        return False


colors = ["w", "b"]
base_dir = "C:/Users/StrangeMP/OneDrive/Desktop/15_by_15_AlphaGomoku-demo/quant/"
execution_plan_file = "model_REPLACE_15_execution_plan.json"
out_file = "model_REPLACE_15_net.cpp"

# ===== C++ CODE GENERATION FUNCTIONS =====


def generate_cpp_header(graph: NetworkGraph) -> str:
    """Generate C++ header includes and namespace"""
    header = f"""
  #include "qnnet.hpp"
  #include "nnet.hpp"
  #include "model_{working_color}_15_params.h"
  #include <array>
  #include <cstdint>

  namespace AlphaGomoku {{
  namespace {'White' if working_color == 'w' else 'Black'} {{
  using namespace model_{working_color}_15_params;

  """
    return header


def generate_parameter_declarations(graph: NetworkGraph) -> str:
    """Generate C++ parameter declarations from initializers"""
    declarations = "// Parameter declarations\n"

    # param_info = extract_parameter_info(graph)

    # # Generate extern declarations for all parameters
    # for category, params in param_info.items():
    #     if params:
    #         declarations += f"\n// {category.replace('_', ' ').title()}\n"
    #         if isinstance(params, list):
    #             # Handle list of parameter dictionaries (conv_weights, conv_biases, etc.)
    #             for param in params:
    #                 if isinstance(param, dict) and 'name' in param:
    #                     cpp_var_name = Node.tensor_name_to_cpp_var(param['name'])
    #                     declarations += f"extern const auto {cpp_var_name};\n"
    #         elif isinstance(params, dict):
    #             # Handle dictionary of tensor names (scales, zero_points)
    #             for base_name, tensor_name in params.items():
    #                 cpp_var_name = Node.tensor_name_to_cpp_var(tensor_name)
    #                 declarations += f"extern const auto {cpp_var_name};\n"

    declarations += "\n"
    return declarations


def generate_network_class_declaration(graph: NetworkGraph) -> str:
    """Generate C++ network class with member declarations"""
    class_def = """struct QuantizedNetwork {
public:
    QuantizedNetwork();
    auto forward(const Tensor<uint8_t, 3, 15, 15>& input);
    static constexpr size_t BOARD_SIZE = 15;
    decltype(auto) output(){
      return  std::pair<Vec<SCALE_T, BOARD_SIZE * BOARD_SIZE>&, SCALE_T&>(pi, v);
    }
    
private:
    // Network layers
"""

    # Generate member declarations for each node
    for node in graph.nodes:
        if hasattr(node, "get_cpp_member_declaration"):
            var_name = node.get_sanitized_name()
            member_decl = node.get_cpp_member_declaration(var_name)
            if member_decl.strip() and not member_decl.startswith("//"):
                class_def += f"    {member_decl}\n"

    class_def += """
    Vec<SCALE_T, BOARD_SIZE * BOARD_SIZE> pi;
    SCALE_T v;
"""

    return class_def


def generate_constructor(graph: NetworkGraph) -> str:
    """Generate C++ constructor with node initializations"""
    constructor = """
QuantizedNetwork::QuantizedNetwork()
"""

    # Find nodes that need constructor arguments
    init_list = []
    for node in graph.nodes:
        if hasattr(node, "get_cpp_constructor_args"):
            var_name = node.get_sanitized_name()
            constructor_args = node.get_cpp_constructor_args()
            if constructor_args.strip():
                init_list.append(f"    {var_name}({constructor_args})")

    if init_list:
        constructor += "    : " + ",\n".join(init_list) + "\n"

    constructor += """{}

"""
    return constructor


def generate_forward_method(graph: NetworkGraph) -> str:
    """Generate C++ forward pass method with direct chaining approach"""
    forward_method = """auto QuantizedNetwork::forward(const Tensor<uint8_t, 3, 15, 15>& input) {
    // Direct chaining approach - no intermediate tensor tracking
    
"""

    def is_res_conn_node(node: Node) -> bool:
        """Check if node is a residual connection node"""
        return node.op_type == "QLinearAdd" and "batchnorm" not in node.name.lower()

    # Build tensor to node mapping
    T = {}
    for node in graph.nodes:
        for output_name in node.output_names:
            T[output_name] = node

    inputs = {}

    # Find Tanh and Softmax nodes (typically the final output nodes)
    tanh_softmax_nodes = []
    for node in graph.nodes:
        if node.op_type in ["Tanh", "Softmax"]:
            tanh_softmax_nodes.append(node)

    # Process each Tanh/Softmax node
    branches = {node.index: [] for node in tanh_softmax_nodes}
    for node in tanh_softmax_nodes:
        N = node
        S = branches[node.index]
        while not is_res_conn_node(N):
            S.append(N)
            # Move to previous node
            if N.input_names:
                prev_tensor = N.input_names[0]
                if prev_tensor in T:
                    N = T[prev_tensor]
                else:
                    break
            else:
                break

            if N.op_type in ["QLinearConv", "QGemm"]:
                for P in S:
                    if P.index not in inputs:
                        inputs[P.index] = []
                    inputs[P.index].append(N.get_sanitized_name() + ".output()")
                S.clear()
            elif N.op_type == "Reshape":
                for P in S:
                    if P.index not in inputs:
                        inputs[P.index] = []
                    inputs[P.index].append(N.index)
                S.clear()

    # Hard-coded map as specified in pseudocode
    map = {
        21: [22, 23, 24, 25, 26],
        18: [19, 20, 21],
        14: [15, 16, 17, 18],
        11: [12, 13, 14],
        7: [8, 9, 10, 11],
        4: [5, 6, 7],
        1: [2, 3, 4],
    }

    # Apply the map: arg.output() is the first argument of nodes
    for arg_idx, node_indices in map.items():
        arg_node = graph.get_node_by_index(arg_idx)
        if arg_node:
            arg_var = arg_node.get_sanitized_name()
            for node_idx in node_indices:
                node = graph.get_node_by_index(node_idx)
                if node:
                    if node.index not in inputs:
                        inputs[node.index] = []
                    inputs[node.index].append(arg_var + ".output()")

    # Special cases for residual connections
    # The second argument of node10 is node1.output()
    # The second argument of node17 is node7.output()
    # The second argument of node24 is node14.output()
    residual_connections = [(1, 10), (7, 17), (14, 24)]

    for source_idx, target_idx in residual_connections:
        source_node = graph.get_node_by_index(source_idx)
        target_node = graph.get_node_by_index(target_idx)
        if source_node and target_node:
            source_var = source_node.get_sanitized_name()
            if target_node.index not in inputs:
                inputs[target_node.index] = []
            inputs[target_node.index].append(source_var + ".output()")

    # Set first argument of node1 as 'input'
    node1 = graph.get_node_by_index(1)
    if node1:
        if node1.index not in inputs:
            inputs[node1.index] = []
        inputs[node1.index].insert(
            0, "input"
        )  # Insert at beginning to make it first argument    # ===== VERIFICATION SECTION =====
    print("=== Input Collection Verification ===")
    
    # Verify that collected inputs match the graph structure
    verification_errors = []
    verification_warnings = []
    verification_info = []
    
    # Skip verification for the first quantization node as it's expected to have no collected inputs
    skip_nodes = {0}  # Node 0 is typically the input quantization node
    
    for node in graph.nodes:
        node_idx = node.index
        collected_inputs = inputs.get(node_idx, [])
        expected_input_count = len(node.input_names)
        
        # Skip input quantization node
        if node_idx in skip_nodes:
            continue
            
        # Check if we have inputs for nodes that should have them
        if expected_input_count > 0 and not collected_inputs:
            verification_errors.append(f"Node {node_idx} ({node.name}) expects {expected_input_count} inputs but got 0 collected inputs")
        
        # Check input count mismatches (for tensor inputs only, not parameter inputs)
        if node.op_type in ["Tanh", "Softmax"]:
            # These should have exactly 1 tensor input
            if len(collected_inputs) != 1:
                verification_errors.append(f"Node {node_idx} ({node.name}, {node.op_type}) should have 1 tensor input but got {len(collected_inputs)}")
            else:
                verification_info.append(f"✓ Node {node_idx} ({node.op_type}) has correct 1 tensor input")
        elif node.op_type == "Reshape":
            # Reshape should have 1 tensor input (the shape is a parameter)
            if len(collected_inputs) != 1:
                verification_errors.append(f"Node {node_idx} ({node.name}, {node.op_type}) should have 1 tensor input but got {len(collected_inputs)}")
            else:
                verification_info.append(f"✓ Node {node_idx} ({node.op_type}) has correct 1 tensor input")
        elif node.op_type in ["QLinearAdd", "QLinearMul"]:
            # These should have 2 tensor inputs (plus scale/zero-point parameters)
            if "batchnorm" in node.name.lower():
                # Batch norm operations have 1 tensor input
                if len(collected_inputs) != 1:
                    verification_warnings.append(f"BatchNorm node {node_idx} ({node.name}) should have 1 tensor input but got {len(collected_inputs)}")
                else:
                    verification_info.append(f"✓ Node {node_idx} (BatchNorm {node.op_type}) has correct 1 tensor input")
            else:
                # Regular element-wise ops have 2 tensor inputs (e.g., residual connections)
                if len(collected_inputs) != 2:
                    verification_warnings.append(f"Element-wise node {node_idx} ({node.name}, {node.op_type}) should have 2 tensor inputs but got {len(collected_inputs)}")
                else:
                    verification_info.append(f"✓ Node {node_idx} (Residual {node.op_type}) has correct 2 tensor inputs")
        elif node.op_type == "QLinearConv":
            # Conv should have 1 tensor input (plus weight/bias parameters)
            if len(collected_inputs) != 1:
                verification_errors.append(f"Conv node {node_idx} ({node.name}) should have 1 tensor input but got {len(collected_inputs)}")
            else:
                verification_info.append(f"✓ Node {node_idx} ({node.op_type}) has correct 1 tensor input")
        elif node.op_type == "QGemm":
            # QGemm should have 1 tensor input (plus weight/bias parameters)  
            if len(collected_inputs) != 1:
                verification_errors.append(f"QGemm node {node_idx} ({node.name}) should have 1 tensor input but got {len(collected_inputs)}")
            else:
                verification_info.append(f"✓ Node {node_idx} ({node.op_type}) has correct 1 tensor input")
        elif node.op_type in ["DequantizeLinear"]:
            # Dequantize nodes should have 1 tensor input
            if len(collected_inputs) != 1:
                verification_warnings.append(f"Dequantize node {node_idx} ({node.name}) should have 1 tensor input but got {len(collected_inputs)}")
            else:
                verification_info.append(f"✓ Node {node_idx} ({node.op_type}) has correct 1 tensor input")
    
    # Check for nodes that have collected inputs but shouldn't
    for node_idx, collected_inputs in inputs.items():
        node = graph.get_node_by_index(node_idx)
        if node is None:
            verification_errors.append(f"Collected inputs for non-existent node {node_idx}")
            continue
            
        if not node.input_names and collected_inputs:
            verification_warnings.append(f"Node {node_idx} ({node.name}) has no expected inputs but got {len(collected_inputs)} collected inputs")
    
    # Analyze the hard-coded mapping (this represents skip connections in the residual network)
    print(f"\n--- Hard-coded Mapping Analysis (Skip Connections) ---")
    skip_connection_count = 0
    for arg_idx, target_indices in map.items():
        arg_node = graph.get_node_by_index(arg_idx)
        if not arg_node:
            verification_errors.append(f"Hard-coded mapping references non-existent source node {arg_idx}")
            continue
            
        verification_info.append(f"Skip connection from {arg_idx} ({arg_node.name}) to {len(target_indices)} targets")
        skip_connection_count += len(target_indices)
        
        for target_idx in target_indices:
            target_node = graph.get_node_by_index(target_idx)
            if not target_node:
                verification_errors.append(f"Hard-coded mapping references non-existent target node {target_idx}")
                continue
    
    verification_info.append(f"Total skip connections: {skip_connection_count}")
    
    # Check residual connections
    print(f"\n--- Residual Connection Analysis ---")
    residual_connection_count = 0
    for source_idx, target_idx in residual_connections:
        source_node = graph.get_node_by_index(source_idx)
        target_node = graph.get_node_by_index(target_idx)
        
        if not source_node:
            verification_errors.append(f"Residual connection references non-existent source node {source_idx}")
            continue
        if not target_node:
            verification_errors.append(f"Residual connection references non-existent target node {target_idx}")
            continue
            
        # For residual connections, we expect that target_node is a QLinearAdd
        if target_node.op_type != "QLinearAdd":
            verification_warnings.append(f"Residual connection target {target_idx} ({target_node.name}) is {target_node.op_type}, expected QLinearAdd")
        else:
            verification_info.append(f"✓ Residual connection: {source_idx} ({source_node.name}) -> {target_idx} ({target_node.name})")
            residual_connection_count += 1
    
    verification_info.append(f"Total residual connections: {residual_connection_count}")
    
    # Print verification results
    if verification_errors:
        print(f"\n❌ VERIFICATION ERRORS ({len(verification_errors)}):")
        for error in verification_errors:
            print(f"  - {error}")
    
    if verification_warnings:
        print(f"\n⚠️  VERIFICATION WARNINGS ({len(verification_warnings)}):")
        for warning in verification_warnings:
            print(f"  - {warning}")
    
    # Print informational messages (only first 10 to avoid spam)
    if verification_info:
        print(f"\n✅ VERIFICATION SUCCESS INFO (showing first 10 of {len(verification_info)}):")
        for info in verification_info[:10]:
            print(f"  - {info}")
        if len(verification_info) > 10:
            print(f"  ... and {len(verification_info) - 10} more")
    
    # Final status
    if not verification_errors:
        print(f"\n🎉 OVERALL STATUS: PASSED - No critical errors found!")
        if verification_warnings:
            print(f"   (With {len(verification_warnings)} warnings to review)")
    else:
        print(f"\n💥 OVERALL STATUS: FAILED - {len(verification_errors)} errors need fixing")
    
    # Print summary of collected inputs
    print(f"\n--- Input Collection Summary ---")
    print(f"Total nodes in graph: {len(graph.nodes)}")
    print(f"Nodes with collected inputs: {len(inputs)}")
    print(f"Nodes in execution order: {len(graph.execution_order)}")
    print(f"Skipped nodes (input quantization): {len(skip_nodes)}")
      # Only show problematic mappings for debugging
    print(f"\n--- Problematic Input Mappings (if any) ---")
    problematic_count = 0
    for node_idx in sorted(inputs.keys()):
        node = graph.get_node_by_index(node_idx)
        if node and node_idx not in skip_nodes:
            collected_inputs = inputs[node_idx]
            expected_tensor_inputs = 1  # Most nodes expect 1 tensor input
            # Adjust expected count for special node types
            if node.op_type in ["QLinearAdd", "QLinearMul"] and "batchnorm" not in node.name.lower():
                expected_tensor_inputs = 2  # Residual connections
                
            if len(collected_inputs) != expected_tensor_inputs:
                print(f"❗ Node {node_idx} ({node.name}, {node.op_type}):")
                print(f"   Expected: {expected_tensor_inputs} tensor inputs")
                print(f"   Collected: {len(collected_inputs)} inputs: {collected_inputs}")
                problematic_count += 1
    
    if problematic_count == 0:
        print("   None found - all input mappings look good!")
    
    # Final summary analysis
    print(f"\n--- Final Analysis Summary ---")
    total_nodes = len(graph.nodes)
    nodes_with_inputs = len(inputs)
    skipped_nodes_count = len(skip_nodes)
    
    print(f"📊 Network Statistics:")
    print(f"   Total nodes: {total_nodes}")
    print(f"   Nodes with collected inputs: {nodes_with_inputs}")
    print(f"   Skipped nodes: {skipped_nodes_count}")
    print(f"   Residual connections: {residual_connection_count}")
    print(f"   Skip connections: {skip_connection_count}")
    
    # Count node types with correct inputs
    correct_conv = sum(1 for node in graph.nodes if node.op_type == "QLinearConv" and len(inputs.get(node.index, [])) == 1)
    total_conv = sum(1 for node in graph.nodes if node.op_type == "QLinearConv")
    
    correct_qgemm = sum(1 for node in graph.nodes if node.op_type == "QGemm" and len(inputs.get(node.index, [])) == 1)
    total_qgemm = sum(1 for node in graph.nodes if node.op_type == "QGemm")
    
    correct_bn = sum(1 for node in graph.nodes if node.op_type in ["QLinearMul", "QLinearAdd"] and "batchnorm" in node.name.lower() and len(inputs.get(node.index, [])) == 1)
    total_bn = sum(1 for node in graph.nodes if node.op_type in ["QLinearMul", "QLinearAdd"] and "batchnorm" in node.name.lower())
    
    correct_residual = sum(1 for node in graph.nodes if node.op_type in ["QLinearAdd"] and "batchnorm" not in node.name.lower() and len(inputs.get(node.index, [])) == 2)
    total_residual = sum(1 for node in graph.nodes if node.op_type in ["QLinearAdd"] and "batchnorm" not in node.name.lower())
    
    print(f"✅ Node Type Validation:")
    print(f"   Conv layers: {correct_conv}/{total_conv} correct")
    print(f"   QGemm layers: {correct_qgemm}/{total_qgemm} correct") 
    print(f"   BatchNorm layers: {correct_bn}/{total_bn} correct")
    print(f"   Residual connections: {correct_residual}/{total_residual} correct")
    
    # Verification of the input collection algorithm
    print(f"\n🔍 Algorithm Verification:")
    print(f"   The input collection algorithm successfully:")
    print(f"   ✓ Identified {len(tanh_softmax_nodes)} output nodes (Tanh/Softmax)")
    print(f"   ✓ Applied {len(map)} hard-coded skip connection mappings")
    print(f"   ✓ Configured {len(residual_connections)} residual connections")
    print(f"   ✓ Set input for the first node (node 1)")
    print(f"   ✓ Handled {sum(1 for inputs_list in inputs.values() for inp in inputs_list if isinstance(inp, int))} reshape operations")
    
    print("=== End Verification ===\n")
    # ===== END VERIFICATION SECTION =====

    # Generate forward calls for each node in execution order
    for node_idx in graph.execution_order:
        node = graph.get_node_by_index(node_idx)
        if not node:
            continue

        var_name = node.get_sanitized_name()
        input_vars = inputs.get(node.index, [])

        for i in range(len(input_vars)):
            if not isinstance(input_vars[i], str):  # this is from reshpae
                reshape_index = input_vars[i]
                real_arg = inputs[reshape_index][0]
                input_vars[i] = real_arg + '.flatten()'

        # Generate forward call
        forward_call = node.get_cpp_forward_call(var_name, input_vars)
        forward_method += f"    {forward_call}\n"

    # Find the final output node (usually Softmax or the last node)
    final_output_node = None
    for node in reversed(graph.nodes):
        if node.op_type in ["Softmax", "Tanh"] or node.index == max(
            n.index for n in graph.nodes
        ):
            final_output_node = node
            break

    if final_output_node:
        final_var = final_output_node.get_sanitized_name()
        forward_method += f"""
    return output();
}}

"""
    else:
        forward_method += f"""
    // Return final output (fallback)
    return /* final_output */;
}}

"""
    return forward_method


def generate_cpp_footer() -> str:
    """Generate C++ namespace closing"""
    return f"""}} // namespace {'White' if working_color == 'w' else 'Black'}
  }} // namespace AlphaGomoku
  """


def generate_complete_cpp_network(graph: NetworkGraph, output_file_path: str):
    """Generate complete C++ network implementation"""
    cpp_code = ""

    # Generate all sections
    cpp_code += generate_cpp_header(graph)
    cpp_code += generate_parameter_declarations(graph)
    cpp_code += generate_network_class_declaration(graph)
    cpp_code += "};\n\n"  # Close class declaration
    cpp_code += generate_constructor(graph)
    cpp_code += generate_forward_method(graph)
    cpp_code += generate_cpp_footer()

    # Write to file
    with open(output_file_path, "w") as f:
        f.write(cpp_code)

    print(f"Generated C++ network implementation: {output_file_path}")
    return cpp_code


if __name__ == "__main__":
    test_parsing_enhanced()

    for c in ['b', 'w']:
      working_color = c
      # Run the main generation
      graph = parse_execution_plan_complete(
          f"C:/Users/StrangeMP/OneDrive/Desktop/15_by_15_AlphaGomoku-demo/quant/model_{c}_15_execution_plan.json"
      )
      generate_complete_cpp_network(        graph,
          f"C:/Users/StrangeMP/OneDrive/Desktop/15_by_15_AlphaGomoku-demo/quant/generated_network_{c}.cpp"
      )
