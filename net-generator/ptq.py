import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tensorflow as tf
import numpy as np
import glob, random, tf2onnx
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat
import onnxruntime as ort

base_path = "./"
model_path = base_path + ""

model_weight_filenames = {
    "b": "model_b_15.h5",
    "w": "model_w_15.h5",
}


class representative_data_gen:
    def __init__(self, tensor):
        self.tensor = tensor

    def __call__(self):
        for input_tensor in self.tensor:
            yield [input_tensor]


def load_representative_datasets(folder_path, num_samples=500):
    representative_datasets = {"b": None, "w": None}

    for color in ["b", "w"]:
        # Find all .npy files for the current color
        pattern = os.path.join(folder_path, f"*_{color}*.npy")
        npy_files = glob.glob(pattern)

        if not npy_files:
            print(f"No .npy files found for color {color} in {folder_path}")
            continue

        # Load and combine data from all files
        all_data = []
        for npy_file in npy_files:
            try:
                data = np.load(npy_file).astype(np.float32)
                all_data.append(data)
            except Exception as e:
                print(f"Error loading {npy_file}: {e}")
                continue

        if not all_data:
            print(f"No valid data loaded for color {color}")
            continue

        # Combine all data
        combined_data = np.concatenate(all_data)

        # Randomly sample from the combined data
        if len(combined_data) > num_samples:
            indices = random.sample(range(len(combined_data)), num_samples)
            sampled_data = combined_data[indices]
        else:
            print(
                f"Warning: Only {len(combined_data)} samples available for {color}, using all"
            )
            sampled_data = combined_data

        representative_datasets[color] = sampled_data

    return representative_datasets


def build_gomoku_network(board_size, l2_coef, lr, momentum):
    """Builds and returns the Gomoku policy-value network (Keras Model)."""
    from tensorflow import keras

    layers = keras.layers
    models = keras.models
    regularizers = keras.regularizers
    optimizers = keras.optimizers
    Input = layers.Input
    add = layers.add
    Conv2D = layers.Conv2D
    Activation = layers.Activation
    Dense = layers.Dense
    Flatten = layers.Flatten
    BatchNormalization = layers.BatchNormalization
    Model = models.Model
    l2 = regularizers.l2
    SGD = optimizers.SGD

    # Input_Layer
    init_x = Input(
        (3, board_size, board_size), name="input"
    )  # the input is a tensor with the shape 3*(15*15)
    x = init_x

    # First Convolutional Layer with 32 filters
    x = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        data_format="channels_first",
        kernel_regularizer=l2(l2_coef),
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Three Residual Blocks
    def residual_block(x):
        x_shortcut = x
        x = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            data_format="channels_first",
            kernel_regularizer=l2(l2_coef),
        )(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            data_format="channels_first",
            kernel_regularizer=l2(l2_coef),
        )(x)
        x = BatchNormalization()(x)
        x = add([x, x_shortcut])  # Skip Connection
        x = Activation("relu")(x)
        return x

    for _ in range(3):
        x = residual_block(x)

    # Policy Head for generating prior probability vector for each action
    policy = Conv2D(
        filters=2,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        data_format="channels_first",
        kernel_regularizer=l2(l2_coef),
    )(x)
    policy = BatchNormalization()(policy)
    policy = Activation("relu")(policy)
    policy = Flatten()(policy)
    policy = Dense(board_size * board_size, kernel_regularizer=l2(l2_coef))(policy)
    policy = Activation("softmax")(policy)

    # Value Head for generating value of each action
    value = Conv2D(
        filters=1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        data_format="channels_first",
        kernel_regularizer=l2(l2_coef),
    )(x)
    value = BatchNormalization()(value)
    value = Activation("relu")(value)
    value = Flatten()(value)
    value = Dense(32, kernel_regularizer=l2(l2_coef))(value)
    value = Activation("relu")(value)
    value = Dense(1, kernel_regularizer=l2(l2_coef))(value)
    value = Activation("tanh")(value)

    # Define Network
    model = Model(inputs=init_x, outputs=[policy, value])

    # Define the Loss Function
    opt = SGD(
        momentum=momentum, nesterov=True
    )  # stochastic gradient descend with momentum
    losses_type = [
        "categorical_crossentropy",
        "mean_squared_error",
    ]  # cross-entrophy and MSE are weighted equally
    model.compile(optimizer=opt, loss=losses_type)

    return model


class CalibrationDataReader(ort.quantization.CalibrationDataReader):
    def __init__(self, calibration_samples):
        self.samples = calibration_samples
        self.count = 0

    def get_next(self):
        if self.count < len(self.samples):
            sample = self.samples[self.count]
            # Ensure sample is 4D: (1, 3, 15, 15)
            if sample.ndim == 3:
                sample = sample[np.newaxis, ...]
            input_dict = {"input": sample}  # Use the correct input name!
            self.count += 1
            return input_dict
        else:
            return None


# Load representative datasets
data_folder = os.path.join(base_path, "quant/rep_dataset")
representative_datasets = load_representative_datasets(data_folder)

for color in ["b", "w"]:
    model = build_gomoku_network(15, 1e-4, 2e-3, 9e-1)
    # print(model.inputs[0].name)
    model.load_weights(model_path + model_weight_filenames[color])
    tensors = representative_datasets[color]

    # # Export to ONNX
    # print(f"Exporting {color} model to ONNX...")
    onnx_model_path = os.path.join(model_path, f"model_{color}_15.onnx")
    onnx_quant_model_path = os.path.join(model_path, f"model_{color}_15_quant.onnx")
    # spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input"),)
    # model_proto, _ = tf2onnx.convert.from_keras(
    #     model, input_signature=spec, output_path=onnx_model_path
    # )
    # print(f"{color} model exported to {onnx_model_path}")

    print(f"Quantizing {color} model...")
    calib_reader = CalibrationDataReader(tensors)
    quantize_static(
        model_input=onnx_model_path,
        model_output=onnx_quant_model_path,
        calibration_data_reader=calib_reader,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        quant_format=QuantFormat.QOperator,
        per_channel=False,
    )

    print(f"{color} model quantized to {onnx_quant_model_path}")

