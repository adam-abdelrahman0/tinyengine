import numpy as np

from code_generator.operators import sigmoid
from code_generator.tflite import Model

from .utils import get_input_tensors, get_output_tensors, getOpCodeStr, getTensorTypeStr


def parse_logistic(op, model: Model.Model):
    op_code_str = getOpCodeStr(op, model)

    input_tensors = get_input_tensors(op, model)
    assert len(input_tensors) == 1, "input tensors length should be 1"
    input_tensor = input_tensors[0]

    output_tensors = get_output_tensors(op, model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]

    input_type = getTensorTypeStr(input_tensor.tensor.Type())
    output_type = getTensorTypeStr(output_tensor.tensor.Type())
    assert input_type == output_type, "tensor type not consistent"

    size = int(np.prod(input_tensor.tensor.ShapeAsNumpy()))

    params = {
        "op": op_code_str,
        "input_idx": input_tensor.tensor_idx,
        "output_idx": output_tensor.tensor_idx,
        "input_size": size,
        "input_dtype": input_type,
        "output_dtype": output_type,
    }

    if input_type != "float32":
        params["input_zero_point"] = input_tensor.qnn_params["zero_point"]
        params["input_scale"] = input_tensor.qnn_params["scale"]
        params["output_zero_point"] = output_tensor.qnn_params["zero_point"]
        params["output_scale"] = output_tensor.qnn_params["scale"]

    return sigmoid.sigmoid(params)
