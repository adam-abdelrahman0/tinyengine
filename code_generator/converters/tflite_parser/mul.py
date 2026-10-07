import numpy as np

from code_generator.operators import mul
from code_generator.tflite import Model
from code_generator.tflite.ActivationFunctionType import ActivationFunctionType
from code_generator.tflite.BuiltinOptions import BuiltinOptions
from code_generator.tflite.MulOptions import MulOptions

from .utils import get_input_tensors, get_output_tensors, getOpCodeStr, getTensorTypeStr

# Two shapes reach here: the plain, same-shape elementwise MUL StarBlock needs
# (act(f1) * f2, both full HxWxC tensors), and a broadcast MUL where one
# operand is a per-channel vector reused across every spatial position (e.g.
# a squeeze-excite gate's sigmoid output, [1,C], scaling the [1,H,W,C]
# feature map it was pooled from). The hard-sigmoid-style SE gate (ADD->Clip->
# MUL->MUL) is still intercepted earlier by SEelement.py's 3-op fusion and
# never reaches here -- this is for a real-Sigmoid gate's standalone MUL.


def parse_mul(op, model: Model.Model):
    op_code_str = getOpCodeStr(op, model)

    input_tensors = get_input_tensors(op, model)
    assert len(input_tensors) == 2, "MUL should have exactly 2 input tensors"
    input_tensor, input2_tensor = input_tensors

    output_tensors = get_output_tensors(op, model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]

    assert op.BuiltinOptionsType() == BuiltinOptions.MulOptions
    op_options = op.BuiltinOptions()
    mul_options = MulOptions()
    mul_options.Init(op_options.Bytes, op_options.Pos)
    assert mul_options.FusedActivationFunction() == ActivationFunctionType.NONE, \
        "fused activation on MUL isn't supported by mul_fpreq yet"

    input_shape = input_tensor.tensor.ShapeAsNumpy()
    input2_shape = input2_tensor.tensor.ShapeAsNumpy()
    output_shape = output_tensor.tensor.ShapeAsNumpy()
    input_size = int(np.prod(input_shape))
    input2_size = int(np.prod(input2_shape))
    output_size = int(np.prod(output_shape))
    assert output_size == max(input_size, input2_size), \
        "MUL output size must match the larger operand (plain elementwise, or one operand broadcast)"
    if input_size != input2_size:
        # broadcast case: the smaller operand must evenly divide the larger
        # one so it tiles across spatial positions with no remainder.
        assert max(input_size, input2_size) % min(input_size, input2_size) == 0, \
            "broadcast MUL requires the smaller operand's size to evenly divide the larger one"

    input_type = getTensorTypeStr(input_tensor.tensor.Type())
    input2_type = getTensorTypeStr(input2_tensor.tensor.Type())
    output_type = getTensorTypeStr(output_tensor.tensor.Type())
    assert input_type == input2_type == output_type, "tensor type not consistent"

    params = {
        "op": op_code_str,
        "input_idx": input_tensor.tensor_idx,
        "input2_idx": input2_tensor.tensor_idx,
        "output_idx": output_tensor.tensor_idx,
        "input_size": input_size,
        "input2_size": input2_size,
        "output_size": output_size,
        "input_dtype": input_type,
        "input2_dtype": input2_type,
        "output_dtype": output_type,
    }

    if input_type != "float32":
        params["input_zero_point"] = input_tensor.qnn_params["zero_point"]
        params["input_scale"] = input_tensor.qnn_params["scale"]
        params["input2_zero_point"] = input2_tensor.qnn_params["zero_point"]
        params["input2_scale"] = input2_tensor.qnn_params["scale"]
        params["output_zero_point"] = output_tensor.qnn_params["zero_point"]
        params["output_scale"] = output_tensor.qnn_params["scale"]

    return mul.mul(params)
