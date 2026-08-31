import numpy as np

from code_generator.operators import star_forward
from code_generator.tflite import Model
from code_generator.tflite.ActivationFunctionType import ActivationFunctionType
from code_generator.tflite.BuiltinOptions import BuiltinOptions
from code_generator.tflite.Conv2DOptions import Conv2DOptions

from .utils import get_input_tensors, get_np_from_wrapper, get_output_tensors, getMultiplierShift, getOpCodeStr, getTensorTypeStr

# StarBlock's forward: act(f1(x)) * f2(x), both f1/f2 1x1 convs sharing the
# same input, f1 has a fused RELU6, f2 has none, feeding a same-shape MUL.
# See static_layers.py: StaticStarBlock.forward(), lines 30-31.
#
#         +-- CONV_2D (1x1, RELU6) --+
#   x ----+                          +-- MUL --> act(f1(x)) * f2(x)
#         +-- CONV_2D (1x1, NONE)  --+
#
# Detected in TfliteConvertor.parseOperatorInfo() via checkIfRequireStarForward(),
# same pattern as checkIfRequireSEelementmult(). See docs/star_forward_notes.md.


def _parse_conv_branch(conv_op, model):
    """Pull weight/bias/per-channel quant params + shape out of a 1x1
    CONV_2D node -- same fields conv2d.py's parser extracts. This branch
    gets folded into the fused star_forward kernel instead of becoming its
    own Conv2d operator, so we read it manually rather than reusing
    parse_conv2d (which would append a standalone Conv2d op)."""
    assert getOpCodeStr(conv_op, model) == "CONV_2D"
    input_tensors = get_input_tensors(conv_op, model)
    assert len(input_tensors) >= 2, "conv should have >= 2 input tensors (input, weight[, bias])"
    input_tensor = input_tensors[0]
    weight_tensor = input_tensors[1]

    output_tensors = get_output_tensors(conv_op, model)
    assert len(output_tensors) == 1
    output_tensor = output_tensors[0]

    assert conv_op.BuiltinOptionsType() == BuiltinOptions.Conv2DOptions
    op_options = conv_op.BuiltinOptions()
    conv_options = Conv2DOptions()
    conv_options.Init(op_options.Bytes, op_options.Pos)
    assert conv_options.StrideH() == 1 and conv_options.StrideW() == 1, \
        "star_forward only supports StarBlock's stride-1 f1/f2 convs"

    output_c, kernel_h, kernel_w, _ = weight_tensor.tensor.ShapeAsNumpy()
    assert kernel_h == 1 and kernel_w == 1, "star_forward only fuses 1x1 pointwise convs"

    _, input_h, input_w, input_c = input_tensor.tensor.ShapeAsNumpy()
    _, output_h, output_w, output_c_dual = output_tensor.tensor.ShapeAsNumpy()
    assert output_c_dual == output_c
    assert output_h == input_h and output_w == input_w, \
        "1x1 stride-1 conv should preserve spatial dims"

    weight_value = get_np_from_wrapper(weight_tensor)
    bias = get_np_from_wrapper(input_tensors[2]) if len(input_tensors) == 3 else None

    input_scale = input_tensor.qnn_params["scale"]
    weight_scale = weight_tensor.qnn_params["scale"]
    output_scale = output_tensor.qnn_params["scale"]
    effective_scale = np.double(input_scale) * np.double(weight_scale) / np.double(output_scale)
    multiplier, shift = getMultiplierShift(effective_scale)

    return {
        "input_idx": input_tensor.tensor_idx,
        "output_idx": output_tensor.tensor_idx,
        "input_h": int(input_h),
        "input_w": int(input_w),
        "input_c": int(input_c),
        "output_c": int(output_c),
        "weight_value": weight_value,
        "bias": bias,
        "effective_scale": effective_scale,
        "multiplier": multiplier,
        "shift": shift,
        "input_zero_point": input_tensor.qnn_params["zero_point"],
        "output_zero_point": output_tensor.qnn_params["zero_point"],
        "input_scale": input_scale,
        "weight_scale": weight_scale,
        "output_scale": output_scale,
        "activation": conv_options.FusedActivationFunction(),
    }


def parse_star_forward(three_op_sequence, model: Model.Model, layer):
    conv_a_op, conv_b_op, mul_op = three_op_sequence

    branch_a = _parse_conv_branch(conv_a_op, model)
    branch_b = _parse_conv_branch(conv_b_op, model)

    assert branch_a["input_idx"] == branch_b["input_idx"], \
        "star_forward's two conv branches must share the same input tensor"
    assert branch_a["output_c"] == branch_b["output_c"], \
        "star_forward's two conv branches must produce the same channel count"

    # Exactly one branch should carry the fused RELU6 (StarBlock's act()).
    # Correctness of the fused kernel doesn't actually depend on which one --
    # each branch's own exported output scale/zero-point already encodes
    # its correct clamp range (see docs/star_forward_notes.md, "Why no
    # explicit RELU6 clamp") -- but we assert the shape as a specificity
    # check so an unrelated CONV/CONV/MUL sequence never gets misfused.
    activations = (branch_a["activation"], branch_b["activation"])
    assert activations.count(ActivationFunctionType.RELU6) == 1, \
        "star_forward expects exactly one branch with fused RELU6 (StarBlock's act(f1)*f2)"
    assert activations.count(ActivationFunctionType.NONE) == 1, \
        "star_forward expects exactly one branch with no fused activation"

    if branch_a["activation"] == ActivationFunctionType.RELU6:
        f1, f2 = branch_a, branch_b
    else:
        f1, f2 = branch_b, branch_a

    # The MUL node's own tensors carry x1/x2/output's *own* exported
    # per-tensor scale/zero-point -- read directly off the graph, same as
    # mul.py's parser does, rather than assumed/derived.
    assert getOpCodeStr(mul_op, model) == "MUL"
    mul_inputs = get_input_tensors(mul_op, model)
    assert len(mul_inputs) == 2, "MUL should have exactly 2 input tensors"
    mul_in1, mul_in2 = mul_inputs
    mul_outputs = get_output_tensors(mul_op, model)
    assert len(mul_outputs) == 1
    mul_out = mul_outputs[0]

    # match MUL's inputs back to f1/f2 by tensor idx (order isn't guaranteed
    # to follow f1/f2's graph declaration order)
    if mul_in1.tensor_idx == f1["output_idx"]:
        assert mul_in2.tensor_idx == f2["output_idx"]
    else:
        assert mul_in2.tensor_idx == f1["output_idx"] and mul_in1.tensor_idx == f2["output_idx"], \
            "MUL's inputs don't match star_forward's two conv branch outputs"

    input_type = getTensorTypeStr(mul_out.tensor.Type())
    assert input_type == "int8", "star_forward only supports the int8 (fp_requantize) path"

    params = {
        "op": "STAR_FORWARD",
        "input_idx": f1["input_idx"],
        "output_idx": mul_out.tensor_idx,
        "input_h": f1["input_h"],
        "input_w": f1["input_w"],
        "input_c": f1["input_c"],
        "output_h": f1["input_h"],
        "output_w": f1["input_w"],
        "mid_c": f1["output_c"],
        # shared conv input quant (both branches read the same x)
        "input_zero_point": f1["input_zero_point"],
        "input_scale": f1["input_scale"],
        # f1 (RELU6 branch)
        "f1_weight_value": f1["weight_value"],
        "f1_bias": f1["bias"],
        "f1_effective_scale": f1["effective_scale"],
        "f1_multiplier": f1["multiplier"],
        "f1_shift": f1["shift"],
        "f1_output_zero_point": f1["output_zero_point"],
        "f1_output_scale": f1["output_scale"],
        # f2 (no activation)
        "f2_weight_value": f2["weight_value"],
        "f2_bias": f2["bias"],
        "f2_effective_scale": f2["effective_scale"],
        "f2_multiplier": f2["multiplier"],
        "f2_shift": f2["shift"],
        "f2_output_zero_point": f2["output_zero_point"],
        "f2_output_scale": f2["output_scale"],
        # final elementwise combine (the fused MUL node's own tensors)
        "output_zero_point": mul_out.qnn_params["zero_point"],
        "output_scale": mul_out.qnn_params["scale"],
    }

    return star_forward.StarForward(params)
