# ----------------------------------------------------------------------
# Project: TinyEngine
# Title:   PatchBasedUtil.py
#
# Reference papers:
#  - MCUNet: Tiny Deep Learning on IoT Device, NeurIPS 2020
#  - MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021
#  - MCUNetV3: On-Device Training Under 256KB Memory, NeurIPS 2022
# Contact authors:
#  - Wei-Ming Chen, wmchen@mit.edu
#  - Wei-Chen Wang, wweichen@mit.edu
#  - Ji Lin, jilin@mit.edu
#  - Ligeng Zhu, ligeng@mit.edu
#  - Song Han, songhan@mit.edu
#
# Target ISA:  ARMv7E-M
# ----------------------------------------------------------------------

def _effective_input_hwc(layers, idx):
    """H/W/C of the tensor flowing INTO layers[idx]. Conv/depthwise/pool/add
    ops all carry this directly as input_h/input_w/input_c. StarBlockV's
    self-gate MUL (act(conv(x))^2) doesn't -- mul.py's default_params has no
    input_h/input_w/input_c at all, it only tracks a flat element count,
    since the op's own math never needs the 2-D shape. But MUL here is
    shape-preserving (elementwise square), so the tensor's real H/W/C is
    whatever the immediately preceding op produced as output. Confirmed via
    a real crash: split_idx=5 and 6 both land exactly on a self-gate MUL in
    best_qat_mcu_model, and getPatchParams() assuming every layer has
    input_h/input_w raised KeyError('input_h')."""
    info = layers[idx].get_layer_info()
    if info.get("input_h") is not None:
        return info["input_h"], info["input_w"], info.get("input_c")
    for i in range(idx - 1, -1, -1):
        prev = layers[i].get_layer_info()
        if prev.get("output_h") is not None:
            return prev["output_h"], prev["output_w"], prev.get("output_c")
    raise ValueError(f"could not resolve H/W/C feeding into layer {idx}")


def getPatchParams(layers, split_idx, n_patch):
    patch_params = {}

    feat_stride = 8
    patch_params["n_patch"] = n_patch
    patch_params["layer_cnt"] = split_idx

    resolution = max(layers[0].get_layer_info()["input_h"], layers[0].get_layer_info()["input_w"])
    layer_cnt = layers[patch_params["layer_cnt"]].get_layer_info()
    in_h, in_w, in_c = _effective_input_hwc(layers, patch_params["layer_cnt"])
    out_shape = max(in_h, in_w)
    feat_stride = resolution // out_shape
    grain_size = out_shape // n_patch

    patch_params["single_rf"] = compute_receptive_field(layers, patch_params["layer_cnt"], 1)
    patch_params["output_c"] = in_c
    patch_params["output_h"] = layer_cnt.get("output_h", in_h)
    patch_params["output_w"] = layer_cnt.get("output_w", in_w)
    patch_params["grain_rf"] = compute_receptive_field(layers, patch_params["layer_cnt"], grain_size)
    patch_params["grain_rf_height"] = compute_receptive_field(
        layers, patch_params["layer_cnt"], in_h // n_patch
    )
    print("receptive field: single {} all {}".format(patch_params["single_rf"], patch_params["grain_rf"]))

    # now generate the padding for each layer (two side)
    patch_params["pad_l"] = patch_params["single_rf"] // 2
    patch_params["pad_r"] = max(
        0,
        patch_params["grain_rf"]
        + feat_stride * grain_size * (n_patch - 1)
        - patch_params["single_rf"] // 2
        - resolution,
    )

    return patch_params


def get_recompute_layer(model, split_idx):
    layer_cnt = 1  # first conv

    for i in range(split_idx):
        block = model["blocks"][i]
        if "pointwise1" in block and block["pointwise1"] is not None:
            layer_cnt += 1
        if "depthwise" in block and block["depthwise"] is not None:
            layer_cnt += 1
        if "pointwise2" in block and block["pointwise2"] is not None:
            layer_cnt += 1

    return layer_cnt


def compute_receptive_field(layers, layer_cnt, grain=1):
    for i in range(layer_cnt):
        op = layers[(layer_cnt - 1) - i]  # trace in a backward manner
        layer_info = op.get_layer_info()
        if layer_info["op"] == "CONV_2D" or layer_info["op"] == "DEPTHWISE_CONV_2D":  # receptive field will increase
            stride = layer_info["stride_h"]
            kernel_size = max(layer_info["kernel_h"], layer_info["kernel_w"])
            if stride in [1, 2]:
                if stride == 1:
                    grain += kernel_size - 1
                else:
                    grain = (grain - 1) * 2 + kernel_size
        else:
            pass

    return grain
