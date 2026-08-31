import warnings

from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts

__all__ = ["StarForward"]

# Fused StarBlock forward step: act(f1(x)) * f2(x). See
# static_layers.py's StaticStarBlock.forward() and
# docs/star_forward_notes.md for the full design trail.
#
# Only one input tensor (x, shared by both conv branches) and one output
# tensor (the post-multiply result) are registered with the memory
# scheduler -- the whole point of this fusion is that x1/x2 (each
# mid_c x H x W) never exist as scheduler-visible buffers at all, only as
# small per-pixel accumulators inside the kernel's own loop. f1/f2's
# weights/bias/per-channel scales are NOT registered as scheduler tensors
# either (matching conv2d.py's Conv2d, which also only tracks its
# activation in/out tensors) -- they're emitted as flash constant arrays by
# CodeGenerator._parseTrainable()'s new STAR_FORWARD branch instead.

default_params = {
    # op related
    "op": "STAR_FORWARD",
    "input_idx": None,
    "output_idx": None,
    # tensor related (x: shared conv input; output: same H,W, mid_c channels)
    "input_h": None,
    "input_w": None,
    "input_c": None,
    "output_h": None,
    "output_w": None,
    "mid_c": None,
    "input_dtype": "int8",
    "output_dtype": "int8",
    # shared conv input quantization
    "input_zero_point": None,
    "input_scale": None,
    # f1 branch (fused RELU6)
    "f1_weight_value": None,
    "f1_bias": None,
    "f1_effective_scale": None,
    "f1_multiplier": None,
    "f1_shift": None,
    "f1_output_zero_point": None,
    "f1_output_scale": None,
    # f2 branch (no activation)
    "f2_weight_value": None,
    "f2_bias": None,
    "f2_effective_scale": None,
    "f2_multiplier": None,
    "f2_shift": None,
    "f2_output_zero_point": None,
    "f2_output_scale": None,
    # final elementwise-combine (the fused MUL node's own tensor quant)
    "output_zero_point": None,
    "output_scale": None,
}


class StarForward(basicOperator):
    def __init__(self, params: dict) -> None:
        self.params = deep_copy_dicts(default_params)
        overwrite_dicts(self.params, params)
        super().__init__()

        self._add_input(
            self.params["input_idx"],
            self.params["input_dtype"],
            self.params["input_c"],
            self.params["input_w"],
            self.params["input_h"],
        )
        self._add_output(
            self.params["output_idx"],
            self.params["output_dtype"],
            self.params["mid_c"],
            self.params["output_w"],
            self.params["output_h"],
        )

        if None in default_params:
            warnings.warn(f"parameters are not all set for op {self.params['op']}")

    def get_macs(self) -> int:
        p = self.params
        # two 1x1 convs (input_c -> mid_c each), plus the elementwise multiply
        # itself (mid_c per pixel) -- matches mul.py's own get_macs() convention
        # (p["input_size"]) for the op this replaces, so a fused config's MACs
        # are comparable to the unfused CONV_2D+CONV_2D+MUL total.
        conv_macs = 2 * p["output_h"] * p["output_w"] * p["input_c"] * p["mid_c"]
        mul_macs = p["output_h"] * p["output_w"] * p["mid_c"]
        return conv_macs + mul_macs

    def get_weights_size(self) -> int:
        p = self.params
        size = 4 if p["input_dtype"] in {"float32", "fp32"} else 1
        # f1 + f2, each [mid_c, input_c] (1x1 kernel)
        return 2 * p["mid_c"] * p["input_c"] * size

    def get_bias_size(self) -> int:
        p = self.params
        return 2 * 4 * p["mid_c"]  # int32 bias, f1 + f2

    def get_scale_size(self) -> int:
        p = self.params
        return 2 * 4 * p["mid_c"]  # float per-channel scale, f1 + f2

    def generate_inference_str(self):
        p = self.params
        idx1 = p["parsed_trainable_f1"]
        idx2 = p["parsed_trainable_f2"]
        input_offset = p["input_zero_point"] * -1

        string = (
            f"star_forward({self._getBufferstr(p['input_buf_add'], p['input_buf_add_offset'])},"
            + f"{str(p['input_h'])},{str(p['input_w'])},{str(p['input_c'])},{str(input_offset)},"
            + f"(const q7_t*) weight{idx1},bias{idx1},scales{idx1},{str(p['f1_output_zero_point'])},"
            + f"(const q7_t*) weight{idx2},bias{idx2},scales{idx2},{str(p['f2_output_zero_point'])},"
            + f"{str(p['mid_c'])},{p['f1_output_scale']}f,{p['f2_output_scale']}f,"
            + f"{p['output_scale']}f,{str(p['output_zero_point'])},"
            + f"{self._getBufferstr(p['output_buf_add'], p['output_buf_add_offset'])});\n"
        )
        return string
