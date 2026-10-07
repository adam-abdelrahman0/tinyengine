# ----------------------------------------------------------------------
# Project: TinyEngine
# Title:   avgpool2d.py
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

import warnings

from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts

__all__ = ["AvgPool2d"]

default_params = {
    # op related
    "op": "AVERAGE_POOL_2D",
    "is_SEBlock": False,
    "filter_h": None,
    "filter_w": None,
    "stride_h": None,
    "stride_w": None,
    "pad_h": None,
    "pad_w": None,
    "input_idx": None,
    "output_idx": None,
    # tensor related
    "input_dim": None,
    "input_h": None,
    "input_w": None,
    "input_c": None,
    "output_dim": None,
    "output_h": None,
    "output_w": None,
    "output_c": None,
    "kernel_h": None,
    "kernel_w": None,
    "input_dtype": "int8",
    "output_dtype": "int8",
    # Optional: only set when the caller knows input/output quantization may
    # differ (see mean1dto2d.py). None when unknown/not applicable -- the
    # plain integer avg_pooling kernel is used as-is in that case, matching
    # prior behavior.
    "input_scale": None,
    "input_zero_point": None,
    "output_scale": None,
    "output_zero_point": None,
}


class AvgPool2d(basicOperator):
    def __init__(self, params: dict) -> None:
        self.params = deep_copy_dicts(default_params)
        overwrite_dicts(self.params, params)
        super().__init__()
        # handle input/output tensors in HWC format
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
            self.params["output_c"],
            self.params["output_w"],
            self.params["output_h"],
        )

        if None in default_params:
            warnings.warn(f"parameters are not all set for op {self.params['op']}")

    def generate_inference_str(self):
        params = self.params
        output_buf = self._getBufferstr(params["output_buf_add"], params["output_buf_add_offset"])
        string = (
            f"avg_pooling({self._getBufferstr(params['input_buf_add'], params['input_buf_add_offset'])},"
            + f"{str(params['input_h'])},{str(params['input_w'])},{str(params['input_c'])},{str(params['filter_h'])},"
        )
        string += f"{str(params['filter_w'])},1,1,-128,127,{output_buf});\n"

        # avg_pooling() has no rescale path -- it just integer-averages,
        # implicitly assuming input and output share one quantization. When
        # they don't (confirmed on a real model: TFLite gave this MEAN
        # different input/output scales), rescale the result in place:
        # dequantize by the input scale, requantize by the output scale.
        in_scale = params.get("input_scale")
        out_scale = params.get("output_scale")
        if in_scale is not None and out_scale is not None and abs(in_scale - out_scale) > 1e-12:
            in_zero = params.get("input_zero_point") or 0
            out_zero = params.get("output_zero_point") or 0
            n = params["output_h"] * params["output_w"] * params["output_c"]
            rescale = float(in_scale) / float(out_scale)
            string += (
                f"{{\n"
                f"q7_t *_rq_out = {output_buf};\n"
                f"for (int _rq_i = 0; _rq_i < {n}; _rq_i++) {{\n"
                f"    float _rq_f = (float)(_rq_out[_rq_i] - ({in_zero})) * {rescale}f;\n"
                f"    int _rq_v = (int)(_rq_f >= 0.0f ? _rq_f + 0.5f : _rq_f - 0.5f) + ({out_zero});\n"
                f"    if (_rq_v < -128) _rq_v = -128;\n"
                f"    if (_rq_v > 127) _rq_v = 127;\n"
                f"    _rq_out[_rq_i] = (q7_t)_rq_v;\n"
                f"}}\n"
                f"}}\n"
            )

        return string
