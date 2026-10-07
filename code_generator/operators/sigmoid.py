import warnings

from .basic_utils import basicOperator, deep_copy_dicts, overwrite_dicts

__all__ = ["sigmoid"]

default_params = {
    # op related
    "op": "LOGISTIC",
    "input_idx": None,
    "output_idx": None,
    # tensor related
    "input_size": None,
    "input_dtype": "float32",
    "output_dtype": "float32",
    # quantization related
    "weight_value": None,
    "bias": None,
    "input_zero_point": None,
    "output_zero_point": None,
    "input_scale": None,
    "output_scale": None,
    "multiplier": None,
    "shift": None,
}


class sigmoid(basicOperator):
    def __init__(self, params: dict) -> None:
        self.params = deep_copy_dicts(default_params)
        overwrite_dicts(self.params, params)
        super().__init__()
        # handle input/output tensors in HWC format
        self._add_input(self.params["input_idx"], self.params["input_dtype"], self.params["input_size"], 1, 1)
        self._add_output(self.params["output_idx"], self.params["output_dtype"], self.params["input_size"], 1, 1)

        if None in default_params:
            warnings.warn(f"parameters are not all set for op {self.params['op']}")

    def generate_inference_str(self):
        params = self.params
        if params["input_dtype"] == "float32":
            string = (
                f"tte_sigmoid({self.params['input_size']},"
                + f"{self._getBufferstrCast(params['input_buf_add'], params['input_buf_add_offset'])},"
                + f"{self._getBufferstrCast(params['output_buf_add'], params['output_buf_add_offset'])});\n"
            )
        elif params["input_dtype"] == "int8":
            string = (
                f"tte_sigmoid_fpreq({self.params['input_size']},"
                + f"{self._getBufferstr(params['input_buf_add'], params['input_buf_add_offset'])},"
                + f"{self.params['input_scale']}f,{self.params['input_zero_point']}.0f,"
                + f"{self.params['output_scale']}f,{self.params['output_zero_point']}.0f,"
                + f"{self._getBufferstr(params['output_buf_add'], params['output_buf_add_offset'])});\n"
            )
        else:
            raise NotImplementedError
        return string
