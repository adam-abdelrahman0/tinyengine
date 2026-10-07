from code_generator.operators import avgpool2d
from code_generator.tflite import Model

from .utils import get_input_tensors, get_output_tensors, getTensorTypeStr


def _get_nhwc(shape):
    """Unpack a raw TFLite tensor shape as (N, H, W, C) -- TFLite's native
    activation tensor layout -- the same convention conv2d.py's parser uses
    (`_, input_h, input_w, input_c = input_tensor.tensor.ShapeAsNumpy()`).
    get_hwc_from_chwshape() (used here previously) instead assumes an
    [N, C, H, W]-ordered shape, which a raw TFLite tensor never is; applying
    it to e.g. [1, 7, 7, 168] silently produced h=7, w=168, c=7 instead of
    the correct h=7, w=7, c=168.
    """
    if len(shape) == 4:
        return int(shape[1]), int(shape[2]), int(shape[3])
    elif len(shape) == 3:
        return int(shape[0]), int(shape[1]), int(shape[2])
    elif len(shape) == 2:
        return 1, 1, int(shape[1])
    return 1, 1, 1


def parse_mead1dto2d(op, model: Model.Model, MEAN2Dholder=None):
    # Incase no params
    input_type = None

    # get input, weight, and output tensors
    input_tensors = get_input_tensors(op, model)
    input_tensor = input_tensors[0]

    output_tensors = get_output_tensors(op, model)
    assert len(output_tensors) == 1, "output tensors length should be 1"
    output_tensor = output_tensors[0]

    # shapes
    input_shape = input_tensor.tensor.ShapeAsNumpy()
    output_shape = output_tensor.tensor.ShapeAsNumpy()

    input_h, input_w, input_c = _get_nhwc(input_shape)
    output_h, output_w, output_c = _get_nhwc(output_shape)
    input_type = getTensorTypeStr(input_tensor.tensor.Type())

    if not MEAN2Dholder.has_first_1D:
        if output_h == 1 and output_w == 1 and (input_h > 1 or input_w > 1):
            # A single MEAN reducing both H and W in one op (e.g. PyTorch's
            # x.mean(dim=[2, 3]) / AdaptiveAvgPool2d(1), as opposed to two
            # sequential 1D reductions). The two-call state machine below
            # waits forever for a second MEAN that will never come and
            # silently drops this op (returns None), leaving whatever reads
            # its output -- typically the classifier's FULLY_CONNECTED head
            # -- consuming stale, unpooled feature-map data instead. Confirmed
            # via a real logit mismatch: final output was wildly wrong while
            # every earlier layer, including all residual ADDs, matched the
            # TFLite reference. Emit AVERAGE_POOL_2D directly from this one
            # op's own shapes instead of waiting.
            params = {
                "op": "AVERAGE_POOL_2D",
                "filter_h": input_h,
                "filter_w": input_w,
                "stride_h": 1,
                "stride_w": 1,
                "pad_h": 0,
                "pad_w": 0,
                "input_idx": input_tensor.tensor_idx,
                "output_idx": output_tensor.tensor_idx,
                "input_h": input_h,
                "input_w": input_w,
                "input_c": input_c,
                "input_dim": 3,
                "output_dim": 3,
                "output_h": output_h,
                "output_w": output_w,
                "output_c": output_c,
                "dtypte": input_type,
                # TFLite's AVERAGE_POOL_2D is usually a same-scale op, but this
                # model's MEAN has genuinely DIFFERENT input/output scales
                # (confirmed: 0.01746768 in vs 0.00601481 out on one real
                # checkpoint) -- the raw `avg_pooling` kernel has no rescale
                # path at all, it just integer-averages, so skipping this
                # silently produced a correct ARGMAX but systematically
                # wrong-magnitude logits. AvgPool2d.generate_inference_str()
                # emits an extra float rescale pass when these differ.
                "input_scale": input_tensor.qnn_params["scale"] if input_tensor.qnn_params else None,
                "input_zero_point": input_tensor.qnn_params["zero_point"] if input_tensor.qnn_params else None,
                "output_scale": output_tensor.qnn_params["scale"] if output_tensor.qnn_params else None,
                "output_zero_point": output_tensor.qnn_params["zero_point"] if output_tensor.qnn_params else None,
            }
            return avgpool2d.AvgPool2d(params)
        MEAN2Dholder.add_first_1D_op(input_tensor.tensor_idx, output_tensor.tensor_idx, input_h, input_w, input_c)
        return None
    elif not MEAN2Dholder.has_second_1D:
        MEAN2Dholder.add_second_1D_op(input_tensor.tensor_idx, output_tensor.tensor_idx, output_h, output_w, output_c)
        filter_h = input_h - output_h + 1
        filter_w = input_w - output_w + 1
        params = {
            # operator
            "op": "AVERAGE_POOL_2D",
            # pool parameters
            "filter_h": filter_h,
            "filter_w": filter_w,
            "stride_h": 1,
            "stride_w": 1,
            "pad_h": 0,
            "pad_w": 0,
            # tensor
            "input_idx": MEAN2Dholder.first_1D_input_idx,
            "output_idx": MEAN2Dholder.second_1D_output_idx,
            "input_h": MEAN2Dholder.input_h,
            "input_w": MEAN2Dholder.input_w,
            "input_c": MEAN2Dholder.input_c,
            "input_dim": 3,
            "output_dim": 3,
            "output_h": MEAN2Dholder.output_h,
            "output_w": MEAN2Dholder.output_w,
            "output_c": MEAN2Dholder.output_c,
            "dtypte": input_type,
        }

        op = avgpool2d.AvgPool2d(params)

        # reset MEAN2Dholder
        MEAN2Dholder.reset_holder()

        return op
    else:
        raise NotImplementedError


class MEAN2D(object):
    def __init__(self):
        self.reset_holder()

    def add_first_1D_op(self, input_idx, output_idx, input_h, input_w, input_c):
        self.first_1D_input_idx = input_idx
        self.first_1D_output_idx = output_idx
        self.input_h = input_h
        self.input_w = input_w
        self.input_c = input_c
        self.has_first_1D = True

    def add_second_1D_op(self, input_idx, output_idx, output_h, output_w, output_c):
        self.second_1D_input_idx = input_idx
        self.second_1D_output_idx = output_idx
        self.output_h = output_h
        self.output_w = output_w
        self.output_c = output_c
        self.has_second_1D = True

    def reset_holder(self):
        self.has_first_1D = False
        self.has_second_1D = False
