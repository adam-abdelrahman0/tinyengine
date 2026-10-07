/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   tte_sigmoid_fpreq.c
 *
 * int8 sigmoid, float-dequantized/requantized -- same convention as
 * add_fpreq.c/mul_fpreq.c: dequantize to float, apply the real op, then
 * requantize to the output's scale/zero-point. Needed for a squeeze-excite
 * gate's Sigmoid activation (see code_generator/operators/sigmoid.py),
 * which runs in the same int8 graph as everything around it.
 *
 * Reference papers:
 *  - MCUNet: Tiny Deep Learning on IoT Device, NeurIPS 2020
 *  - MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021
 *  - MCUNetV3: On-Device Training Under 256KB Memory, NeurIPS 2022
 * Contact authors:
 *  - Wei-Ming Chen, wmchen@mit.edu
 *  - Wei-Chen Wang, wweichen@mit.edu
 *  - Ji Lin, jilin@mit.edu
 *  - Ligeng Zhu, ligeng@mit.edu
 *  - Song Han, songhan@mit.edu
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include <math.h>
#include "arm_math.h"
#include "tinyengine_function.h"

tinyengine_status tte_sigmoid_fpreq(int size, const int8_t* input_data, const float input_scale,
			const float input_zero, const float output_scale, const float zero_y, int8_t* output_data) {
  for (int i = 0; i < size; ++i) {
    float input_fp = ((float)*input_data++ - input_zero) * input_scale;
    float sigmoid_fp = 1.0f / (1.0f + exp(-input_fp));

    int clamped_output = (int)round(sigmoid_fp / output_scale + zero_y);
    clamped_output = TN_MAX(clamped_output, -128);
    clamped_output = TN_MIN(clamped_output, 127);

    output_data[i] = (int8_t)(clamped_output);
  }
  return STATE_SUCCESS;
}
