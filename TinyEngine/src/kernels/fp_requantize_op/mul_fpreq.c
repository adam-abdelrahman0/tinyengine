/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   mul_fpreq.c
 *
 * Elementwise int8 multiply of two same-shape tensors, float-requantized.
 * Added for StarBlock (act(f1) * f2), which needs the same-shape case
 * element_mult_nx1() (spatial nxnxc * 1x1xc broadcast, for SE blocks)
 * doesn't cover. Mirrors add_fpreq.c's dequant/requant convention.
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

tinyengine_status mul_fpreq(int size, const int8_t* input1_data, const float input1_scale, const float input1_zero,
			const int8_t* input2_data, const float input2_scale, const float input2_zero, const float output_scale,
			const float zero_y, int8_t* output_data) {
  for (int i = 0; i < size; ++i) {
	  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;

    int clamped_output = (int)round((input1_fp * input2_fp) / output_scale + zero_y);
    clamped_output = TN_MAX(clamped_output, -128);
    clamped_output = TN_MIN(clamped_output, 127);
    
    output_data[i] = (int8_t)(clamped_output);
  }
  return STATE_SUCCESS;
}
