/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   add_fpreq.c
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

/* Each element is one long chain of dependent float operations and the Cortex-M7 cannot reorder them, so four elements
 * are done per iteration, all loads first: the chains overlap and the stalls disappear. Same arithmetic per element.
 * Inputs and output may overlap element for element (in-place), which is why a block's loads precede its stores. */
#define ADD_FPREQ_ELEM(a8, b8)                                                                     \
  ({ float fa = ((float)(a8) - input1_zero) * input1_scale;                                        \
     float fb = ((float)(b8) - input2_zero) * input2_scale;                                        \
     int o = (int)roundf((fa + fb) / output_scale + zero_y); /* == round() of the float, single precision */\
     o = TN_MAX(o, -128);                                                                          \
     TN_MIN(o, 127); })

tinyengine_status add_fpreq(int size, const int8_t* input1_data, const float input1_scale, const float input1_zero,
			const int8_t* input2_data, const float input2_scale, const float input2_zero, const float output_scale,
			const float zero_y, int8_t* output_data) {
  int i = 0;
  for (; i + 3 < size; i += 4) {
    const int8_t a0 = input1_data[i], a1 = input1_data[i + 1], a2 = input1_data[i + 2], a3 = input1_data[i + 3];
    const int8_t b0 = input2_data[i], b1 = input2_data[i + 1], b2 = input2_data[i + 2], b3 = input2_data[i + 3];
    const int o0 = ADD_FPREQ_ELEM(a0, b0);
    const int o1 = ADD_FPREQ_ELEM(a1, b1);
    const int o2 = ADD_FPREQ_ELEM(a2, b2);
    const int o3 = ADD_FPREQ_ELEM(a3, b3);
    output_data[i] = (int8_t)o0;
    output_data[i + 1] = (int8_t)o1;
    output_data[i + 2] = (int8_t)o2;
    output_data[i + 3] = (int8_t)o3;
  }
  for (; i < size; ++i) {
    output_data[i] = (int8_t)ADD_FPREQ_ELEM(input1_data[i], input2_data[i]);
  }
  return STATE_SUCCESS;
}

const int activation_min = -128;
const int activation_max = 127;
tinyengine_status add_fpreq_mask(int size, const int8_t* input1_data, const float input1_scale, const float input1_zero,
			const int8_t* input2_data, const float input2_scale, const float input2_zero, const float output_scale,
			const float zero_y, int8_t* output_data, int8_t* output_mask) {
  for (int i = 0; i < size; ++i) {
	  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
      int8_t mask_value = 1;
	  if (clamped_output < activation_min){
		  clamped_output = activation_min;
		  mask_value = 0;
	  }
	  if (clamped_output > activation_max){
		  clamped_output = activation_max;
		  mask_value = 0;
	  }
      output_data[i] = (int8_t)(clamped_output);
      output_mask[i] = mask_value;
  }
}


tinyengine_status add_fpreq_bitmask(int size, const int8_t* input1_data, const float input1_scale, const float input1_zero,
			const int8_t* input2_data, const float input2_scale, const float input2_zero, const float output_scale,
			const float zero_y, int8_t* output_data, int8_t* output_mask) {
  int mask_idx = 0;
  for (int i = 0; i < size; ++i) {
	  float input1_fp = ((float)*input1_data++ - input1_zero) * input1_scale;
	  float input2_fp = ((float)*input2_data++ - input2_zero) * input2_scale;
      int clamped_output = (int)round((input1_fp + input2_fp) / output_scale + zero_y); // to align with tvm implementation
      int8_t mask_value = 1;
	  if (clamped_output < activation_min){
		  clamped_output = activation_min;
		  mask_value = 0;
	  }
	  if (clamped_output > activation_max){
		  clamped_output = activation_max;
		  mask_value = 0;
	  }
      output_data[i] = (int8_t)(clamped_output);
	  if (mask_value == 1)
		  BIT_SET(*output_mask, mask_idx);
	  else
		  BIT_CLEAR(*output_mask, mask_idx);
	  mask_idx++;
	  if (mask_idx == 8){
		  mask_idx = 0;
		  output_mask++;
	  }
  }
}
