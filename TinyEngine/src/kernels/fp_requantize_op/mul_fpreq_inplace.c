/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   mul_fpreq_inplace.c
 *
 * In-place variant of mul_fpreq for the case where both multiply operands
 * trace back to the identical graph tensor (same buffer, same scale/
 * zero-point) -- e.g. StarBlockV's self-gate act(f(x)) * x, when the
 * quantizer/graph optimizer has collapsed the two operands to one shared
 * tensor (see GeneralMemoryScheduler.allocateMemory()'s MUL case, and
 * mul.py's generate_inference_str(), for the same-tensor-identity check
 * that selects this over mul_fpreq). Squares the dequantized value and
 * writes back into the same buffer, so no second SRAM allocation is
 * needed for the multiply's output the way mul_fpreq requires -- the
 * two-buffer version can't be in-place because it takes two independent
 * inputs in the general case.
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

tinyengine_status mul_fpreq_inplace(int size, int8_t* data, const float scale, const float zero,
			const float output_scale, const float zero_y) {
  for (int i = 0; i < size; ++i) {
	  float val = ((float)data[i] - zero) * scale;

    int clamped_output = (int)round((val * val) / output_scale + zero_y);
    clamped_output = TN_MAX(clamped_output, -128);
    clamped_output = TN_MIN(clamped_output, 127);

    data[i] = (int8_t)(clamped_output);
  }
  return STATE_SUCCESS;
}
