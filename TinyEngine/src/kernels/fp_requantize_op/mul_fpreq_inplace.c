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
  const float inv_output_scale = 1.0f / output_scale;
  int i = 0;

  for (; i + 1 < size; i += 2) {
    const float v0 = ((float) data[i]     - zero) * scale;
    const float v1 = ((float) data[i + 1] - zero) * scale;

    int32_t o0 = (int32_t) roundf(v0 * v0 * inv_output_scale + zero_y);
    int32_t o1 = (int32_t) roundf(v1 * v1 * inv_output_scale + zero_y);
    o0 = TN_MAX(o0, -128); o0 = TN_MIN(o0, 127);
    o1 = TN_MAX(o1, -128); o1 = TN_MIN(o1, 127);
    data[i] = (int8_t) o0;
    data[i + 1] = (int8_t) o1;
  }
  for (; i < size; i++) {
    const float v = ((float) data[i] - zero) * scale;
    int32_t o = (int32_t) roundf(v * v * inv_output_scale + zero_y);
    o = TN_MAX(o, -128); o = TN_MIN(o, 127);
    data[i] = (int8_t) o;
  }
  return STATE_SUCCESS;
}
