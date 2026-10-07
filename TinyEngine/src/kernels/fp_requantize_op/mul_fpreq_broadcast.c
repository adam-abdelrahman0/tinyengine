/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   mul_fpreq_broadcast.c
 *
 * Broadcast int8 multiply of a full HxWxC tensor by a per-channel C-length
 * vector, reused across every spatial position -- e.g. a squeeze-excite
 * gate's sigmoid output ([1,C]) scaling the feature map it was pooled
 * from ([1,H,W,C]). mul_fpreq.c only covers the same-shape case; this adds
 * the broadcast loop mul.py's float32 path already had, but through the
 * same dequant/requant convention as mul_fpreq.c.
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

tinyengine_status mul_fpreq_broadcast(int size, const int8_t* input_data, const float input_scale,
			const float input_zero, const int8_t* scaler_data, int scaler_size, const float scaler_scale,
			const float scaler_zero, const float output_scale, const float zero_y, int8_t* output_data) {
  const int hw_count = size / scaler_size;

  for (int hw = 0; hw < hw_count; hw++) {
    for (int c = 0; c < scaler_size; c++) {
      float input_fp = ((float)*input_data++ - input_zero) * input_scale;
      float scaler_fp = ((float)scaler_data[c] - scaler_zero) * scaler_scale;

      int clamped_output = (int)round((input_fp * scaler_fp) / output_scale + zero_y);
      clamped_output = TN_MAX(clamped_output, -128);
      clamped_output = TN_MIN(clamped_output, 127);

      *output_data++ = (int8_t)(clamped_output);
    }
  }
  return STATE_SUCCESS;
}
