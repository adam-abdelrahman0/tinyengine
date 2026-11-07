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

/* Four channels per iteration with the loads first (see add_fpreq.c): the dependent float chains of neighbouring elements
 * overlap instead of stalling the in-order pipeline. Same arithmetic per element. */
#define MUL_BC_ELEM(x8, s8)                                                                      \
  ({ float xf = ((float)(x8) - input_zero) * input_scale;                                        \
     float sf = ((float)(s8) - scaler_zero) * scaler_scale;                                      \
     int o = (int)round((xf * sf) / output_scale + zero_y);                                      \
     o = TN_MAX(o, -128);                                                                        \
     TN_MIN(o, 127); })

tinyengine_status mul_fpreq_broadcast(int size, const int8_t* input_data, const float input_scale,
			const float input_zero, const int8_t* scaler_data, int scaler_size, const float scaler_scale,
			const float scaler_zero, const float output_scale, const float zero_y, int8_t* output_data) {
  const int hw_count = size / scaler_size;

  for (int hw = 0; hw < hw_count; hw++) {
    int c = 0;
    for (; c + 3 < scaler_size; c += 4) {
      const int8_t x0 = input_data[0], x1 = input_data[1], x2 = input_data[2], x3 = input_data[3];
      const int8_t s0 = scaler_data[c], s1 = scaler_data[c + 1], s2 = scaler_data[c + 2], s3 = scaler_data[c + 3];
      const int o0 = MUL_BC_ELEM(x0, s0);
      const int o1 = MUL_BC_ELEM(x1, s1);
      const int o2 = MUL_BC_ELEM(x2, s2);
      const int o3 = MUL_BC_ELEM(x3, s3);
      input_data += 4;
      output_data[0] = (int8_t)o0;
      output_data[1] = (int8_t)o1;
      output_data[2] = (int8_t)o2;
      output_data[3] = (int8_t)o3;
      output_data += 4;
    }
    for (; c < scaler_size; c++) {
      *output_data++ = (int8_t)MUL_BC_ELEM(*input_data++, scaler_data[c]);
    }
  }
  return STATE_SUCCESS;
}
