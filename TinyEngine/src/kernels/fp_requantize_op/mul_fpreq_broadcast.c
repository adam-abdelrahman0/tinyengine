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

/* TE_FPREQ_UNROLL channels per iteration (default 4) with the loads first (see add_fpreq.c): the dependent float chains of
 * neighbouring elements overlap instead of stalling the in-order pipeline. Same arithmetic per element. */
#ifndef TE_FPREQ_UNROLL
#define TE_FPREQ_UNROLL 4
#endif
#ifdef TE_FPREQ_ROUND_DOUBLE
#define FPREQ_ROUND(x) round(x)
#else
#define FPREQ_ROUND(x) roundf(x)
#endif
#define MUL_BC_ELEM(x8, s8)                                                                      \
  ({ float xf = ((float)(x8) - input_zero) * input_scale;                                        \
     float sf = ((float)(s8) - scaler_zero) * scaler_scale;                                      \
     int o = (int)FPREQ_ROUND((xf * sf) / output_scale + zero_y);                                \
     o = TN_MAX(o, -128);                                                                        \
     TN_MIN(o, 127); })

tinyengine_status mul_fpreq_broadcast(int size, const int8_t* input_data, const float input_scale,
			const float input_zero, const int8_t* scaler_data, int scaler_size, const float scaler_scale,
			const float scaler_zero, const float output_scale, const float zero_y, int8_t* output_data) {
  const int hw_count = size / scaler_size;

  for (int hw = 0; hw < hw_count; hw++) {
    int c = 0;
    for (; c + TE_FPREQ_UNROLL <= scaler_size; c += TE_FPREQ_UNROLL) {
      int8_t x[TE_FPREQ_UNROLL], sc[TE_FPREQ_UNROLL];
      int o[TE_FPREQ_UNROLL];
#pragma GCC unroll 16
      for (int k = 0; k < TE_FPREQ_UNROLL; k++) { x[k] = input_data[k]; sc[k] = scaler_data[c + k]; }
#pragma GCC unroll 16
      for (int k = 0; k < TE_FPREQ_UNROLL; k++) { o[k] = MUL_BC_ELEM(x[k], sc[k]); }
      input_data += TE_FPREQ_UNROLL;
#pragma GCC unroll 16
      for (int k = 0; k < TE_FPREQ_UNROLL; k++) { output_data[k] = (int8_t)o[k]; }
      output_data += TE_FPREQ_UNROLL;
    }
    for (; c < scaler_size; c++) {
      *output_data++ = (int8_t)MUL_BC_ELEM(*input_data++, scaler_data[c]);
    }
  }
  return STATE_SUCCESS;
}
