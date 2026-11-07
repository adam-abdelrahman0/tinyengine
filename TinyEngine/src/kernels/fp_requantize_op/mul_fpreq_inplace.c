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
#include <string.h>
#include "arm_math.h"
#include "tinyengine_function.h"

tinyengine_status mul_fpreq_inplace(int size, int8_t* data, const float scale, const float zero,
			const float output_scale, const float zero_y) {
  const float inv_output_scale = 1.0f / output_scale;
  int i = 0;

  /* The result depends only on the int8 in data[i], so for anything bigger than the table itself evaluate the
   * formula once per possible value (same expression as the loops below) and look every element up instead:
   * about 2 cycles per element against about 23 for the float math. */
  if (size >= 256) {
    uint8_t lut[256]; /* indexed by the raw byte, so no sign adjustment per element */
    for (int b = 0; b < 256; b++) {
      const float v = ((float) (int8_t) b - zero) * scale;
      int32_t o = (int32_t) roundf(v * v * inv_output_scale + zero_y);
      o = TN_MAX(o, -128); o = TN_MIN(o, 127);
      lut[b] = (uint8_t) (int8_t) o;
    }
#ifdef TE_INPLACE_BYTE_LUT /* sweep: the earlier byte-at-a-time table loop */
    for (; i + 3 < size; i += 4) {
      const uint8_t a = lut[(uint8_t) data[i]], b = lut[(uint8_t) data[i + 1]], c = lut[(uint8_t) data[i + 2]], d = lut[(uint8_t) data[i + 3]];
      data[i] = (int8_t) a; data[i + 1] = (int8_t) b; data[i + 2] = (int8_t) c; data[i + 3] = (int8_t) d;
    }
#else
    /* one word load and store per four elements (the byte loop was bound by a store per element that could alias the
     * next load); little endian, element i in the low byte */
    for (; i + 3 < size; i += 4) {
      uint32_t v;
      memcpy(&v, data + i, 4);
      const uint32_t o = (uint32_t) lut[v & 0xFF] | ((uint32_t) lut[(v >> 8) & 0xFF] << 8)
                       | ((uint32_t) lut[(v >> 16) & 0xFF] << 16) | ((uint32_t) lut[v >> 24] << 24);
      memcpy(data + i, &o, 4);
    }
#endif
  }

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
