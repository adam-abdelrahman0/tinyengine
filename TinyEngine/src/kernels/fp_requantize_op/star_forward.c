/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   star_forward.c
 *
 * Fused StarBlock forward step: act(f1(x)) * f2(x), where f1/f2 are both
 * 1x1 (pointwise) int8 convs sharing the same input x, f1's fused
 * activation is RELU6, f2 has none, and the two outputs feed a same-shape
 * elementwise multiply. See static_layers.py's StaticStarBlock.forward().
 *
 * This replaces 3 separate ops (2x convolve_1x1_s8_fpreq + mul_fpreq) with
 * one fused loop so the mid_dim x H x W intermediate tensors (x1, x2)
 * never get materialized as full-size SRAM buffers -- only a per-pixel
 * accumulator pair lives at a time.
 *
 * v2: vectorized. The per-pixel input row is shared by every output
 * channel's f1 and f2 dot product, so it's packed once per pixel (4 int8
 * -> two offset-added q15x2 words via read_and_pad, the same helper
 * convolve_1x1_s8_fpreq.c's tail case implicitly relies on) and reused
 * across all mid_ch channels and both branches. Each dot product then
 * runs two __SMLAD-based MACs per 4 input channels instead of 4 scalar
 * multiplies. Falls back to the original scalar loop for any input_ch
 * remainder not a multiple of 4. Verified bit-identical against the v1
 * scalar implementation via a QEMU (Cortex-M7) numeric equivalence test
 * across randomized shapes, including non-multiple-of-4 input_ch --
 * see Data_Scripts/kernel_simd_verify/.
 *
 * Reference papers:
 *  - MCUNet: Tiny Deep Learning on IoT Device, NeurIPS 2020
 *  - MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021
 *  - MCUNetV3: On-Device Training Under 256KB Memory, NeurIPS 2022
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include <math.h>
#include "arm_math.h"
#include "arm_nnsupportfunctions.h"
#include "tinyengine_function.h"

/*
 * input:          [input_h * input_w, input_ch] int8, HWC layout (shared by both branches)
 * w1/bias1/scales1/out_offset1: f1's 1x1 conv weights [mid_ch, input_ch], int32 bias [mid_ch],
 *                  per-channel effective_scale [mid_ch] (already input_scale*weight_scale/x1_scale),
 *                  x1's own output zero point. f1's fused RELU6 needs no special-casing here --
 *                  it's already encoded in x1's exported int8 range/zero-point, exactly as it
 *                  would be if f1 ran as an ordinary CONV_2D (see docs/star_forward_notes.md,
 *                  "Why no explicit RELU6 clamp" section).
 * w2/bias2/scales2/out_offset2: f2's conv, same shape, no activation.
 * input_offset:    -input_zero_point (matches conv2d.py's calling convention: negated already)
 * x1_scale/x2_scale: x1's and x2's own exported per-tensor dequant scales (used only in the
 *                  final multiply-requantize step, same convention as mul_fpreq.c)
 * output_scale/output_offset: the fused MUL node's own output tensor's scale/zero-point
 * output:          [input_h * input_w, mid_ch] int8, the block's act(f1)*f2 result
 */
tinyengine_status star_forward(const q7_t *input, const uint16_t input_h, const uint16_t input_w,
        const uint16_t input_ch, const int32_t input_offset,
        const q7_t *w1, const int32_t *bias1, const float *scales1, const int32_t out_offset1,
        const q7_t *w2, const int32_t *bias2, const float *scales2, const int32_t out_offset2,
        const uint16_t mid_ch, const float x1_scale, const float x2_scale,
        const float output_scale, const int32_t output_offset,
        q7_t *output) {
    const int32_t num_pixels = (int32_t) input_h * (int32_t) input_w;
    const int32_t ch4 = (int32_t) input_ch & ~3;          /* largest multiple of 4 <= input_ch */
    const int32_t npacked = (ch4 >> 1) + 2;               /* 2 q15x2 words per 4 channels, +2 slack */
    const int16_t inoff16 = (int16_t) input_offset;
    const q31_t offset_q15x2 = __PKHBT(inoff16, inoff16, 16);
    const float inv_output_scale = 1.0f / output_scale;

    for (int32_t p = 0; p < num_pixels; p++) {
        const q7_t *x_row = &input[p * input_ch];
        q7_t *out_row = &output[p * mid_ch];

        /* Pack + offset-add the shared input row once per pixel; every
         * mid_ch channel's f1/f2 dot product below reads from this
         * instead of re-deriving it from x_row. */
        q31_t x_packed[npacked];
        {
            const q7_t *src = x_row;
            for (int32_t ci = 0; ci < ch4; ci += 4) {
                q31_t a, b;
                src = read_and_pad(src, &a, &b);
                x_packed[ci >> 1] = __SADD16(a, offset_q15x2);
                x_packed[(ci >> 1) + 1] = __SADD16(b, offset_q15x2);
            }
        }

        for (uint16_t c = 0; c < mid_ch; c++) {
            const q7_t *w1_row = &w1[(int32_t) c * input_ch];
            const q7_t *w2_row = &w2[(int32_t) c * input_ch];

            q31_t s1 = bias1[c];
            q31_t s2 = bias2[c];

            for (int32_t ci = 0; ci < ch4; ci += 4) {
                q31_t w1a, w1b, w2a, w2b;
                read_and_pad(&w1_row[ci], &w1a, &w1b);
                read_and_pad(&w2_row[ci], &w2a, &w2b);
                const q31_t xa = x_packed[ci >> 1];
                const q31_t xb = x_packed[(ci >> 1) + 1];
                s1 = __SMLAD(w1a, xa, s1);
                s1 = __SMLAD(w1b, xb, s1);
                s2 = __SMLAD(w2a, xa, s2);
                s2 = __SMLAD(w2b, xb, s2);
            }
            /* scalar tail for any input_ch remainder not a multiple of 4 */
            for (int32_t ci = ch4; ci < (int32_t) input_ch; ci++) {
                const int32_t xv = (int32_t) x_row[ci] + input_offset;
                s1 += (int32_t) w1_row[ci] * xv;
                s2 += (int32_t) w2_row[ci] * xv;
            }

            /* per-branch requantize + clamp, identical formula to
             * convolve_1x1_s8_fpreq.c's scalar tail case */
            s1 = (int32_t) ((float) s1 * scales1[c]);
            s1 += out_offset1;
            s1 = TN_MAX(s1, -128);
            s1 = TN_MIN(s1, 127);

            s2 = (int32_t) ((float) s2 * scales2[c]);
            s2 += out_offset2;
            s2 = TN_MAX(s2, -128);
            s2 = TN_MIN(s2, 127);

            /* elementwise combine, identical formula to mul_fpreq.c (roundf,
             * not round -- this target has no hardware double support, and
             * a reciprocal instead of a per-channel division) */
            float a_fp = ((float) s1 - (float) out_offset1) * x1_scale;
            float b_fp = ((float) s2 - (float) out_offset2) * x2_scale;

            int32_t v = (int32_t) roundf(a_fp * b_fp * inv_output_scale + (float) output_offset);
            v = TN_MAX(v, -128);
            v = TN_MIN(v, 127);

            out_row[c] = (q7_t) v;
        }
    }

    return STATE_SUCCESS;
}
