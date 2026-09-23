/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   convolve_1x1_s8_selfgate.c
 *
 * Fused 1x1 conv + StarBlockV self-gate square: relu6(conv(x))^2, in one
 * pass. StarBlockV's "act(f(x)) * x" MUL always turns out, once the graph
 * optimizer runs, to have both operands trace back to the identical
 * tensor (see mul.py's generate_inference_str() int8 branch and
 * mul_fpreq_inplace.c) -- i.e. it's really relu6(f(x)) squared, not a
 * multiply against a second tensor. Previously codegen'd as two separate
 * steps: this file's conv math (unfused, via convolve_1x1_s8) writes its
 * RELU6-clamped int8 result to SRAM, then a second full pass
 * (mul_fpreq_inplace) reads it back, squares it, and writes it again.
 *
 * This does both in one loop: each output element's conv accumulator is
 * requantized/clamped (exactly convolve_1x1_s8.c's per-channel formula,
 * called out below), then immediately dequantized, squared, and
 * requantized back to int8 (exactly mul_fpreq_inplace.c's formula) before
 * ever being written to memory -- no separate read-modify-write pass over
 * the buffer. SRAM footprint is unchanged (the two steps already shared
 * one buffer via the scheduler's in-place aliasing -- see
 * GeneralMemoryScheduler.allocateMemory()'s MUL case); the saving here is
 * one fewer full pass over the output (fewer loads/stores, no second
 * function-call/loop overhead), i.e. latency/code-size, not memory.
 *
 * This is a portable scalar implementation (no CMSIS SIMD packing, unlike
 * convolve_1x1_s8.c's 2-column im2col path) -- correctness and a single
 * clean fusion point over the columnwise-SIMD throughput convolve_1x1_s8
 * gets from arm_nn_mat_mult_kernel_s8_s16_reordered, which has no per-
 * element hook to fuse into without modifying that shared routine (used
 * by every other 1x1 conv in the binary). A SIMD version akin to
 * star_forward.c's v2 rewrite is a possible follow-up if this becomes a
 * measured latency bottleneck.
 *
 * Reference papers:
 *  - MCUNet: Tiny Deep Learning on IoT Device, NeurIPS 2020
 *  - MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NeurIPS 2021
 *  - MCUNetV3: On-Device Training Under 256KB Memory, NeurIPS 2022
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include <math.h>
#include "arm_nnsupportfunctions.h"
#include "tinyengine_function.h"

/*
 * input:  [input_h * input_w, input_ch] int8, HWC layout
 * kernel: [output_ch, input_ch] int8 (1x1, OHWI==OI for a pointwise conv)
 * bias/output_shift/output_mult: per-output-channel int32 arrays, same
 *         convention as convolve_1x1_s8()
 * out_offset/input_offset/out_activation_min/out_activation_max: same
 *         convention as convolve_1x1_s8() -- this is the conv's own
 *         RELU6 clamp, fused into its requantization exactly as it
 *         already is for every other quantized conv.
 * mul_scale/mul_zero: the conv output's own dequant scale/zero-point --
 *         same values mul_fpreq_inplace() would have received as its
 *         (scale, zero) pair for this tensor.
 * mul_output_scale/mul_output_zero: the squared result's output tensor's
 *         scale/zero-point -- same as mul_fpreq_inplace()'s
 *         (output_scale, zero_y).
 * output: [output_h * output_w, output_ch] int8, the fused
 *         relu6(conv(x))^2 result.
 */
tinyengine_status convolve_1x1_s8_selfgate(
    const q7_t *input, const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
    const q7_t *kernel, const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
    const int32_t out_offset, const int32_t input_offset, const int32_t out_activation_min,
    const int32_t out_activation_max, q7_t *output, const uint16_t output_x, const uint16_t output_y,
    const uint16_t output_ch, q15_t *runtime_buf, const float mul_scale, const float mul_zero,
    const float mul_output_scale, const float mul_output_zero) {
    (void)input_x;
    (void)input_y;
    (void)runtime_buf; /* unused in this scalar path; kept for call-site symmetry with convolve_1x1_s8() */

    const int32_t num_pixels = (int32_t)output_x * (int32_t)output_y;
    const float inv_mul_output_scale = 1.0f / mul_output_scale;

    for (int32_t p = 0; p < num_pixels; p++) {
        const q7_t *x_row = &input[p * input_ch];
        q7_t *out_row = &output[p * output_ch];

        for (int32_t o = 0; o < output_ch; o++) {
            const q7_t *w_row = &kernel[o * input_ch];

            /* conv accumulator -- identical formula to convolve_1x1_s8.c's
             * scalar tail case */
            q31_t acc = bias[o];
            for (int32_t ci = 0; ci < input_ch; ci++) {
                acc += ((int32_t)x_row[ci] + input_offset) * (int32_t)w_row[ci];
            }

            /* requantize + RELU6 clamp -- identical to convolve_1x1_s8.c */
            q31_t conv_out = arm_nn_requantize(acc, output_mult[o], output_shift[o]);
            conv_out += out_offset;
            conv_out = TN_MAX(conv_out, out_activation_min);
            conv_out = TN_MIN(conv_out, out_activation_max);

            /* self-gate square -- identical formula to mul_fpreq_inplace.c,
             * applied in-register instead of via a second buffer pass */
            float val = ((float)conv_out - mul_zero) * mul_scale;
            int32_t sq = (int32_t)roundf((val * val) * inv_mul_output_scale + mul_output_zero);
            sq = TN_MAX(sq, -128);
            sq = TN_MIN(sq, 127);

            out_row[o] = (q7_t)sq;
        }
    }

    return STATE_SUCCESS;
}
