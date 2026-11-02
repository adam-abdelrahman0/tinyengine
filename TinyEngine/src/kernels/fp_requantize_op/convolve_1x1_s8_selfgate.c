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
 *
 * This file has two implementations of the same operation:
 *  - convolve_1x1_s8_selfgate_scalar(): the original portable loop, one MAC at
 *    a time. Kept as the fallback (input_ch % 4 != 0) and as the reference the
 *    SIMD version is tested against.
 *  - convolve_1x1_s8_selfgate(): the SIMD version. Same im2col + 2-channel x
 *    2-column SMLAD micro-kernel as convolve_1x1_s8(), with the epilogue
 *    requantize + clamp followed by the square through a 256-entry table
 *    (the square's input is an int8, so the table is exactly the float
 *    formula evaluated for every possible input).
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include <math.h>
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "img2col_element.h"
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
tinyengine_status convolve_1x1_s8_selfgate_scalar(
    const q7_t *input, const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
    const q7_t *kernel, const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
    const int32_t out_offset, const int32_t input_offset, const int32_t out_activation_min,
    const int32_t out_activation_max, q7_t *output, const uint16_t output_x, const uint16_t output_y,
    const uint16_t output_ch, q15_t *runtime_buf, const float mul_scale, const float mul_zero,
    const float mul_output_scale, const float mul_output_zero) {
    (void)input_x;
    (void)input_y;

    const int32_t num_pixels = (int32_t)output_x * (int32_t)output_y;
    const float inv_mul_output_scale = 1.0f / mul_output_scale;

    for (int32_t p = 0; p < num_pixels; p++) {
        const q7_t *x_row = &input[p * input_ch];
        q7_t *out_row = &output[p * output_ch];

        /* Copy this pixel's whole input row to scratch before writing any of
         * its outputs. The scheduler's pointwise-overwrite growth gap lets
         * the output overlap the input buffer and is only safe if a pixel's
         * input is fully consumed before that pixel's first output byte is
         * written (convolve_1x1_s8() does the same via its im2col copy). The
         * last pixel's output starts exactly at its own input row, so reading
         * x_row per output channel would see already-overwritten values. */
        for (int32_t ci = 0; ci < input_ch; ci++) {
            runtime_buf[ci] = (q15_t)(x_row[ci] + input_offset);
        }

        for (int32_t o = 0; o < output_ch; o++) {
            const q7_t *w_row = &kernel[o * input_ch];

            /* conv accumulator -- identical formula to convolve_1x1_s8.c's
             * scalar tail case */
            q31_t acc = bias[o];
            for (int32_t ci = 0; ci < input_ch; ci++) {
                acc += (int32_t)runtime_buf[ci] * (int32_t)w_row[ci];
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

/* ---------------------------------------------------------------------------
 * SIMD version
 * ------------------------------------------------------------------------ */
#define DIM_KER_X (1U)
#define DIM_KER_Y (1U)

/* lut[q + 128] = square of the conv output q, requantized: the formula of mul_fpreq_inplace() for each int8 q */
static void selfgate_lut(q7_t *lut, const float mul_scale, const float mul_zero, const float mul_output_scale,
                         const float mul_output_zero) {
    const float inv_output_scale = 1.0f / mul_output_scale;
    for (int q = -128; q < 128; q++) {
        const float v = ((float)q - mul_zero) * mul_scale;
        int32_t o = (int32_t)roundf(v * v * inv_output_scale + mul_output_zero);
        o = TN_MAX(o, -128);
        o = TN_MIN(o, 127);
        lut[q + 128] = (q7_t)o;
    }
}

/* requantize + relu6 clamp exactly as arm_nn_mat_mult_kernel_s8_s16_reordered(), then the square via the table */
#define SG_OUT(acc, mult, shift)                                                         \
    ({                                                                                   \
        q31_t r_ = arm_nn_requantize((acc), *(mult), *(shift));                          \
        r_ += out_offset;                                                                \
        r_ = MAX(r_, activation_min);                                                    \
        r_ = MIN(r_, activation_max);                                                    \
        lut[(r_ + 128) & 0xFF];                                                          \
    })

/* arm_nn_mat_mult_kernel_s8_s16_reordered() with the square folded into the epilogue; the loop is unchanged. */
static q7_t *mat_mult_selfgate(const q7_t *input_a, const q15_t *input_b, const uint16_t output_ch,
                               const int32_t *out_shift, const int32_t *out_mult, const int32_t out_offset,
                               const int16_t activation_min, const int16_t activation_max,
                               const uint16_t num_col_a, const int32_t *const output_bias, q7_t *out_0,
                               const q7_t *lut) {
    q7_t *out_1 = out_0 + output_ch;
    const int32_t *bias = output_bias;

    uint16_t row_count = output_ch / 2;
    const q7_t *ip_a0 = input_a;
    while (row_count) {
        const q15_t *ip_b0 = input_b;
        const q15_t *ip_b1 = ip_b0 + num_col_a;
        const q7_t *ip_a1 = ip_a0 + num_col_a;

        q31_t ch_0_out_0 = *bias;
        q31_t ch_0_out_1 = *bias++;
        q31_t ch_1_out_0 = *bias;
        q31_t ch_1_out_1 = *bias++;

        uint16_t col_count = num_col_a / 4;
        while (col_count) {
            q31_t a01, a02, a11, a12;
            q31_t b0 = arm_nn_read_q15x2_ia(&ip_b0);
            q31_t b1 = arm_nn_read_q15x2_ia(&ip_b1);

            ip_a0 = read_and_pad_reordered(ip_a0, &a01, &a02);
            ip_a1 = read_and_pad_reordered(ip_a1, &a11, &a12);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a01, b1, ch_0_out_1);
            ch_1_out_0 = __SMLAD(a11, b0, ch_1_out_0);
            ch_1_out_1 = __SMLAD(a11, b1, ch_1_out_1);

            b0 = arm_nn_read_q15x2_ia(&ip_b0);
            b1 = arm_nn_read_q15x2_ia(&ip_b1);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a02, b1, ch_0_out_1);
            ch_1_out_0 = __SMLAD(a12, b0, ch_1_out_0);
            ch_1_out_1 = __SMLAD(a12, b1, ch_1_out_1);

            col_count--;
        }

        *out_0++ = SG_OUT(ch_0_out_0, out_mult, out_shift);
        *out_1++ = SG_OUT(ch_0_out_1, out_mult, out_shift);
        out_mult++;
        out_shift++;

        *out_0++ = SG_OUT(ch_1_out_0, out_mult, out_shift);
        *out_1++ = SG_OUT(ch_1_out_1, out_mult, out_shift);
        out_mult++;
        out_shift++;

        ip_a0 += num_col_a;
        row_count--;
    }

    if (output_ch & 1) {
        const q15_t *ip_b0 = input_b;
        const q15_t *ip_b1 = ip_b0 + num_col_a;

        q31_t ch_0_out_0 = *bias;
        q31_t ch_0_out_1 = ch_0_out_0;

        int32_t col_count = num_col_a / 4;
        while (col_count) {
            q31_t a01, a02;
            q31_t b0 = arm_nn_read_q15x2_ia(&ip_b0);
            q31_t b1 = arm_nn_read_q15x2_ia(&ip_b1);

            ip_a0 = read_and_pad_reordered(ip_a0, &a01, &a02);

            ch_0_out_0 = __SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a01, b1, ch_0_out_1);

            b0 = arm_nn_read_q15x2_ia(&ip_b0);
            b1 = arm_nn_read_q15x2_ia(&ip_b1);

            ch_0_out_0 = __SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = __SMLAD(a02, b1, ch_0_out_1);

            col_count--;
        }

        *out_0++ = SG_OUT(ch_0_out_0, out_mult, out_shift);
        *out_1++ = SG_OUT(ch_0_out_1, out_mult, out_shift);
    }

    out_0 += output_ch;
    return out_0;
}

tinyengine_status convolve_1x1_s8_selfgate(
    const q7_t *input, const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
    const q7_t *kernel, const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
    const int32_t out_offset, const int32_t input_offset, const int32_t out_activation_min,
    const int32_t out_activation_max, q7_t *output, const uint16_t output_x, const uint16_t output_y,
    const uint16_t output_ch, q15_t *runtime_buf, const float mul_scale, const float mul_zero,
    const float mul_output_scale, const float mul_output_zero) {
    if (input_ch % 4 != 0) {
        return convolve_1x1_s8_selfgate_scalar(input, input_x, input_y, input_ch, kernel, bias, output_shift,
                                               output_mult, out_offset, input_offset, out_activation_min,
                                               out_activation_max, output, output_x, output_y, output_ch,
                                               runtime_buf, mul_scale, mul_zero, mul_output_scale,
                                               mul_output_zero);
    }

    q7_t lut[256];
    selfgate_lut(lut, mul_scale, mul_zero, mul_output_scale, mul_output_zero);

    int32_t i_element;
    q15_t *two_column_buffer = runtime_buf; /* partial (two columns) im2col buffer */
    q7_t *out = output;
    const int32_t num_elements = output_x * output_y;
    const int channel_div4 = (input_ch >> 2);

    const int16_t inoff16 = input_offset;
    q31_t offset_q15x2 = __PKHBT(inoff16, inoff16, 16);

    for (i_element = 0; i_element < num_elements / 2; i_element++) {
        const q7_t *src = &input[i_element * input_ch * 2];
        q15_t *dst = two_column_buffer;

        q31_t in_q7x4;
        q31_t out_q15x2_1;
        q31_t out_q15x2_2;

        int cnt = channel_div4; /* two columns */
        while (cnt > 0) {
            q7_q15_offset_reordered_ele(src, dst)
            q7_q15_offset_reordered_ele(src, dst)
            cnt--;
        }

        out = mat_mult_selfgate(kernel, two_column_buffer, output_ch, output_shift, output_mult, (q7_t)out_offset,
                                out_activation_min, out_activation_max, input_ch * DIM_KER_Y * DIM_KER_X, bias,
                                out, lut);
    }

    /* odd column left over */
    if (num_elements & 0x1) {
        int32_t i_ch_out;
        const q7_t *ker_a = kernel;
        const q7_t *src = &input[(num_elements - 1) * input_ch];
        q15_t *dst = two_column_buffer;

        q31_t in_q7x4;
        q31_t out_q15x2_1;
        q31_t out_q15x2_2;

        int cnt = channel_div4;
        while (cnt > 0) {
            q7_q15_offset_reordered_ele(src, dst)
            cnt--;
        }

        for (i_ch_out = 0; i_ch_out < output_ch; i_ch_out++) {
            q31_t sum = bias[i_ch_out];
            const q15_t *ip_as_col = runtime_buf;
            uint16_t col_count = (input_ch * DIM_KER_X * DIM_KER_Y) >> 2;

            while (col_count) {
                q31_t ker_a1, ker_a2;
                q31_t in_b1, in_b2;
                ker_a = read_and_pad_reordered(ker_a, &ker_a1, &ker_a2);

                in_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
                sum = __SMLAD(ker_a1, in_b1, sum);
                in_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
                sum = __SMLAD(ker_a2, in_b2, sum);

                col_count--;
            }

            sum = arm_nn_requantize(sum, output_mult[i_ch_out], output_shift[i_ch_out]);
            sum += out_offset;
            sum = MAX(sum, out_activation_min);
            sum = MIN(sum, out_activation_max);
            *out++ = lut[(sum + 128) & 0xFF];
        }
    }

    return STATE_SUCCESS;
}
