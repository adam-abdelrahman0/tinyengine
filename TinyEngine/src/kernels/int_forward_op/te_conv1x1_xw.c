/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   te_conv1x1_xw.c
 *
 * 1x1 convolution over pairs of output pixels with the weights pre-expanded to q15.
 *
 * The CMSIS-style micro-kernel (te_mat_mult_s8.c) re-expands every int8 weight word to two q15x2 words
 * (SXTB16, SXTB16 ROR 8) each time it is used, i.e. once per pair of pixels. Here a block of weight rows is
 * expanded once into a workspace (the idle tail of sbuf, DTCMRAM, no memory of its own) and reused for every pixel pair, so the inner loop is only loads and
 * SMLAD: per 4 input channels, 2 output channels x 2 pixels = 16 MACs = 4 weight words + 4 activation words
 * (two LDRD each) + 8 SMLAD, against 2 + 4 loads + 4 SXTB16 + 8 SMLAD before.
 *
 * Layouts (words are q15x2; "1"/"2" are the channel pairs (0,2) and (1,3) of a group of 4, as in CMSIS):
 *   weights, per row pair and group of 4 input channels:  r.w1  (r+1).w1  r.w2  (r+1).w2
 *   activations, per group of 4:                          px0.x1  px1.x1  px0.x2  px1.x2
 * Integer result is identical to convolve_1x1_s8() with te_mat_mult_kernel_s8_s16_reordered_s8(): same products,
 * same sums (just a different association order of exact int32 additions), same te_requantize_s8() epilogue.
 *
 * Only for layers where te_s8_fast_ok() holds, input_ch % 4 == 0 and output_ch even; the caller checks.
 * -------------------------------------------------------------------- */

#ifdef TE_DTCM

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "te_requant.h"
#include "tinyengine_function.h"

/* number of weight rows (even, >= 2) whose expanded copy fits the workspace; 0 if even two rows do not */
int te_conv1x1_xw_rows(const size_t wspace_bytes, const uint16_t input_ch)
{
    return (int)((wspace_bytes / (2U * input_ch)) & ~(size_t)1);
}

/* computes pixel pairs [0, num_pairs) of a 1x1 conv; pixel p reads input + p * input_ch, writes output + p * output_ch.
 * runtime_buf needs 2 * input_ch q15 (the interleaved activation columns). */
void te_conv1x1_pairs_xw(const q7_t *input, const uint16_t input_ch, const q7_t *kernel, const int32_t *bias,
                         const int32_t *output_shift, const int32_t *output_mult, const int32_t out_offset,
                         const int32_t input_offset, q7_t *output, const int32_t num_pairs, const uint16_t output_ch,
                         q15_t *runtime_buf, q31_t *wspace, const size_t wspace_bytes)
{
    const int groups = input_ch >> 2;
    const int block_rows = te_conv1x1_xw_rows(wspace_bytes, input_ch);
    const int16_t inoff16 = input_offset;
    const q31_t offset_q15x2 = __PKHBT(inoff16, inoff16, 16);
    q31_t *const wblk = wspace;

    for (int row0 = 0; row0 < output_ch; row0 += block_rows)
    {
        const int rows = (output_ch - row0) < block_rows ? (output_ch - row0) : block_rows;

        /* expand rows [row0, row0 + rows) of the weights */
        {
            q31_t *wd = wblk;
            for (int r = row0; r < row0 + rows; r += 2)
            {
                const q7_t *k0 = kernel + (size_t)r * input_ch;
                const q7_t *k1 = k0 + input_ch;
                for (int g = 0; g < groups; g++)
                {
                    const q31_t v0 = arm_nn_read_q7x4(k0 + 4 * g);
                    const q31_t v1 = arm_nn_read_q7x4(k1 + 4 * g);
                    wd[0] = __SXTB16(v0);
                    wd[1] = __SXTB16(v1);
                    wd[2] = __SXTB16(__ROR(v0, 8));
                    wd[3] = __SXTB16(__ROR(v1, 8));
                    wd += 4;
                }
            }
        }

        for (int p = 0; p < num_pairs; p++)
        {
            /* interleaved activation columns for pixels 2p and 2p + 1 */
            {
                const q7_t *s0 = input + (size_t)p * 2 * input_ch;
                const q7_t *s1 = s0 + input_ch;
                q31_t *dst = (q31_t *)runtime_buf;
                for (int g = 0; g < groups; g++)
                {
                    const q31_t u0 = arm_nn_read_q7x4(s0 + 4 * g);
                    const q31_t u1 = arm_nn_read_q7x4(s1 + 4 * g);
                    dst[0] = __SADD16(__SXTB16(u0), offset_q15x2);
                    dst[1] = __SADD16(__SXTB16(u1), offset_q15x2);
                    dst[2] = __SADD16(__SXTB16(__ROR(u0, 8)), offset_q15x2);
                    dst[3] = __SADD16(__SXTB16(__ROR(u1, 8)), offset_q15x2);
                    dst += 4;
                }
            }

            q7_t *out_0 = output + (size_t)p * 2 * output_ch + row0;
            q7_t *out_1 = out_0 + output_ch;
            const uint64_t *w = (const uint64_t *)wblk;
            for (int r = row0; r < row0 + rows; r += 2)
            {
                const uint64_t *b = (const uint64_t *)runtime_buf;
                const uint64_t *const b_end = b + 2 * groups;
                q31_t acc00 = bias[r], acc01 = bias[r];
                q31_t acc10 = bias[r + 1], acc11 = bias[r + 1];
                /* 64-bit loads (the workspace and the scratch buffer are 8-byte aligned) so the compiler emits LDRD:
                 * 4 loads + 8 SMLAD per group of 4 channels (groups >= 1: input_ch >= 4) */
                do
                {
                    const uint64_t wa = w[0], ba = b[0];
                    acc00 = __SMLAD((q31_t)wa, (q31_t)ba, acc00);
                    acc01 = __SMLAD((q31_t)wa, (q31_t)(ba >> 32), acc01);
                    acc10 = __SMLAD((q31_t)(wa >> 32), (q31_t)ba, acc10);
                    acc11 = __SMLAD((q31_t)(wa >> 32), (q31_t)(ba >> 32), acc11);
                    const uint64_t wb = w[1], bb = b[1];
                    acc00 = __SMLAD((q31_t)wb, (q31_t)bb, acc00);
                    acc01 = __SMLAD((q31_t)wb, (q31_t)(bb >> 32), acc01);
                    acc10 = __SMLAD((q31_t)(wb >> 32), (q31_t)bb, acc10);
                    acc11 = __SMLAD((q31_t)(wb >> 32), (q31_t)(bb >> 32), acc11);
                    w += 2;
                    b += 2;
                } while (b != b_end);
                *out_0++ = (q7_t)te_requantize_s8(acc00, output_mult[r], output_shift[r], out_offset);
                *out_1++ = (q7_t)te_requantize_s8(acc01, output_mult[r], output_shift[r], out_offset);
                *out_0++ = (q7_t)te_requantize_s8(acc10, output_mult[r + 1], output_shift[r + 1], out_offset);
                *out_1++ = (q7_t)te_requantize_s8(acc11, output_mult[r + 1], output_shift[r + 1], out_offset);
            }
        }
    }
}

#endif /* TE_DTCM */
