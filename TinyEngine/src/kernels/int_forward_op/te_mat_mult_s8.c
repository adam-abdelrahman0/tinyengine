/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   te_mat_mult_s8.c
 *
 * arm_nn_mat_mult_kernel_s8_s16_reordered() (CMSIS-NN, 2 output channels x 2 columns,
 * SMLAD) with a cheaper epilogue: te_requantize_s8() -- right-shift-only requantize
 * and one SSAT instead of the compare-and-conditional-move clamp. The multiply loop
 * is byte-for-byte the CMSIS one. Only valid when te_s8_fast_ok() is true for the
 * layer (every per-channel shift <= 0, activation range -128..127); callers check
 * that and otherwise use the CMSIS routine. About 14 instructions fewer per output.
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "te_requant.h"
#include "tinyengine_function.h"

q7_t *te_mat_mult_kernel_s8_s16_reordered_s8(const q7_t *input_a,
                                              const q15_t *input_b,
                                              const uint16_t output_ch,
                                              const int32_t *out_shift,
                                              const int32_t *out_mult,
                                              const int32_t out_offset,
                                              const uint16_t num_col_a,
                                              const int32_t *const output_bias,
                                              q7_t *out_0)
{
    /* set up the second output pointers */
    q7_t *out_1 = out_0 + output_ch;
    const int32_t *bias = output_bias;

    uint16_t row_count = output_ch / 2;
    const q7_t *ip_a0 = input_a;
    /* this loop over rows in A */
    while (row_count)
    {
        /* setup pointers for B */
        const q15_t *ip_b0 = input_b;
        const q15_t *ip_b1 = ip_b0 + num_col_a;

        /* align the second pointer for A */
        const q7_t *ip_a1 = ip_a0 + num_col_a;

        /* Init accumulator with bias for channel N and N + 1 */
        q31_t ch_0_out_0 = *bias;
        q31_t ch_0_out_1 = *bias++;
        q31_t ch_1_out_0 = *bias;
        q31_t ch_1_out_1 = *bias++;

        uint16_t col_count = num_col_a / 4;
        /* accumulate over the vector */
        while (col_count)
        {
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
        } /* while over col_count */

        ch_0_out_0 = te_requantize_s8(ch_0_out_0, *out_mult, *out_shift, out_offset);
        *out_0++ = (q7_t)ch_0_out_0;

        ch_0_out_1 = te_requantize_s8(ch_0_out_1, *out_mult, *out_shift, out_offset);
        *out_1++ = (q7_t)ch_0_out_1;
        out_mult++;
        out_shift++;

        ch_1_out_0 = te_requantize_s8(ch_1_out_0, *out_mult, *out_shift, out_offset);
        *out_0++ = (q7_t)ch_1_out_0;

        ch_1_out_1 = te_requantize_s8(ch_1_out_1, *out_mult, *out_shift, out_offset);
        *out_1++ = (q7_t)ch_1_out_1;
        out_mult++;
        out_shift++;

        /* skip row */
        ip_a0 += num_col_a;
        row_count--;
    }

    if (output_ch & 1)
    {
        /* setup pointers for B */
        const q15_t *ip_b0 = input_b;
        const q15_t *ip_b1 = ip_b0 + num_col_a;

        /* Init accumulator with bias for channel N + 1 */
        q31_t ch_0_out_0 = *bias;
        q31_t ch_0_out_1 = ch_0_out_0;

        int32_t col_count = num_col_a / 4;
        while (col_count)
        {
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
        } /* while over col_count */

        ch_0_out_0 = te_requantize_s8(ch_0_out_0, *out_mult, *out_shift, out_offset);
        *out_0++ = (q7_t)ch_0_out_0;

        ch_0_out_1 = te_requantize_s8(ch_0_out_1, *out_mult, *out_shift, out_offset);
        *out_1++ = (q7_t)ch_0_out_1;
    }

    out_0 += output_ch;

    /* return the new output pointer with offset */
    return out_0;
}
