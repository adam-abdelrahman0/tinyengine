/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   arm_nn_mat_mult_kernel3_input3_s8_s16_fast.c
 *
 * arm_nn_mat_mult_kernel3_input3_s8_s16() (3x3, 3 input channels, the stem conv) with
 * the cheaper epilogue of te_requant.h: right-shift-only requantize + one SSAT. Same
 * signature as the original; the activation range is not used (it is the SSAT range).
 * Only valid when te_s8_fast_ok() holds for the layer; the caller checks and otherwise
 * uses the original. Everything before the epilogue is identical.
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "te_requant.h"

q7_t *arm_nn_mat_mult_kernel3_input3_s8_s16_fast(const q7_t *input_a,
        const q15_t *input_b,
        const uint16_t output_ch,
        const int32_t *out_shift,
        const int32_t *out_mult,
        const int32_t out_offset,
        const int16_t activation_min,
        const int16_t activation_max,
        const uint16_t num_col_a,
        const int32_t *const output_bias,
        q7_t *out_0,
		q15_t *kbuf)
{
    (void)activation_min;
    (void)activation_max;
    /* set up the second output pointers */
    q7_t *out_1 = out_0 + output_ch;
    const int32_t *bias = output_bias;

    uint16_t row_count = output_ch / 2;
    const q15_t *ksrc = &kbuf[0];
    /* this loop over rows in A */
    while (row_count)
    {
        /* setup pointers for B */
        const q15_t *ip_b0 = input_b;
        const q15_t *ip_b1 = ip_b0 + num_col_a;
        const q31_t *ip31_b0 = ip_b0;
        const q31_t *ip31_b1 = ip_b1;

        /* align the second pointer for A */
        const q15_t *ksrc2 = ksrc + 27;
        q31_t *ksrc_31 = ksrc;
        q31_t *ksrc2_31 = ksrc2;

        /* Init accumulator with bias for channel N and N + 1 */
        q31_t ch_0_out_0 = *bias;
        q31_t ch_0_out_1 = *bias++;
        q31_t ch_1_out_0 = *bias;
        q31_t ch_1_out_1 = *bias++;

		//------------------4
		q31_t a01, a02, a11, a12;
		q31_t b0 = arm_nn_read_q15x2_ia(&ip_b0);
		q31_t b1 = arm_nn_read_q15x2_ia(&ip_b1);

		ch_0_out_0 = __SMLAD(ksrc_31[0], b0, ch_0_out_0);
		ch_0_out_1 = __SMLAD(ksrc_31[0], b1, ch_0_out_1);
		ch_1_out_0 = __SMLAD(ksrc2_31[0], b0, ch_1_out_0);
		ch_1_out_1 = __SMLAD(ksrc2_31[0], b1, ch_1_out_1);

		b0 = arm_nn_read_q15x2_ia(&ip_b0);
		b1 = arm_nn_read_q15x2_ia(&ip_b1);

		ch_0_out_0 = __SMLAD(ksrc_31[1], b0, ch_0_out_0);
		ch_0_out_1 = __SMLAD(ksrc_31[1], b1, ch_0_out_1);
		ch_1_out_0 = __SMLAD(ksrc2_31[1], b0, ch_1_out_0);
		ch_1_out_1 = __SMLAD(ksrc2_31[1], b1, ch_1_out_1);

		//------------------8
		b0 = arm_nn_read_q15x2_ia(&ip_b0);
		b1 = arm_nn_read_q15x2_ia(&ip_b1);

		ch_0_out_0 = __SMLAD(ksrc_31[2], b0, ch_0_out_0);
		ch_0_out_1 = __SMLAD(ksrc_31[2], b1, ch_0_out_1);
		ch_1_out_0 = __SMLAD(ksrc2_31[2], b0, ch_1_out_0);
		ch_1_out_1 = __SMLAD(ksrc2_31[2], b1, ch_1_out_1);

		b0 = arm_nn_read_q15x2_ia(&ip_b0);
		b1 = arm_nn_read_q15x2_ia(&ip_b1);

		ch_0_out_0 = __SMLAD(ksrc_31[3], b0, ch_0_out_0);
		ch_0_out_1 = __SMLAD(ksrc_31[3], b1, ch_0_out_1);
		ch_1_out_0 = __SMLAD(ksrc2_31[3], b0, ch_1_out_0);
		ch_1_out_1 = __SMLAD(ksrc2_31[3], b1, ch_1_out_1);

		//------------------12
		b0 = arm_nn_read_q15x2_ia(&ip_b0);
		b1 = arm_nn_read_q15x2_ia(&ip_b1);

		ch_0_out_0 = __SMLAD(ksrc_31[4], b0, ch_0_out_0);
		ch_0_out_1 = __SMLAD(ksrc_31[4], b1, ch_0_out_1);
		ch_1_out_0 = __SMLAD(ksrc2_31[4], b0, ch_1_out_0);
		ch_1_out_1 = __SMLAD(ksrc2_31[4], b1, ch_1_out_1);

		b0 = arm_nn_read_q15x2_ia(&ip_b0);
		b1 = arm_nn_read_q15x2_ia(&ip_b1);

		ch_0_out_0 = __SMLAD(ksrc_31[5], b0, ch_0_out_0);
		ch_0_out_1 = __SMLAD(ksrc_31[5], b1, ch_0_out_1);
		ch_1_out_0 = __SMLAD(ksrc2_31[5], b0, ch_1_out_0);
		ch_1_out_1 = __SMLAD(ksrc2_31[5], b1, ch_1_out_1);

		//------------------16
		b0 = arm_nn_read_q15x2_ia(&ip_b0);
		b1 = arm_nn_read_q15x2_ia(&ip_b1);
		ch_0_out_0 = __SMLAD(ksrc_31[6], b0, ch_0_out_0);
		ch_0_out_1 = __SMLAD(ksrc_31[6], b1, ch_0_out_1);
		ch_1_out_0 = __SMLAD(ksrc2_31[6], b0, ch_1_out_0);
		ch_1_out_1 = __SMLAD(ksrc2_31[6], b1, ch_1_out_1);

		b0 = arm_nn_read_q15x2_ia(&ip_b0);
		b1 = arm_nn_read_q15x2_ia(&ip_b1);

		ch_0_out_0 = __SMLAD(ksrc_31[7], b0, ch_0_out_0);
		ch_0_out_1 = __SMLAD(ksrc_31[7], b1, ch_0_out_1);
		ch_1_out_0 = __SMLAD(ksrc2_31[7], b0, ch_1_out_0);
		ch_1_out_1 = __SMLAD(ksrc2_31[7], b1, ch_1_out_1);

		//------------------20
		b0 = arm_nn_read_q15x2_ia(&ip_b0);
		b1 = arm_nn_read_q15x2_ia(&ip_b1);
		ch_0_out_0 = __SMLAD(ksrc_31[8], b0, ch_0_out_0);
		ch_0_out_1 = __SMLAD(ksrc_31[8], b1, ch_0_out_1);
		ch_1_out_0 = __SMLAD(ksrc2_31[8], b0, ch_1_out_0);
		ch_1_out_1 = __SMLAD(ksrc2_31[8], b1, ch_1_out_1);

		b0 = arm_nn_read_q15x2_ia(&ip_b0);
		b1 = arm_nn_read_q15x2_ia(&ip_b1);

		ch_0_out_0 = __SMLAD(ksrc_31[9], b0, ch_0_out_0);
		ch_0_out_1 = __SMLAD(ksrc_31[9], b1, ch_0_out_1);
		ch_1_out_0 = __SMLAD(ksrc2_31[9], b0, ch_1_out_0);
		ch_1_out_1 = __SMLAD(ksrc2_31[9], b1, ch_1_out_1);

		//------------------24
		b0 = arm_nn_read_q15x2_ia(&ip_b0);
		b1 = arm_nn_read_q15x2_ia(&ip_b1);
		ch_0_out_0 = __SMLAD(ksrc_31[10], b0, ch_0_out_0);
		ch_0_out_1 = __SMLAD(ksrc_31[10], b1, ch_0_out_1);
		ch_1_out_0 = __SMLAD(ksrc2_31[10], b0, ch_1_out_0);
		ch_1_out_1 = __SMLAD(ksrc2_31[10], b1, ch_1_out_1);

		b0 = arm_nn_read_q15x2_ia(&ip_b0);
		b1 = arm_nn_read_q15x2_ia(&ip_b1);

		ch_0_out_0 = __SMLAD(ksrc_31[11], b0, ch_0_out_0);
		ch_0_out_1 = __SMLAD(ksrc_31[11], b1, ch_0_out_1);
		ch_1_out_0 = __SMLAD(ksrc2_31[11], b0, ch_1_out_0);
		ch_1_out_1 = __SMLAD(ksrc2_31[11], b1, ch_1_out_1);

		//------------------25,26,27
		b0 = arm_nn_read_q15x2_ia(&ip_b0);
		b1 = arm_nn_read_q15x2_ia(&ip_b1);
		ch_0_out_0 = __SMLAD(ksrc_31[12], b0, ch_0_out_0);
		ch_0_out_1 = __SMLAD(ksrc_31[12], b1, ch_0_out_1);
		ch_1_out_0 = __SMLAD(ksrc2_31[12], b0, ch_1_out_0);
		ch_1_out_1 = __SMLAD(ksrc2_31[12], b1, ch_1_out_1);
		q15_t _b0 = *ip_b0++;
		q15_t _b1 = *ip_b1++;

		ch_0_out_0 += ksrc[26] * _b0;
		ch_0_out_1 += ksrc[26] * _b1;
		ch_1_out_0 += ksrc2[26] * _b0;
		ch_1_out_1 += ksrc2[26] * _b1;

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
        ksrc += 54;
        row_count--;
    }

    out_0 += output_ch;

    /* return the new output pointer with offset */
    return out_0;
}
