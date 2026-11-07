/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   convolve_1x1_s8.c
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

#include <string.h>
#include "arm_nnfunctions.h"
#include "img2col_element.h"
#include "te_requant.h"
#include "tinyengine_function.h"

#define DIM_KER_X (1U)
#define DIM_KER_Y (1U)

tinyengine_status convolve_1x1_s8(const q7_t *input, const uint16_t input_x,
		const uint16_t input_y, const uint16_t input_ch, const q7_t *kernel,
		const int32_t *bias, const int32_t *output_shift,
		const int32_t *output_mult, const int32_t out_offset,
		const int32_t input_offset, const int32_t out_activation_min,
		const int32_t out_activation_max, q7_t *output, const uint16_t output_x,
		const uint16_t output_y, const uint16_t output_ch, q15_t *runtime_buf) {
	if (input_ch % 4 != 0 || input_ch % 2 != 0) {
		return PARAM_NO_SUPPORT;
	}

	int32_t i_element;
	(void) input_x;
	(void) input_y;

	/* Partial(two columns) im2col buffer */
	q15_t *two_column_buffer = runtime_buf;
	q7_t *out = output;
	const int32_t num_elements = output_x * output_y;
	const int channel_div4 = (input_ch >> 2);
	/* right-shift-only requantize + SSAT clamp when the layer allows it (see te_requant.h) */
	const int fast = te_s8_fast_ok(output_shift, output_ch, out_activation_min, out_activation_max);
	int pairs_done = 0;
#ifdef TE_DTCM
	/* weights pre-expanded to q15 (te_conv1x1_xw.c), in the idle part of sbuf, for layers with enough pixel pairs to repay the
	 * expansion. If the whole layer's expanded weights fit, they are expanded once; if not, pixels are taken in tiles
	 * (the tile's inputs are read before its outputs are written, so a layer whose output overwrites its input is fine). */
	/* the workspace is the part of sbuf after this layer's two im2col columns (4 * input_ch bytes), if runtime_buf is sbuf */
	const size_t wspace_off = 4 * (size_t) input_ch;
	const size_t wspace_bytes = (runtime_buf == te_sbuf_base && (size_t) SBuffer_size > wspace_off) ? ((size_t) SBuffer_size - wspace_off) & ~(size_t) 7 : 0;
	q31_t *const wspace = (q31_t *) ((q7_t *) runtime_buf + wspace_off);
	const int xw_rows = te_conv1x1_xw_rows(wspace_bytes, input_ch);
	/* tiled mode uses all of sbuf, im2col columns included */
	const size_t tile_bytes = (runtime_buf == te_sbuf_base) ? ((size_t) SBuffer_size & ~(size_t) 7) : 0;
	if (fast && output_ch % 2 == 0 && num_elements >= 16 && xw_rows >= output_ch) {
		te_conv1x1_pairs_xw(input, input_ch, kernel, bias, output_shift, output_mult, out_offset, input_offset, output,
				num_elements / 2, output_ch, runtime_buf, wspace, wspace_bytes);
		pairs_done = num_elements / 2;
		out += (size_t) pairs_done * 2 * output_ch;
	} else if (fast && output_ch % 2 == 0 && num_elements >= 16 && te_conv1x1_xw_tiled_ok(tile_bytes, num_elements / 2, input_ch)) {
		te_conv1x1_pairs_xw_tiled(input, input_ch, kernel, bias, output_shift, output_mult, out_offset, input_offset, output,
				num_elements / 2, output_ch, (q31_t *) runtime_buf, tile_bytes);
		pairs_done = num_elements / 2;
		out += (size_t) pairs_done * 2 * output_ch;
	} else {
		const size_t wbytes = (size_t) input_ch * output_ch;
		if (wbytes > TE_WBUF_MIN && wbytes <= wspace_bytes) {
			memcpy(wspace, kernel, wbytes);  /* tiny against the layer: weights are re-read for every pair of columns */
			kernel = (const q7_t *) wspace;
		}
	}
#endif

	const int16_t inoff16 = input_offset;
	q31_t offset_q15x2 = __PKHBT(inoff16, inoff16, 16);

	for (i_element = pairs_done; i_element < num_elements / 2; i_element++) {
		/* Fill buffer for partial im2col - two columns at a time */
		q7_t *src = &input[i_element * input_ch * 2];
		q15_t *dst = two_column_buffer;

		//use variables
		q31_t in_q7x4;
		q31_t in_q15x2_1;
		q31_t in_q15x2_2;
		q31_t out_q15x2_1;
		q31_t out_q15x2_2;

		int cnt = channel_div4;	//two columns
		while (cnt > 0) {
			q7_q15_offset_reordered_ele(src, dst)
			q7_q15_offset_reordered_ele(src, dst)
			cnt--;
		}

		if (fast) {
			out = te_mat_mult_kernel_s8_s16_reordered_s8(kernel,
					two_column_buffer, output_ch, output_shift, output_mult,
					(q7_t) out_offset, input_ch * DIM_KER_Y * DIM_KER_X,
					bias, out);
		} else {
			out = arm_nn_mat_mult_kernel_s8_s16_reordered(kernel,
					two_column_buffer, output_ch, output_shift, output_mult,
					(q7_t) out_offset, out_activation_min,
					out_activation_max, input_ch * DIM_KER_Y * DIM_KER_X,
					bias, out);
		}
	}

	/* check if there is an odd column left-over for computation */
	if (num_elements & 0x1) {
		int32_t i_ch_out;
		const q7_t *ker_a = kernel;
		q7_t *src = &input[(num_elements - 1) * input_ch];
		q15_t *dst = two_column_buffer;

		//use variables
		q31_t in_q7x4;
		q31_t in_q15x2_1;
		q31_t in_q15x2_2;
		q31_t out_q15x2_1;
		q31_t out_q15x2_2;

		int cnt = channel_div4;	//two * numof2col columns
		while (cnt > 0) {
			q7_q15_offset_reordered_ele(src, dst)
			cnt--;
		}

		for (i_ch_out = 0; i_ch_out < output_ch; i_ch_out++) {
			q31_t sum = bias[i_ch_out];

			/* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
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

			sum = arm_nn_requantize(sum, output_mult[i_ch_out],
					output_shift[i_ch_out]);
			sum += out_offset;
			sum = MAX(sum, out_activation_min);
			sum = MIN(sum, out_activation_max);
			*out++ = (q7_t) sum;
		}
	}

	/* Return to application */
	return STATE_SUCCESS;
}
