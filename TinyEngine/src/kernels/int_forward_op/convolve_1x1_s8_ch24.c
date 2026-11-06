/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   convolve_1x1_s8_ch24.c
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

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "img2col_element.h"
#include "tinyengine_function.h"

#define DIM_KER_X (1U)
#define DIM_KER_Y (1U)

/* Widens the weights once per layer call (not per pixel pair): for each pair of output channels and each group of 4
 * input channels, the four q15x2 words a01 a02 a11 a12 that mat_mult_kernel_s8_s16_preprocessed_ch24() consumes.
 * Needs output_ch * num_col_a * 2 bytes (1536 for 32 x 24); the caller lends it the idle tail of sbuf. */
static void preprocess_weights(
	const q7_t *raw_weights,
	q31_t *prepacked_weights,
	uint16_t output_ch,
	uint16_t num_col_a)
{
	q31_t *dest_ptr = prepacked_weights;

// Process output channels in pairs (matching row_count = output_ch / 2)
	for (uint16_t ch = 0; ch < output_ch; ch += 2)
	{
// Point to the start of Channel N and Channel N+1 in the raw weight matrix
		const q7_t *row_n = raw_weights + ch * num_col_a;
		const q7_t *row_n1 = raw_weights + (ch + 1) * num_col_a;

// Iterate through columns in blocks of 4 (since read_and_pad reads 4 q7 bytes at a time)
		for (uint16_t col = 0; col < num_col_a; col += 4)
		{
			q31_t a01, a02, a11, a12;

		// 1. Process Channel N (reads 4 bytes, packs into two q31_t words: a01, a02)
			row_n = read_and_pad_reordered(row_n, &a01, &a02);
			*dest_ptr++ = a01;
			*dest_ptr++ = a02;

// 2. Process Channel N+1 (reads 4 bytes, packs into two q31_t words: a11, a12)
			row_n1 = read_and_pad_reordered(row_n1, &a11, &a12);
			*dest_ptr++ = a11;
			*dest_ptr++ = a12;
		}
}
}

tinyengine_status convolve_1x1_s8_ch24(const q7_t *input, const uint16_t input_x,
		const uint16_t input_y, const uint16_t input_ch, const q7_t *kernel,
		const int32_t *bias, const int32_t *output_shift,
		const int32_t *output_mult, const int32_t out_offset,
		const int32_t input_offset, const int32_t out_activation_min,
		const int32_t out_activation_max, q7_t *output, const uint16_t output_x,
		const uint16_t output_y, const uint16_t output_ch, q15_t *runtime_buf) {
	int32_t i_element;
	(void) input_x;
	(void) input_y;

	/* Partial(two columns) im2col buffer */
	q15_t *two_column_buffer = runtime_buf;
	q7_t *out = output;
	const int32_t num_elements = output_x * output_y;
	const int channel_div4 = (input_ch >> 2);

	/* pre-widened weights: the preprocessed kernel is unrolled for 24 input channels and needs an even output_ch. Its
	 * workspace is the part of sbuf after this call's two im2col columns (4 * input_ch bytes), which nothing else uses
	 * while the layer runs; without it (no TE_DTCM, or too small) the original kernel runs on the raw weights. */
	int use_prepacked = 0;
#ifdef TE_DTCM
	const size_t wspace_off = 4 * (size_t) input_ch;
	const size_t wspace_need = (size_t) output_ch * input_ch * 2;
	q31_t *const prepacked = (q31_t *) ((q7_t *) runtime_buf + wspace_off);
	if (input_ch == 24 && output_ch % 2 == 0 && runtime_buf == te_sbuf_base
			&& (size_t) SBuffer_size >= wspace_off + wspace_need) {
		preprocess_weights(kernel, prepacked, output_ch, input_ch);
		use_prepacked = 1;
	}
#endif

	const int16_t inoff16 = input_offset;
	q31_t offset_q15x2 = __PKHBT(inoff16, inoff16, 16);

	for (i_element = 0; i_element < num_elements / 2; i_element++) {
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

#ifdef TE_DTCM
		if (use_prepacked) {
			out = mat_mult_kernel_s8_s16_preprocessed_ch24(prepacked,
					two_column_buffer, output_ch, output_shift, output_mult,
					(q7_t) out_offset, out_activation_min,
					out_activation_max, input_ch * DIM_KER_Y * DIM_KER_X,
					bias, out);
			continue;
		}
#endif
		out = mat_mult_kernel_s8_s16_reordered_ch24(kernel,
				two_column_buffer, output_ch, output_shift, output_mult,
				(q7_t) out_offset, out_activation_min,
				out_activation_max, input_ch * DIM_KER_Y * DIM_KER_X,
				bias, out);
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
