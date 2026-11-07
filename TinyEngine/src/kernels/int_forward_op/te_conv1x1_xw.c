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
 *
 * Two entry points. te_conv1x1_pairs_xw(): the layer's expanded weights fit the workspace, so they are expanded once and
 * every pixel pair is computed against them. te_conv1x1_pairs_xw_tiled(): they do not, so pixels are taken in tiles of
 * up to TE_XW_TILE pairs: the tile's activations are converted once into the workspace, then for each block of weight rows
 * (expanded once per tile, not once per pair) the whole tile is computed. The weights come from flash once per tile
 * instead of once per pair, and because a tile's inputs are fully read before any of its outputs are written, a layer
 * whose output overwrites its input (--pointwise-overwrite) is safe.
 * -------------------------------------------------------------------- */

#ifdef TE_DTCM

#include <string.h>
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "te_requant.h"
#include "tinyengine_function.h"

/* Tunables (all -D at build time; the defaults are what ships). tinyengine_bench.py passes TE_EXTRA_DEFINES through, and
 * Data_Scripts/sweeps/ measures each one on the board.
 *   TE_XW_TILE       pixel pairs per tile: the weight expansion costs 1/TE_XW_TILE of the multiply work (default 16)
 *   TE_XW_TILE_DIV   the tile buffer may take 1/TE_XW_TILE_DIV of the workspace (default 2)
 *   TE_XW_NO_QUAD    use the 2 x 2 pair kernel only (no 2 x 4 block); also disables the stem path
 *   TE_XW_GROUP_UNROLL  groups of 4 input channels per quad-loop iteration, 1 or 2 (default 2) */
#ifndef TE_XW_TILE
#define TE_XW_TILE 16
#endif
#ifndef TE_XW_TILE_DIV
#define TE_XW_TILE_DIV 2
#endif
#ifndef TE_XW_GROUP_UNROLL
#define TE_XW_GROUP_UNROLL 2
#endif

/* number of weight rows (even, >= 2) whose expanded copy fits the workspace; 0 if even two rows do not */
int te_conv1x1_xw_rows(const size_t wspace_bytes, const uint16_t input_ch)
{
    return (int)((wspace_bytes / (2U * input_ch)) & ~(size_t)1);
}

/* pixel pairs per tile for a workspace, 0 if the tile buffer plus two weight rows do not fit */
static int xw_tile_pairs(const size_t wspace_bytes, const int num_pairs, const uint16_t input_ch)
{
    const size_t pair_bytes = 4U * input_ch;
    size_t tp = (wspace_bytes / TE_XW_TILE_DIV) / pair_bytes; /* at most 1/TE_XW_TILE_DIV of the workspace for the tile */
    if (tp > TE_XW_TILE) tp = TE_XW_TILE;
    if (tp > (size_t)num_pairs) tp = num_pairs;
    if (tp < 1 || wspace_bytes < tp * pair_bytes + 2U * 2U * input_ch) return 0;
    return (int)tp;
}

int te_conv1x1_xw_tiled_ok(const size_t wspace_bytes, const int num_pairs, const uint16_t input_ch)
{
    return xw_tile_pairs(wspace_bytes, num_pairs, input_ch) > 0;
}

/* expand weight rows [row0, row0 + rows) (rows even) into wd, 4 words per row pair and group of 4 input channels */
static inline void xw_expand_rows(q31_t *wd, const q7_t *kernel, const int row0, const int rows, const int groups,
                                  const uint16_t input_ch)
{
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

/* interleaved, offset-added activation columns for the two pixels whose inputs start at s0 and s0 + input_ch */
static inline void xw_convert_pair(q31_t *dst, const q7_t *s0, const int groups, const uint16_t input_ch,
                                   const q31_t offset_q15x2)
{
    const q7_t *s1 = s0 + input_ch;
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

/* output rows [row0, row0 + rows) of one pixel pair: wblk holds the expanded rows, b the pair's converted activations.
 * out_0 / out_1 point at row0 of the two pixels' outputs. */
static inline void xw_compute_pair(const q31_t *wblk, const q31_t *bcol, const int groups, const int row0, const int rows,
                                   const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
                                   const int32_t out_offset, q7_t *out_0, q7_t *out_1)
{
    const uint64_t *w = (const uint64_t *)wblk;
    for (int r = row0; r < row0 + rows; r += 2)
    {
        const uint64_t *b = (const uint64_t *)bcol;
        const uint64_t *const b_end = b + 2 * groups;
        q31_t acc00 = bias[r], acc01 = bias[r];
        q31_t acc10 = bias[r + 1], acc11 = bias[r + 1];
        /* 64-bit loads (the workspace is 8-byte aligned) so the compiler emits LDRD:
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

/* four pixels (two pairs) at once: per group of 4 input channels 8 words, p0.x1 p1.x1 p2.x1 p3.x1 p0.x2 p1.x2 p2.x2 p3.x2.
 * A 2-row x 4-pixel block does 32 MACs from 6 LDRD + 16 SMLAD (the 2 x 2 block: 16 MACs from 4 LDRD + 8 SMLAD), because the
 * weight words are shared by four pixels. The Cortex-M7 issues these about one per cycle, so fewer instructions per MAC
 * is the only lever. */
static inline void xw_convert_quad(q31_t *dst, const q7_t *s0, const int groups, const uint16_t input_ch,
                                   const q31_t offset_q15x2)
{
    const q7_t *s1 = s0 + input_ch;
    const q7_t *s2 = s1 + input_ch;
    const q7_t *s3 = s2 + input_ch;
    for (int g = 0; g < groups; g++)
    {
        const q31_t u0 = arm_nn_read_q7x4(s0 + 4 * g);
        const q31_t u1 = arm_nn_read_q7x4(s1 + 4 * g);
        const q31_t u2 = arm_nn_read_q7x4(s2 + 4 * g);
        const q31_t u3 = arm_nn_read_q7x4(s3 + 4 * g);
        dst[0] = __SADD16(__SXTB16(u0), offset_q15x2);
        dst[1] = __SADD16(__SXTB16(u1), offset_q15x2);
        dst[2] = __SADD16(__SXTB16(u2), offset_q15x2);
        dst[3] = __SADD16(__SXTB16(u3), offset_q15x2);
        dst[4] = __SADD16(__SXTB16(__ROR(u0, 8)), offset_q15x2);
        dst[5] = __SADD16(__SXTB16(__ROR(u1, 8)), offset_q15x2);
        dst[6] = __SADD16(__SXTB16(__ROR(u2, 8)), offset_q15x2);
        dst[7] = __SADD16(__SXTB16(__ROR(u3, 8)), offset_q15x2);
        dst += 8;
    }
}

/* output rows [row0, row0 + rows) of four pixels (the converted quad in bcol); out points at channel 0 of the first pixel */
static inline void xw_compute_quad(const q31_t *wblk, const q31_t *bcol, const int groups, const int row0, const int rows,
                                   const int32_t *bias, const int32_t *output_shift, const int32_t *output_mult,
                                   const int32_t out_offset, q7_t *out, const int output_ch)
{
    const uint64_t *w = (const uint64_t *)wblk;
    for (int r = row0; r < row0 + rows; r += 2)
    {
        const uint64_t *b = (const uint64_t *)bcol;
        const uint64_t *const b_end = b + 8 * (groups >> 1); /* TE_XW_GROUP_UNROLL 2: two groups per iteration halves the loop overhead */
        q31_t a00 = bias[r], a01 = bias[r], a02 = bias[r], a03 = bias[r];
        q31_t a10 = bias[r + 1], a11 = bias[r + 1], a12 = bias[r + 1], a13 = bias[r + 1];
#define XW_QUAD_GROUP(WO, BO)                                  \
    {                                                          \
        const uint64_t wa = w[WO];                             \
        const q31_t w0 = (q31_t)wa, w1 = (q31_t)(wa >> 32);    \
        uint64_t bv = b[BO];                                   \
        a00 = __SMLAD(w0, (q31_t)bv, a00);                     \
        a01 = __SMLAD(w0, (q31_t)(bv >> 32), a01);             \
        a10 = __SMLAD(w1, (q31_t)bv, a10);                     \
        a11 = __SMLAD(w1, (q31_t)(bv >> 32), a11);             \
        bv = b[BO + 1];                                        \
        a02 = __SMLAD(w0, (q31_t)bv, a02);                     \
        a03 = __SMLAD(w0, (q31_t)(bv >> 32), a03);             \
        a12 = __SMLAD(w1, (q31_t)bv, a12);                     \
        a13 = __SMLAD(w1, (q31_t)(bv >> 32), a13);             \
        const uint64_t wb = w[WO + 1];                         \
        const q31_t w2 = (q31_t)wb, w3 = (q31_t)(wb >> 32);    \
        bv = b[BO + 2];                                        \
        a00 = __SMLAD(w2, (q31_t)bv, a00);                     \
        a01 = __SMLAD(w2, (q31_t)(bv >> 32), a01);             \
        a10 = __SMLAD(w3, (q31_t)bv, a10);                     \
        a11 = __SMLAD(w3, (q31_t)(bv >> 32), a11);             \
        bv = b[BO + 3];                                        \
        a02 = __SMLAD(w2, (q31_t)bv, a02);                     \
        a03 = __SMLAD(w2, (q31_t)(bv >> 32), a03);             \
        a12 = __SMLAD(w3, (q31_t)bv, a12);                     \
        a13 = __SMLAD(w3, (q31_t)(bv >> 32), a13);             \
    }
#if TE_XW_GROUP_UNROLL == 2
        while (b != b_end)
        {
            XW_QUAD_GROUP(0, 0)
            XW_QUAD_GROUP(2, 4)
            w += 4;
            b += 8;
        }
        if (groups & 1)
        {
            XW_QUAD_GROUP(0, 0)
            w += 2;
        }
#else
        for (int g = 0; g < groups; g++)
        {
            XW_QUAD_GROUP(0, 0)
            w += 2;
            b += 4;
        }
#endif
#undef XW_QUAD_GROUP
        const int32_t m0 = output_mult[r], s0 = output_shift[r], m1 = output_mult[r + 1], s1 = output_shift[r + 1];
        q7_t *o = out + r;
        o[0] = (q7_t)te_requantize_s8(a00, m0, s0, out_offset);
        o[1] = (q7_t)te_requantize_s8(a10, m1, s1, out_offset);
        o += output_ch;
        o[0] = (q7_t)te_requantize_s8(a01, m0, s0, out_offset);
        o[1] = (q7_t)te_requantize_s8(a11, m1, s1, out_offset);
        o += output_ch;
        o[0] = (q7_t)te_requantize_s8(a02, m0, s0, out_offset);
        o[1] = (q7_t)te_requantize_s8(a12, m1, s1, out_offset);
        o += output_ch;
        o[0] = (q7_t)te_requantize_s8(a03, m0, s0, out_offset);
        o[1] = (q7_t)te_requantize_s8(a13, m1, s1, out_offset);
    }
}

/* computes pixel pairs [0, num_pairs) of a 1x1 conv; pixel p reads input + p * input_ch, writes output + p * output_ch.
 * runtime_buf needs 2 * input_ch q15 (the interleaved activation columns). The expanded weights of one block of rows
 * (te_conv1x1_xw_rows()) must fit wspace; with several blocks the output must not overlap the input. */
void te_conv1x1_pairs_xw(const q7_t *input, const uint16_t input_ch, const q7_t *kernel, const int32_t *bias,
                         const int32_t *output_shift, const int32_t *output_mult, const int32_t out_offset,
                         const int32_t input_offset, q7_t *output, const int32_t num_pairs, const uint16_t output_ch,
                         q15_t *runtime_buf, q31_t *wspace, const size_t wspace_bytes)
{
    const int groups = input_ch >> 2;
    const int block_rows = te_conv1x1_xw_rows(wspace_bytes, input_ch);
    const int16_t inoff16 = input_offset;
    const q31_t offset_q15x2 = __PKHBT(inoff16, inoff16, 16);

    for (int row0 = 0; row0 < output_ch; row0 += block_rows)
    {
        const int rows = (output_ch - row0) < block_rows ? (output_ch - row0) : block_rows;
        xw_expand_rows(wspace, kernel, row0, rows, groups, input_ch);

        int p = 0;
#ifndef TE_XW_NO_QUAD
        for (; p + 2 <= num_pairs; p += 2)
        {
            xw_convert_quad((q31_t *)runtime_buf, input + (size_t)p * 2 * input_ch, groups, input_ch, offset_q15x2);
            xw_compute_quad(wspace, (const q31_t *)runtime_buf, groups, row0, rows, bias, output_shift, output_mult,
                            out_offset, output + (size_t)p * 2 * output_ch, output_ch);
        }
#endif
        for (; p < num_pairs; p++)
        {
            xw_convert_pair((q31_t *)runtime_buf, input + (size_t)p * 2 * input_ch, groups, input_ch, offset_q15x2);
            q7_t *out_0 = output + (size_t)p * 2 * output_ch + row0;
            xw_compute_pair(wspace, (const q31_t *)runtime_buf, groups, row0, rows, bias, output_shift, output_mult,
                            out_offset, out_0, out_0 + output_ch);
        }
    }
}

/* same result, for layers whose expanded weights do not fit: wspace (>= te_conv1x1_xw_tiled_ok()) is split into a tile
 * buffer of tp pixel pairs and a weight block. Safe when the output overwrites the input in order (see the file header). */
void te_conv1x1_pairs_xw_tiled(const q7_t *input, const uint16_t input_ch, const q7_t *kernel, const int32_t *bias,
                               const int32_t *output_shift, const int32_t *output_mult, const int32_t out_offset,
                               const int32_t input_offset, q7_t *output, const int32_t num_pairs,
                               const uint16_t output_ch, q31_t *wspace, const size_t wspace_bytes)
{
    const int groups = input_ch >> 2;
    const int tp = xw_tile_pairs(wspace_bytes, num_pairs, input_ch);
    const size_t tile_bytes = (size_t)tp * 4U * input_ch;
    q31_t *const tile = wspace;
    q31_t *const wblk = wspace + tile_bytes / 4;
    const int block_rows = te_conv1x1_xw_rows(wspace_bytes - tile_bytes, input_ch);
    const int16_t inoff16 = input_offset;
    const q31_t offset_q15x2 = __PKHBT(inoff16, inoff16, 16);

    for (int p0 = 0; p0 < num_pairs; p0 += tp)
    {
        const int n = (num_pairs - p0) < tp ? (num_pairs - p0) : tp;
        /* read every input of the tile before writing any output of it: quads first, then the leftover pairs (at most
         * one, or all of them with TE_XW_NO_QUAD) */
#ifdef TE_XW_NO_QUAD
        const int nq = 0;
#else
        const int nq = n / 2;
#endif
        for (int q = 0; q < nq; q++)
        {
            xw_convert_quad(tile + (size_t)q * groups * 8, input + (size_t)(p0 + 2 * q) * 2 * input_ch, groups, input_ch,
                            offset_q15x2);
        }
        for (int q = 2 * nq; q < n; q++)
        {
            xw_convert_pair(tile + (size_t)nq * groups * 8 + (size_t)(q - 2 * nq) * groups * 4,
                            input + (size_t)(p0 + q) * 2 * input_ch, groups, input_ch, offset_q15x2);
        }
        for (int row0 = 0; row0 < output_ch; row0 += block_rows)
        {
            const int rows = (output_ch - row0) < block_rows ? (output_ch - row0) : block_rows;
            xw_expand_rows(wblk, kernel, row0, rows, groups, input_ch);
            for (int q = 0; q < nq; q++)
            {
                xw_compute_quad(wblk, tile + (size_t)q * groups * 8, groups, row0, rows, bias, output_shift,
                                output_mult, out_offset, output + (size_t)(p0 + 2 * q) * 2 * output_ch, output_ch);
            }
            for (int q = 2 * nq; q < n; q++)
            {
                q7_t *out_0 = output + (size_t)(p0 + q) * 2 * output_ch + row0;
                xw_compute_pair(wblk, tile + (size_t)nq * groups * 8 + (size_t)(q - 2 * nq) * groups * 4, groups, row0,
                                rows, bias, output_shift, output_mult, out_offset, out_0, out_0 + output_ch);
            }
        }
    }
}

/* ---- stem: 3x3 stride-2 pad-1 conv over a 3-channel image, as a 1x1-style product with the quad kernel ----
 * A patch is 27 values (3 rows x 3 pixels x 3 channels, contiguous in HWC). It is padded to 28 with a zero weight so it is
 * seven groups of four, and the same 2 x 4 block that runs the 1x1 layers computes four output pixels at once, against
 * weights expanded once. Requires input_x == 2 * output_x, input_y == 2 * output_y (so only the top/left edge is padded),
 * output_x % 4 == 0 and an even output_ch. Workspace bytes needed: te_stem_xw_bytes(). */
#define STEM_PATCH 28

size_t te_stem_xw_bytes(const uint16_t output_x, const uint16_t output_ch)
{
    /* padded kernel, expanded weights, one row of patches, one converted quad (each region 8-byte aligned) */
    return (size_t)output_ch * STEM_PATCH + (size_t)output_ch * STEM_PATCH * 2 + (size_t)output_x * STEM_PATCH
           + (STEM_PATCH / 4) * 32 + 4 * 8;
}

int te_stem_xw_ok(const uint16_t input_x, const uint16_t input_y, const uint16_t output_x, const uint16_t output_y,
                  const uint16_t output_ch, const size_t wspace_bytes)
{
#ifdef TE_XW_NO_QUAD
    return 0; /* the stem path needs the 2 x 4 block */
#else
    return input_x == 2 * output_x && input_y == 2 * output_y && output_x % 4 == 0 && output_ch % 2 == 0 &&
           te_stem_xw_bytes(output_x, output_ch) <= wspace_bytes;
#endif
}

void te_stem_xw(const q7_t *input, const uint16_t input_x, const q7_t *kernel, const int32_t *bias,
                const int32_t *output_shift, const int32_t *output_mult, const int32_t out_offset,
                const int32_t input_offset, q7_t *output, const uint16_t output_x, const uint16_t output_y,
                const uint16_t output_ch, q31_t *wspace, const q7_t pad_value)
{
    const int groups = STEM_PATCH / 4;
    const int16_t inoff16 = input_offset;
    const q31_t offset_q15x2 = __PKHBT(inoff16, inoff16, 16);

    q7_t *kpad = (q7_t *)wspace;
    q31_t *wblk = (q31_t *)((q7_t *)kpad + (((size_t)output_ch * STEM_PATCH + 7) & ~(size_t)7));
    q7_t *patch = (q7_t *)wblk + (((size_t)output_ch * STEM_PATCH * 2 + 7) & ~(size_t)7);
    q31_t *qbuf = (q31_t *)(patch + (((size_t)output_x * STEM_PATCH + 7) & ~(size_t)7));

    for (int r = 0; r < output_ch; r++)
    {
        memcpy(kpad + (size_t)r * STEM_PATCH, kernel + (size_t)r * 27, 27);
        kpad[(size_t)r * STEM_PATCH + 27] = 0;
    }
    xw_expand_rows(wblk, kpad, 0, output_ch, groups, STEM_PATCH);

    for (int oy = 0; oy < output_y; oy++)
    {
        /* one output row of patches */
        for (int ox = 0; ox < output_x; ox++)
        {
            q7_t *pt = patch + (size_t)ox * STEM_PATCH;
            const int ix0 = 2 * ox - 1;
            for (int ky = 0; ky < 3; ky++)
            {
                const int iy = 2 * oy - 1 + ky;
                if (iy < 0)
                {
                    memset(pt, pad_value, 9);
                }
                else if (ix0 < 0)
                {
                    memset(pt, pad_value, 3);
                    memcpy(pt + 3, input + ((size_t)iy * input_x) * 3, 6);
                }
                else
                {
                    memcpy(pt, input + ((size_t)iy * input_x + ix0) * 3, 9);
                }
                pt += 9;
            }
            *pt = 0;
        }
        q7_t *out_row = output + (size_t)oy * output_x * output_ch;
        for (int ox = 0; ox < output_x; ox += 4)
        {
            xw_convert_quad(qbuf, patch + (size_t)ox * STEM_PATCH, groups, STEM_PATCH, offset_q15x2);
            xw_compute_quad(wblk, qbuf, groups, 0, output_ch, bias, output_shift, output_mult, out_offset,
                            out_row + (size_t)ox * output_ch, output_ch);
        }
    }
}

#endif /* TE_DTCM */
