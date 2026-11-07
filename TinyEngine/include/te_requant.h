/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   te_requant.h
 *
 * te_requantize(): bit-identical to CMSIS-NN's arm_nn_requantize() (double
 * rounding: saturating-doubling-high-multiply, then a rounding right shift that
 * rounds ties away from zero), written with fewer instructions.
 *
 * arm_nn_requantize() ends with arm_nn_divide_by_power_of_two(), which compares the
 * remainder with a threshold and conditionally increments. For an exponent e > 0 that
 * is the same as adding half an LSB, biased by the sign, and shifting once:
 *
 *     (x + ((1 << e) >> 1) + (x >> 31)) >> e
 *
 * (x >> 31 is -1 for negative x: it turns the tie at exactly -k.5 away from zero).
 * e == 0 is the identity and must not take the bias, so it is split out.
 *
 * Precondition: |hi| < 2^30, where hi is the result of the high multiply. That always
 * holds for conv/depthwise accumulators (|acc| is below about 2^27) and is what makes
 * "hi + half" safe from int32 overflow; CMSIS's compare-and-increment has no such
 * addition, so it would also accept values within 2^(e-1) of INT32_MAX. The host test
 * (see the commit message) checks equality against arm_nn_requantize() for every
 * shift in [-31, 12] and random and boundary (val, multiplier) inside that domain.
 * -------------------------------------------------------------------- */
#ifndef TINYENGINE_TE_REQUANT_H_
#define TINYENGINE_TE_REQUANT_H_

#include "arm_nnsupportfunctions.h"

__STATIC_FORCEINLINE q31_t te_requantize(const q31_t val, const q31_t multiplier, const q31_t shift)
{
    const q31_t left = shift > 0 ? shift : 0;
    const q31_t right = shift > 0 ? 0 : -shift;
    /* saturating rounding doubling high multiply: (val << left) * multiplier * 2 / 2^32, rounded; no saturation
     * case needed (multiplier is never INT32_MIN) */
    const q31_t hi = (q31_t)((((q63_t)(val * (1 << left)) * multiplier) + (1 << 30)) >> 31);
    if (right == 0)
    {
        return hi;
    }
    return (hi + (q31_t)(((uint32_t)1 << right) >> 1) + (hi >> 31)) >> right;
}

/* te_requantize() specialised for the common case: every per-channel shift <= 0 (a pure right shift) and the
 * activation clamp is the whole int8 range, so the clamp is one SSAT. Same result as
 *     r = arm_nn_requantize(acc, mult, shift); r += out_offset; r = MIN(MAX(r, -128), 127);
 * about 14 instructions fewer per output (no left shift, no shift sign tests, no clamp compares or stack reloads).
 * te_s8_fast_ok() says whether a layer qualifies; callers fall back to the generic code if it does not. */
/* (a * b + 2^31) >> 32 in one instruction (SMMULR); the C fallback is the same expression */
__STATIC_FORCEINLINE q31_t te_smmulr(const q31_t a, const q31_t b)
{
#if defined(__arm__)
    q31_t r;
    __asm("smmulr %0, %1, %2" : "=r"(r) : "r"(a), "r"(b));
    return r;
#else
    return (q31_t)((((q63_t)a * b) + ((q63_t)1 << 31)) >> 32);
#endif
}

/* hi = (acc * multiplier + 2^30) >> 31 == smmulr(2 * acc, multiplier); 2 * acc cannot overflow for |acc| < 2^30
 * (conv accumulators are below about 2^27) */
__STATIC_FORCEINLINE q31_t te_requantize_s8(const q31_t acc, const q31_t multiplier, const q31_t shift,
                                            const q31_t out_offset)
{
    const q31_t right = -shift;
    const q31_t hi = te_smmulr(acc * 2, multiplier);
    const q31_t r = right ? (hi + (q31_t)(((uint32_t)1 << right) >> 1) + (hi >> 31)) >> right : hi;
    return __SSAT(r + out_offset, 8);
}

__STATIC_FORCEINLINE int te_s8_fast_ok(const int32_t *shifts, const int n, const int32_t act_min, const int32_t act_max)
{
    if (act_min != -128 || act_max != 127)
    {
        return 0;
    }
    for (int i = 0; i < n; i++)
    {
        if (shifts[i] > 0)
        {
            return 0;
        }
    }
    return 1;
}

#endif /* TINYENGINE_TE_REQUANT_H_ */
