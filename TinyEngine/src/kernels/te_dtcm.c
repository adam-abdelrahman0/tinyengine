/* Weight copy buffer in DTCMRAM, see convolve_1x1_s8.c. Only built with -DTE_DTCM (tinyengine_bench.py, TE_DTCM=1); the
 * .dtcm_bss section comes from the patched linker script it generates. */
#include "tinyengine_function.h"

#ifdef TE_DTCM
q7_t te_wbuf[TE_WBUF_BYTES] __attribute__((section(".dtcm_bss"), aligned(8)));
#endif
