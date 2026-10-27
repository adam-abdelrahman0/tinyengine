/* Debug aid: per-layer output summary, emitted by genModel.c only when the
 * code was generated with TE_TRACE=1. main.c sets g_te_trace_on around the
 * logit-check run; otherwise te_trace() does nothing. */
#include <stdio.h>
#include "tinyengine_function.h"

volatile int g_te_trace_on = 0;

void te_trace(const char *tensor_idx, const q7_t *p, int n)
{
	if (!g_te_trace_on)
		return;
	int32_t sum = 0;
	for (int i = 0; i < n; i++)
		sum += p[i];
	printf("TRACE %s n=%d sum=%ld first=", tensor_idx, n, (long) sum);
	for (int i = 0; i < 256 && i < n; i++)
		printf("%d,", p[i]);
	printf("\r\n");
}
