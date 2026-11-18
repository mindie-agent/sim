/* Wide-topology, 72-AIV, large-slot and bounded-stack counterexamples.
 *
 * [Role] Drive the real C11 kernel with the capacities the raised arena claims:
 *   more than 256 resources, 36-AIC/72-AIV owner groups (unit ids at and above
 *   64), solver entry indices above 65535, more than the former 8192 solver
 *   slots, and the coupled solver on a 512 KiB pthread stack.
 * [Invariants] Every assertion reads a kernel status, completion or counter
 *   cell; nothing here reimplements scheduling arithmetic.  The unit masks must
 *   never shift a 64-bit word by 64 or more, which UBSan would report as
 *   undefined behaviour, and the slot solver must keep its scratch off the C
 *   stack.
 * [Timeouts & Locks] One bounded pthread join; all loops carry explicit caps.
 * [Side-Effect Policy] Allocates and frees only its own buffers and contexts.
 */
#ifndef ACCEL_SANITIZER_WIDE_H
#define ACCEL_SANITIZER_WIDE_H

#include <pthread.h>

#define WIDE_RESOURCES 300
#define WIDE_AIV_UNITS 72
#define WIDE_AIC_UNITS 36
#define WIDE_SLOT_GROUPS 5000
#define WIDE_SLOT_UNITS 128
#define WIDE_ENTRY_TASKS 513
#define WIDE_STACK_BYTES 524288u

#define WIDE_COMP_RECORDS 4096
#define WIDE_SVC_RECORDS 32768
#define WIDE_START_RECORDS 8192
#define WIDE_DRAIN_GUARD 64

/* Drain every completion of one frontier across the fixed wire buffers. */
static size_t wide_drain(void *ctx, double *now, int32_t *st, uint64_t *hdr,
                         double *io, uint64_t *comp, double *svc,
                         uint64_t *started, size_t total)
{
    size_t done = 0, guard = 0;
    while (done < total && guard++ < WIDE_DRAIN_GUARD) {
        run(ctx, now, INFINITY, st, hdr, io, comp, svc, started);
        done += (size_t)st[4];
        if (hdr[15] == 0) break;
    }
    return done;
}

/* One context with ``nres`` fair resources declared by parallel arrays. */
static void *wide_context(const double *rates, const int32_t *units,
                          const int32_t *policy, size_t nres)
{
    int32_t st[8] = {0};
    void *ctx = msim_ctx_create();
    assert(ctx != NULL);
    assert(msim_ctx_config(ctx, rates, units, policy, nres, st, 8) == 0);
    return ctx;
}

/* A 72-AIV owner group whose disjoint single-unit placements straddle the
 * former 64-unit word: every unit must receive the same equal share. */
static void wide_seventy_two_owner_units(void)
{
    const size_t n = WIDE_AIV_UNITS;
    const double rate = (double)WIDE_AIV_UNITS;
    uint64_t *rows = (uint64_t *)calloc(n * TASK_STRIDE, sizeof(uint64_t));
    uint64_t *links = (uint64_t *)calloc(n * LINK_STRIDE, sizeof(uint64_t));
    uint32_t *place = (uint32_t *)calloc(n, sizeof(uint32_t));
    int32_t *slots = (int32_t *)calloc(n, sizeof(int32_t));
    uint64_t *comp = (uint64_t *)calloc(COMP_STRIDE * WIDE_COMP_RECORDS, sizeof(uint64_t));
    double *svc = (double *)calloc(SVC_STRIDE * WIDE_SVC_RECORDS, sizeof(double));
    uint64_t *started = (uint64_t *)calloc(2 * WIDE_START_RECORDS, sizeof(uint64_t));
    uint64_t hdr[16] = {0};
    int32_t st[8] = {0};
    double io[8] = {0};
    double now = 0.0;
    size_t i;
    void *ctx;
    assert(WIDE_AIC_UNITS < WIDE_AIV_UNITS);
    ctx = context(rate, WIDE_AIV_UNITS);
    for (i = 0; i < n; i++) {
        row(rows + i * TASK_STRIDE, i + 1, 1, 0, 1, i, 0.0, 0.0, 0.0);
        link(links + i * LINK_STRIDE, i, 0, 1.0, INFINITY, i, 1);
        place[i] = (uint32_t)i;
    }
    assert(msim_ctx_submit(ctx, rows, n, TASK_STRIDE, links, n, LINK_STRIDE,
                           place, n, NULL, NULL, 0, slots, n, st, 8) == 0);
    run(ctx, &now, INFINITY, st, hdr, io, comp, svc, started);
    assert(st[4] == (int32_t)n);
    /* Every disjoint single-unit owner gets rate / units == 1.0 per second,
     * including owner ids 64..71 that a single-word mask cannot address. */
    assert(fabs(now - 1.0) < 1e-12);
    msim_ctx_free(ctx);
    free(rows); free(links); free(place); free(slots);
    free(comp); free(svc); free(started);
    puts("wide_seventy_two_owner_units: ok");
}

/* More than 256 declared resources, one task each, all completing together. */
static void wide_resources_beyond_256(void)
{
    const size_t n = WIDE_RESOURCES;
    double *rates = (double *)calloc(n, sizeof(double));
    int32_t *units = (int32_t *)calloc(n, sizeof(int32_t));
    int32_t *policy = (int32_t *)calloc(n, sizeof(int32_t));
    uint64_t *rows = (uint64_t *)calloc(n * TASK_STRIDE, sizeof(uint64_t));
    uint64_t *links = (uint64_t *)calloc(n * LINK_STRIDE, sizeof(uint64_t));
    int32_t *slots = (int32_t *)calloc(n, sizeof(int32_t));
    uint64_t *comp = (uint64_t *)calloc(COMP_STRIDE * WIDE_COMP_RECORDS, sizeof(uint64_t));
    double *svc = (double *)calloc(SVC_STRIDE * WIDE_SVC_RECORDS, sizeof(double));
    uint64_t *started = (uint64_t *)calloc(2 * WIDE_START_RECORDS, sizeof(uint64_t));
    double busy[WIDE_RESOURCES] = {0}, serv[WIDE_RESOURCES] = {0};
    uint64_t hdr[16] = {0};
    int32_t st[8] = {0};
    double io[8] = {0};
    double now = 0.0;
    size_t i;
    void *ctx;
    for (i = 0; i < n; i++) {
        rates[i] = 100.0;
        units[i] = 1;
        policy[i] = (int32_t)(i & 1u);
    }
    ctx = wide_context(rates, units, policy, n);
    for (i = 0; i < n; i++) {
        row(rows + i * TASK_STRIDE, i + 1, 1, 0, 1, i, 0.0, 0.0, 0.0);
        link(links + i * LINK_STRIDE, i, (uint64_t)i, 300.0, INFINITY, 0, 0);
    }
    assert(msim_ctx_submit(ctx, rows, n, TASK_STRIDE, links, n, LINK_STRIDE,
                           NULL, 0, NULL, NULL, 0, slots, n, st, 8) == 0);
    run(ctx, &now, INFINITY, st, hdr, io, comp, svc, started);
    assert(st[4] == (int32_t)n);
    assert(fabs(now - 3.0) < 1e-12);
    assert(msim_ctx_counters(ctx, busy, serv, n) == 0);
    for (i = 0; i < n; i++) {
        assert(fabs(busy[i] - 3.0) < 1e-12);
        assert(fabs(serv[i] - 300.0) < 1e-12);
    }
    msim_ctx_free(ctx);
    free(rates); free(units); free(policy); free(rows); free(links);
    free(slots); free(comp); free(svc); free(started);
    puts("wide_resources_beyond_256: ok");
}

/* More than the former 8192-slot ceiling: 2 x 5000 two-consumer slots. */
static void wide_slot_groups_beyond_former_cap(void)
{
    const size_t groups = WIDE_SLOT_GROUPS;
    const size_t nres = 2;
    const size_t n = groups * 2 * nres;
    double rates[2] = {1000.0, 1000.0};
    int32_t units[2] = {WIDE_SLOT_UNITS, WIDE_SLOT_UNITS};
    int32_t policy[2] = {0, 0};
    uint64_t *rows = (uint64_t *)calloc(n * TASK_STRIDE, sizeof(uint64_t));
    uint64_t *links = (uint64_t *)calloc(n * LINK_STRIDE, sizeof(uint64_t));
    uint32_t *place = (uint32_t *)calloc(2 * n, sizeof(uint32_t));
    int32_t *slots = (int32_t *)calloc(n, sizeof(int32_t));
    uint64_t *comp = (uint64_t *)calloc(COMP_STRIDE * WIDE_COMP_RECORDS, sizeof(uint64_t));
    double *svc = (double *)calloc(SVC_STRIDE * WIDE_SVC_RECORDS, sizeof(double));
    uint64_t *started = (uint64_t *)calloc(2 * WIDE_START_RECORDS, sizeof(uint64_t));
    uint64_t hdr[16] = {0};
    int32_t st[8] = {0};
    double io[8] = {0};
    double now = 0.0;
    size_t i = 0, r, g, cell = 0;
    void *ctx = wide_context(rates, units, policy, nres);
    for (r = 0; r < nres; r++) {
        for (g = 0; g < groups; g++) {
            uint32_t a = (uint32_t)((g * 2) % WIDE_SLOT_UNITS);
            uint32_t b = (uint32_t)((g * 2 + 1) % WIDE_SLOT_UNITS);
            uint32_t pair[2] = {a, b};
            size_t copy;
            if (a == b) continue;
            for (copy = 0; copy < 2; copy++) {
                size_t index = i;
                row(rows + index * TASK_STRIDE, index + 1, 1, 0, 1, index,
                    0.0, 0.0, 0.0);
                link(links + index * LINK_STRIDE, index, r, 1.0, INFINITY,
                     cell, 2);
                place[cell] = pair[0];
                place[cell + 1] = pair[1];
                cell += 2;
                i++;
            }
        }
    }
    assert(i > 0 && i <= n);
    assert(msim_ctx_submit(ctx, rows, i, TASK_STRIDE, links, i, LINK_STRIDE,
                           place, cell, NULL, NULL, 0, slots, i, st, 8) == 0);
    assert(wide_drain(ctx, &now, st, hdr, io, comp, svc, started, i) == i);
    assert(now > 0.0 && isfinite(now));
    msim_ctx_free(ctx);
    free(rows); free(links); free(place); free(slots);
    free(comp); free(svc); free(started);
    puts("wide_slot_groups_beyond_former_cap: ok");
}

/* Solver entry cells above 65535: 513 consumers of all 128 owner units. */
static void wide_slot_entry_indices_above_16_bit(void)
{
    const size_t n = WIDE_ENTRY_TASKS;
    uint64_t *rows = (uint64_t *)calloc(n * TASK_STRIDE, sizeof(uint64_t));
    uint64_t *links = (uint64_t *)calloc(n * LINK_STRIDE, sizeof(uint64_t));
    uint32_t *place = (uint32_t *)calloc(n * WIDE_SLOT_UNITS, sizeof(uint32_t));
    int32_t *slots = (int32_t *)calloc(n, sizeof(int32_t));
    uint64_t *comp = (uint64_t *)calloc(COMP_STRIDE * WIDE_COMP_RECORDS, sizeof(uint64_t));
    double *svc = (double *)calloc(SVC_STRIDE * WIDE_SVC_RECORDS, sizeof(double));
    uint64_t *started = (uint64_t *)calloc(2 * WIDE_START_RECORDS, sizeof(uint64_t));
    uint64_t hdr[16] = {0};
    int32_t st[8] = {0};
    double io[8] = {0};
    double now = 0.0;
    size_t i, k;
    void *ctx = context((double)WIDE_SLOT_UNITS * 4.0, WIDE_SLOT_UNITS);
    for (i = 0; i < n; i++) {
        row(rows + i * TASK_STRIDE, i + 1, 1, 0, 1, i, 0.0, 0.0, 0.0);
        link(links + i * LINK_STRIDE, i, 0, 1.0, INFINITY,
             i * WIDE_SLOT_UNITS, WIDE_SLOT_UNITS);
        for (k = 0; k < WIDE_SLOT_UNITS; k++)
            place[i * WIDE_SLOT_UNITS + k] = (uint32_t)k;
    }
    assert(msim_ctx_submit(ctx, rows, n, TASK_STRIDE, links, n, LINK_STRIDE,
                           place, n * WIDE_SLOT_UNITS, NULL, NULL, 0,
                           slots, n, st, 8) == 0);
    assert(wide_drain(ctx, &now, st, hdr, io, comp, svc, started, n) == n);
    /* 128 solver slots, each holding all 513 consumers: 65664 entry cells. */
    assert(now > 0.0 && isfinite(now));
    msim_ctx_free(ctx);
    free(rows); free(links); free(place); free(slots);
    free(comp); free(svc); free(started);
    puts("wide_slot_entry_indices_above_16_bit: ok");
}

/* The coupled solver on a 512 KiB worker stack must not touch the C stack. */
static void *wide_bounded_stack_worker(void *arg)
{
    uint64_t *rows = (uint64_t *)calloc(4 * TASK_STRIDE, sizeof(uint64_t));
    uint64_t *links = (uint64_t *)calloc(4 * LINK_STRIDE, sizeof(uint64_t));
    uint32_t *place = (uint32_t *)calloc(4 * 8, sizeof(uint32_t));
    int32_t *slots = (int32_t *)calloc(4, sizeof(int32_t));
    uint64_t *comp = (uint64_t *)calloc(COMP_STRIDE * WIDE_COMP_RECORDS, sizeof(uint64_t));
    double *svc = (double *)calloc(SVC_STRIDE * WIDE_SVC_RECORDS, sizeof(double));
    uint64_t *started = (uint64_t *)calloc(2 * WIDE_START_RECORDS, sizeof(uint64_t));
    uint64_t hdr[16] = {0};
    int32_t st[8] = {0};
    double io[8] = {0};
    double now = 0.0;
    size_t i, k;
    void *ctx = context(8.0, 8);
    int32_t *result = (int32_t *)arg;
    for (i = 0; i < 4; i++) {
        row(rows + i * TASK_STRIDE, i + 1, 1, 0, 1, i, 0.0, 0.0, 0.0);
        if (i == 0) rows[i * TASK_STRIDE + 1] = 1;  /* MSIM_F_COUPLED */
        link(links + i * LINK_STRIDE, i, 0, 1.0, INFINITY, i * 8, 8);
        for (k = 0; k < 8; k++) place[i * 8 + k] = (uint32_t)k;
    }
    if (msim_ctx_submit(ctx, rows, 4, TASK_STRIDE, links, 4, LINK_STRIDE,
                        place, 4 * 8, NULL, NULL, 0, slots, 4, st, 8) != 0) {
        *result = -1;
    } else {
        run(ctx, &now, INFINITY, st, hdr, io, comp, svc, started);
        *result = (st[4] == 4 && fabs(now - 0.5) < 1e-12) ? 1 : 0;
    }
    msim_ctx_free(ctx);
    free(rows); free(links); free(place); free(slots);
    free(comp); free(svc); free(started);
    return NULL;
}

static void wide_bounded_stack_thread(void)
{
    pthread_attr_t attr;
    pthread_t thread;
    int32_t result = 0;
    assert(pthread_attr_init(&attr) == 0);
    assert(pthread_attr_setstacksize(&attr, (size_t)WIDE_STACK_BYTES) == 0);
    assert(pthread_create(&thread, &attr, wide_bounded_stack_worker, &result) == 0);
    assert(pthread_join(thread, NULL) == 0);
    assert(pthread_attr_destroy(&attr) == 0);
    assert(result == 1);
    puts("wide_bounded_stack_thread: ok");
}

static void wide_probes(void)
{
    wide_seventy_two_owner_units();
    wide_resources_beyond_256();
    wide_slot_groups_beyond_former_cap();
    wide_slot_entry_indices_above_16_bit();
    wide_bounded_stack_thread();
}

#endif /* ACCEL_SANITIZER_WIDE_H */
