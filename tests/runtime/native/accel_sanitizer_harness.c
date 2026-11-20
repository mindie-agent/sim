/* Executed ABI/fuzz counterexample harness for engine-arena/2.
 *
 * [Role] Drive the real C11 kernel through the exact Round-2/4/5/6
 *   counterexamples: overlapping link ownership, start/completion-buffer
 *   continuation, subnormal underflow, full unsigned-64 sequences/cohorts,
 *   incremental placements, accepted-prefix publication and its absence.
 * [Invariants] This harness never reimplements scheduling arithmetic; every
 *   assertion reads a kernel status/completion cell.  It is compiled and run
 *   with ``-fsanitize=address,undefined -fno-sanitize-recover=all``, so any
 *   out-of-bounds write or undefined operation aborts the process.
 * [Timeouts & Locks] No threads, sleeps or I/O; loops are bounded by explicit
 *   iteration caps.
 * [Side-Effect Policy] Allocates and frees only its own local buffers and
 *   contexts; writes nothing outside the process.
 */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TASK_STRIDE 12u
#define LINK_STRIDE 6u
#define COMP_STRIDE 3u
#define SVC_STRIDE 3u
#define ST_FAILED 1
#define ST_PREFIX 2
#define B_HORIZON 2
#define B_OUTPUT 8

int32_t msim_ctx_config(void *ctx, const double *rates, const int32_t *units,
                        const int32_t *policy, size_t nres, int32_t *st,
                        size_t nst);
void msim_ctx_free(void *ctx);
void *msim_ctx_create(void);
int32_t msim_ctx_submit(void *ctx, const uint64_t *rows, size_t nrows,
                        size_t row_stride, const uint64_t *links, size_t nlinks,
                        size_t link_stride, const uint32_t *place, size_t nplace,
                        const int32_t *dep_task, const int32_t *dep_pred,
                        size_t ndeps, int32_t *out_slots, size_t nslots,
                        int32_t *st, size_t nst);
int32_t msim_ctx_run(void *ctx, uint64_t *hdr, size_t nhdr, int32_t *st,
                     size_t nst, double *io, size_t nio, uint64_t *comp,
                     size_t ncomp, double *svc, size_t nsvc, uint64_t *started,
                     size_t nstarted);
int32_t msim_ctx_link(void *ctx, const int32_t *pred, const int32_t *succ,
                      const uint64_t *sseq, size_t n, int32_t *st, size_t nst);
int32_t msim_ctx_reset(void *ctx);
int32_t msim_ctx_counters(void *ctx, double *busy, double *serv, size_t nres);

static uint64_t bits(double value)
{
    uint64_t out;
    memcpy(&out, &value, sizeof out);
    return out;
}

static double from_bits(uint64_t value)
{
    double out;
    memcpy(&out, &value, sizeof out);
    return out;
}

static void *context(double rate, int32_t units)
{
    double rates[1] = {rate};
    int32_t unit_cells[1] = {units};
    int32_t policy[1] = {0};
    int32_t st[8] = {0};
    void *ctx = msim_ctx_create();
    assert(ctx != NULL);
    assert(msim_ctx_config(ctx, rates, unit_cells, policy, 1, st, 8) == 0);
    return ctx;
}

static void row(uint64_t *out, uint64_t seq, uint64_t cohort, uint64_t waiting,
                uint64_t nlink, uint64_t lstart, double start, double latency,
                double release)
{
    memset(out, 0, TASK_STRIDE * sizeof(uint64_t));
    out[0] = seq;
    out[3] = cohort;
    out[4] = waiting;
    out[6] = bits(start);
    out[7] = bits(latency);
    out[8] = bits(release);
    out[9] = nlink;
    out[10] = lstart;
}

static void link(uint64_t *out, uint64_t owner, uint64_t res, double rem,
                 double cap, uint64_t pstart, uint64_t plen)
{
    memset(out, 0, LINK_STRIDE * sizeof(uint64_t));
    out[0] = owner;
    out[1] = res;
    out[2] = bits(rem);
    out[3] = bits(cap);
    out[4] = pstart;
    out[5] = plen;
}

static int32_t run(void *ctx, double *now, double horizon, int32_t *st,
                   uint64_t *hdr, double *io, uint64_t *comp, double *svc,
                   uint64_t *started)
{
    hdr[0] = 2;
    hdr[1] = 0;
    hdr[2] = 100000;
    io[0] = *now;
    io[1] = horizon;
    io[2] = 0.0;
    io[3] = INFINITY;
    memset(st, 0, 8 * sizeof(int32_t));
    assert(msim_ctx_run(ctx, hdr, 16, st, 8, io, 8, comp, COMP_STRIDE * 4096,
                        svc, SVC_STRIDE * 32768, started, 2 * 4096) == 0);
    *now = io[0];
    return st[0];
}

/* R2: 65 rows each claiming the same only link cell must be rejected. */
static void overlap_rejected(void)
{
    const size_t n = 65;
    uint64_t *rows = (uint64_t *)calloc(n * TASK_STRIDE, sizeof(uint64_t));
    uint64_t *links = (uint64_t *)calloc(LINK_STRIDE, sizeof(uint64_t));
    int32_t *slots = (int32_t *)calloc(n, sizeof(int32_t));
    int32_t st[8] = {0};
    size_t i;
    void *ctx = context(10.0, 1);
    for (i = 0; i < n; i++)
        row(rows + i * TASK_STRIDE, i + 1, 1, 0, 1, 0, 0.0, 0.0, 0.0);
    link(links, 0, 0, 1.0, INFINITY, 0, 0);
    assert(msim_ctx_submit(ctx, rows, n, TASK_STRIDE, links, 1, LINK_STRIDE,
                           NULL, 0, NULL, NULL, 0, slots, n, st, 8) != 0);
    assert(st[0] == ST_FAILED && st[1] == 1);
    msim_ctx_free(ctx);
    free(rows);
    free(links);
    free(slots);
    puts("overlap_rejected: ok");
}

/* R2/R6-1: 4097 starts and 4097 simultaneous completions cross the fixed
 * 4096-record wire buffers.  The overflow is a same-instant continuation
 * (B_OUTPUT), and a ``max_steps=0`` drain publishes the last record without
 * integrating or reporting caller step-budget exhaustion. */
static void start_buffer_exhaustion(void)
{
    const size_t n = 4097;
    uint64_t *rows = (uint64_t *)calloc(n * TASK_STRIDE, sizeof(uint64_t));
    uint64_t *links = (uint64_t *)calloc(n * LINK_STRIDE, sizeof(uint64_t));
    int32_t *slots = (int32_t *)calloc(n, sizeof(int32_t));
    uint64_t *comp = (uint64_t *)calloc(COMP_STRIDE * 4096, sizeof(uint64_t));
    double *svc = (double *)calloc(SVC_STRIDE * 32768, sizeof(double));
    uint64_t *started = (uint64_t *)calloc(2 * 4096, sizeof(uint64_t));
    uint64_t hdr[16] = {0};
    int32_t st[8] = {0};
    double io[8] = {0};
    unsigned char *seen = (unsigned char *)calloc(n, 1);
    size_t i, iter;
    int completions = 0;
    double now = 0.0;
    void *ctx = context((double)n, 1);
    for (i = 0; i < n; i++) {
        row(rows + i * TASK_STRIDE, i + 1, 1, 0, 1, i, 0.0, 0.0, 0.0);
        link(links + i * LINK_STRIDE, i, 0, 1.0, INFINITY, 0, 0);
    }
    assert(msim_ctx_submit(ctx, rows, n, TASK_STRIDE, links, n, LINK_STRIDE,
                           NULL, 0, NULL, NULL, 0, slots, n, st, 8) == 0);
    for (iter = 0; iter < 64; iter++) {
        run(ctx, &now, INFINITY, st, hdr, io, comp, svc, started);
        if (iter == 0)
            assert(st[2] == B_OUTPUT && st[3] == 0 && st[6] == 4096);
        for (i = 0; i < (size_t)st[6]; i++) {
            uint64_t seq = started[2 * i];
            assert(seq >= 1 && seq <= n);
            assert(seen[seq - 1] == 0);
            seen[seq - 1] = 1;
        }
        if (st[2] == B_OUTPUT && st[3] > 0) {
            /* A completion/service-buffer overflow after one real step. */
            assert(st[4] == 4096);
            completions += st[4];
            hdr[0] = 2; hdr[2] = 0; io[0] = now; io[1] = now;
            io[3] = INFINITY;
            memset(st, 0, sizeof st);
            assert(msim_ctx_run(ctx, hdr, 16, st, 8, io, 8, comp,
                                COMP_STRIDE * 4096, svc, SVC_STRIDE * 32768,
                                started, 2 * 4096) == 0);
            assert(io[0] == now && st[2] == B_HORIZON && st[3] == 0);
            puts("output_continuation_zero_budget: ok");
        }
        completions += st[4];
        if (hdr[15] == 0 && completions == (int)n)
            break;
    }
    assert(completions == (int)n);
    for (i = 0; i < n; i++) assert(seen[i] == 1);
    msim_ctx_free(ctx);
    free(rows); free(links); free(slots); free(comp); free(svc);
    free(started); free(seen);
    puts("start_buffer_exhaustion: ok");
}

/* R11: the minimum subnormal split across two consumers is an error at t=0. */
static void subnormal_underflow(void)
{
    const double tiny = 5e-324;
    uint64_t rows[2 * TASK_STRIDE];
    uint64_t links[2 * LINK_STRIDE];
    int32_t slots[2] = {0};
    uint64_t comp[COMP_STRIDE * 8] = {0};
    double svc[SVC_STRIDE * 8] = {0};
    uint64_t started[2 * 8] = {0};
    uint64_t hdr[16] = {0};
    int32_t st[8] = {0};
    double io[8] = {0};
    void *ctx = context(tiny, 1);
    row(rows, 1, 1, 0, 1, 0, 0.0, 0.0, 0.0);
    row(rows + TASK_STRIDE, 2, 1, 0, 1, 1, 0.0, 0.0, 0.0);
    link(links, 0, 0, tiny, INFINITY, 0, 0);
    link(links + LINK_STRIDE, 1, 0, tiny, INFINITY, 0, 0);
    assert(msim_ctx_submit(ctx, rows, 2, TASK_STRIDE, links, 2, LINK_STRIDE,
                           NULL, 0, NULL, NULL, 0, slots, 2, st, 8) == 0);
    hdr[0] = 2;
    hdr[2] = 100000;
    io[1] = 1.0;
    io[3] = INFINITY;
    assert(msim_ctx_run(ctx, hdr, 16, st, 8, io, 8, comp, COMP_STRIDE * 8,
                        svc, SVC_STRIDE * 8, started, 2 * 8) != 0);
    assert(st[1] == 4);
    assert(st[7] == 1);
    assert(st[4] == 0);
    assert(from_bits(bits(io[0])) == 0.0);
    msim_ctx_free(ctx);
    puts("subnormal_underflow: ok");
}

/* R11 control: one consumer at the same subnormal scale is representable. */
static void subnormal_single_ok(void)
{
    const double tiny = 5e-324;
    uint64_t rows[TASK_STRIDE];
    uint64_t links[LINK_STRIDE];
    int32_t slots[1] = {0};
    uint64_t comp[COMP_STRIDE * 8] = {0};
    double svc[SVC_STRIDE * 8] = {0};
    uint64_t started[2 * 8] = {0};
    uint64_t hdr[16] = {0};
    int32_t st[8] = {0};
    double io[8] = {0};
    double now = 0.0;
    void *ctx = context(tiny, 1);
    row(rows, 1, 1, 0, 1, 0, 0.0, 0.0, 0.0);
    link(links, 0, 0, tiny, INFINITY, 0, 0);
    assert(msim_ctx_submit(ctx, rows, 1, TASK_STRIDE, links, 1, LINK_STRIDE,
                           NULL, 0, NULL, NULL, 0, slots, 1, st, 8) == 0);
    run(ctx, &now, INFINITY, st, hdr, io, comp, svc, started);
    assert(st[4] == 1 && now == 1.0);
    msim_ctx_free(ctx);
    puts("subnormal_single_ok: ok");
}

/* R11: an all-zero allocated frontier under a None/infinite horizon is the
 * same per-consumer arithmetic error, not an infinite-dt stall. */
static void subnormal_zero_frontier_horizon_none(void)
{
    const double tiny = 5e-324;
    uint64_t rows[TASK_STRIDE];
    uint64_t links[LINK_STRIDE];
    uint32_t place[1] = {0};
    int32_t slots[1] = {0};
    uint64_t comp[COMP_STRIDE * 8] = {0};
    double svc[SVC_STRIDE * 8] = {0};
    uint64_t started[2 * 8] = {0};
    uint64_t hdr[16] = {0};
    int32_t st[8] = {0};
    double io[8] = {0};
    void *ctx = context(tiny, 2);
    row(rows, 1, 1, 0, 1, 0, 0.0, 0.0, 0.0);
    link(links, 0, 0, tiny, INFINITY, 0, 1);
    assert(msim_ctx_submit(ctx, rows, 1, TASK_STRIDE, links, 1, LINK_STRIDE,
                           place, 1, NULL, NULL, 0, slots, 1, st, 8) == 0);
    hdr[0] = 2;
    hdr[2] = 100000;
    io[1] = INFINITY;
    io[3] = INFINITY;
    assert(msim_ctx_run(ctx, hdr, 16, st, 8, io, 8, comp, COMP_STRIDE * 8,
                        svc, SVC_STRIDE * 8, started, 2 * 8) != 0);
    assert(st[1] == 4);
    assert(st[7] == 1);
    assert(st[4] == 0);
    assert(io[0] == 0.0);
    msim_ctx_free(ctx);
    puts("subnormal_zero_frontier_horizon_none: ok");
}

/* F3: the placement pool is growable and reclaimable, never a lifetime quota. */
static void sequential_placement_reclaim(void)
{
    const size_t n = 2000;
    uint64_t rows[TASK_STRIDE];
    uint64_t links[LINK_STRIDE];
    uint32_t place[64];
    int32_t slots[1] = {0};
    uint64_t comp[COMP_STRIDE * 8] = {0};
    double svc[SVC_STRIDE * 8] = {0};
    uint64_t started[2 * 8] = {0};
    uint64_t hdr[16] = {0};
    int32_t st[8] = {0};
    double io[8] = {0};
    size_t i, completed = 0;
    double now = 0.0;
    void *ctx = context(64.0, 64);
    for (i = 0; i < 64; i++) place[i] = (uint32_t)i;
    for (i = 0; i < n; i++) {
        row(rows, i + 1, 1, 0, 1, 0, 0.0, 0.0, 0.0);
        link(links, 0, 0, 1.0, INFINITY, 0, 64);
        assert(msim_ctx_submit(ctx, rows, 1, TASK_STRIDE, links, 1, LINK_STRIDE,
                               place, 64, NULL, NULL, 0, slots, 1, st, 8) == 0);
        run(ctx, &now, INFINITY, st, hdr, io, comp, svc, started);
        completed += (size_t)st[4];
        assert(st[4] == 1 && st[6] == 1);
    }
    assert(completed == n);
    assert(fabs(now - (double)n / 64.0) < 1e-12);
    /* The bridge resets at an empty boundary; the raw context must allow it. */
    assert(msim_ctx_reset(ctx) == 0);
    now = 0.0;
    row(rows, n + 1, 1, 0, 1, 0, 0.0, 0.0, 0.0);
    link(links, 0, 0, 1.0, INFINITY, 0, 64);
    assert(msim_ctx_submit(ctx, rows, 1, TASK_STRIDE, links, 1, LINK_STRIDE,
                           place, 64, NULL, NULL, 0, slots, 1, st, 8) == 0);
    run(ctx, &now, INFINITY, st, hdr, io, comp, svc, started);
    assert(st[4] == 1 && st[6] == 1);
    msim_ctx_free(ctx);
    puts("sequential_placement_reclaim: ok");
}

/* R3/R4: UINT64_MAX sequence and a 64-bit cohort survive exactly. */
static void wide_sequence_and_cohort(void)
{
    uint64_t rows[TASK_STRIDE];
    uint64_t links[LINK_STRIDE];
    int32_t slots[1] = {0};
    uint64_t comp[COMP_STRIDE * 8] = {0};
    double svc[SVC_STRIDE * 8] = {0};
    uint64_t started[2 * 8] = {0};
    uint64_t hdr[16] = {0};
    int32_t st[8] = {0};
    double io[8] = {0};
    double now = 0.0;
    void *ctx = context(1.0, 1);
    row(rows, UINT64_MAX, UINT64_MAX - 1, 0, 1, 0, 0.0, 0.0, 0.0);
    link(links, 0, 0, 1.0, INFINITY, 0, 0);
    assert(msim_ctx_submit(ctx, rows, 1, TASK_STRIDE, links, 1, LINK_STRIDE,
                           NULL, 0, NULL, NULL, 0, slots, 1, st, 8) == 0);
    run(ctx, &now, INFINITY, st, hdr, io, comp, svc, started);
    assert(st[4] == 1);
    assert(comp[0] == UINT64_MAX);
    assert(started[0] == UINT64_MAX);
    msim_ctx_free(ctx);
    puts("wide_sequence_and_cohort: ok");
}

/* Blocker 5: a second batch appends placements and rebases its offsets. */
static void incremental_placement(void)
{
    uint64_t a_rows[TASK_STRIDE];
    uint64_t b_rows[TASK_STRIDE];
    uint64_t a_links[LINK_STRIDE];
    uint64_t b_links[LINK_STRIDE];
    uint32_t a_place[1] = {0};
    uint32_t b_place[1] = {1};
    int32_t a_slots[1] = {0}, b_slots[1] = {0};
    uint64_t comp[COMP_STRIDE * 8] = {0};
    double svc[SVC_STRIDE * 8] = {0};
    uint64_t started[2 * 8] = {0};
    uint64_t hdr[16] = {0};
    int32_t st[8] = {0};
    double io[8] = {0};
    double now = 0.0;
    double a_time = -1.0, b_time = -1.0;
    int i;
    void *ctx = context(2.0, 2);
    row(a_rows, 1, 1, 0, 1, 0, 0.0, 0.0, 0.0);
    link(a_links, 0, 0, 2.0, INFINITY, 0, 1);
    assert(msim_ctx_submit(ctx, a_rows, 1, TASK_STRIDE, a_links, 1, LINK_STRIDE,
                           a_place, 1, NULL, NULL, 0, a_slots, 1, st, 8) == 0);
    run(ctx, &now, 0.5, st, hdr, io, comp, svc, started);
    assert(now == 0.5 && st[4] == 0 && st[6] == 1);
    row(b_rows, 2, 1, 0, 1, 0, 0.0, 0.0, 0.0);
    link(b_links, 0, 0, 1.0, INFINITY, 0, 1);
    assert(msim_ctx_submit(ctx, b_rows, 1, TASK_STRIDE, b_links, 1, LINK_STRIDE,
                           b_place, 1, NULL, NULL, 0, b_slots, 1, st, 8) == 0);
    {
        size_t iter;
        for (iter = 0; iter < 8; iter++) {
            run(ctx, &now, INFINITY, st, hdr, io, comp, svc, started);
            for (i = 0; i < st[4]; i++) {
                uint64_t seq = comp[COMP_STRIDE * i];
                double when = from_bits(comp[COMP_STRIDE * i + 1]);
                if (seq == 1) a_time = when;
                if (seq == 2) b_time = when;
            }
            if (hdr[15] == 0) break;
        }
    }
    assert(a_time == 2.0);
    assert(b_time == 1.5);
    (void)a_slots; (void)b_slots;
    msim_ctx_free(ctx);
    puts("incremental_placement: ok");
}

/* Malformed spans, slices and dependencies must be rejected, not touched. */
static void malformed_preflight(void)
{
    uint64_t rows[TASK_STRIDE];
    uint64_t links[LINK_STRIDE];
    uint32_t place_cell[1] = {7};
    int32_t slots[1] = {0};
    int32_t st[8] = {0};
    void *ctx = context(10.0, 1);

    row(rows, 1, 1, 0, 1, 1, 0.0, 0.0, 0.0);
    link(links, 0, 0, 1.0, INFINITY, 0, 0);
    memset(st, 0, sizeof st);
    assert(msim_ctx_submit(ctx, rows, 1, TASK_STRIDE, links, 1, LINK_STRIDE,
                           NULL, 0, NULL, NULL, 0, slots, 1, st, 8) != 0);
    assert(st[1] == 1);

    row(rows, 1, 1, 0, 1, 0, 0.0, 0.0, 0.0);
    link(links, 0, 0, 1.0, INFINITY, 1, 1);
    assert(msim_ctx_submit(ctx, rows, 1, TASK_STRIDE, links, 1, LINK_STRIDE,
                           place_cell, 1, NULL, NULL, 0, slots, 1, st, 8) != 0);
    assert(st[1] == 1);

    link(links, 0, 0, 1.0, INFINITY, 0, 0);
    assert(msim_ctx_submit(ctx, rows, 1, TASK_STRIDE, links, 1, LINK_STRIDE,
                           NULL, 0, NULL, NULL, 0, slots, 1, st, 8) == 0);
    {
        int32_t bad_task[1] = {1};
        int32_t bad_pred[1] = {slots[0]};
        assert(msim_ctx_submit(ctx, rows, 1, TASK_STRIDE, links, 1, LINK_STRIDE,
                               NULL, 0, bad_task, bad_pred, 1,
                               slots, 1, st, 8) != 0);
        assert(st[1] == 1);
    }
    msim_ctx_free(ctx);
    puts("malformed_preflight: ok");
}

/* Wrong ABI/NaN/capacity status must be reported before any progress. */
static void malformed_run_cells(void)
{
    uint64_t rows[TASK_STRIDE];
    uint64_t links[LINK_STRIDE];
    int32_t slots[1] = {0};
    int32_t st[8] = {0};
    uint64_t comp[COMP_STRIDE * 8] = {0};
    double svc[SVC_STRIDE * 8] = {0};
    uint64_t started[2 * 8] = {0};
    uint64_t hdr[16] = {0};
    double io[8] = {0};
    void *ctx = context(10.0, 1);

    row(rows, 1, 1, 0, 1, 0, 0.0, 0.0, 0.0);
    link(links, 0, 0, 1.0, INFINITY, 0, 0);
    assert(msim_ctx_submit(ctx, rows, 1, TASK_STRIDE, links, 1, LINK_STRIDE,
                           NULL, 0, NULL, NULL, 0, slots, 1, st, 8) == 0);
    hdr[0] = 99;
    hdr[2] = 10;
    io[1] = INFINITY;
    io[3] = INFINITY;
    assert(msim_ctx_run(ctx, hdr, 16, st, 8, io, 8, comp, COMP_STRIDE * 8,
                        svc, SVC_STRIDE * 8, started, 2 * 8) != 0);
    assert(st[1] == 7);
    hdr[0] = 2;
    assert(msim_ctx_run(ctx, hdr, 16, st, 8, io, 8, comp, 0,
                        svc, SVC_STRIDE * 8, started, 2 * 8) != 0);
    assert(st[1] == 5);
    io[0] = NAN;
    assert(msim_ctx_run(ctx, hdr, 16, st, 8, io, 8, comp, COMP_STRIDE * 8,
                        svc, SVC_STRIDE * 8, started, 2 * 8) != 0);
    assert(st[1] == 3);
    io[0] = 0.0;
    assert(msim_ctx_run(ctx, hdr, 16, st, 8, io, 8, comp, COMP_STRIDE * 8,
                        svc, SVC_STRIDE * 8, started, 2 * 8) == 0);
    assert(st[4] == 1 && st[6] == 1);
    msim_ctx_free(ctx);
    puts("malformed_run_cells: ok");
}

/* R4-5: a later E_ARITH publishes the valid prefix (clock, starts,
 * completions, counters) before returning the failing status. */
static void arithmetic_prefix_published(void)
{
    double rates[2] = {5e-324, 1.0};
    int32_t unit_cells[2] = {1, 1};
    int32_t policy[2] = {0, 0};
    int32_t st[8] = {0};
    uint64_t rows[3 * TASK_STRIDE];
    uint64_t links[3 * LINK_STRIDE];
    int32_t slots[3] = {0};
    uint64_t comp[COMP_STRIDE * 64] = {0};
    double svc[SVC_STRIDE * 128] = {0};
    uint64_t started[2 * 64] = {0};
    uint64_t hdr[16] = {0};
    double io[8] = {0};
    int32_t i, rc;
    void *ctx = msim_ctx_create();
    assert(ctx != NULL);
    assert(msim_ctx_config(ctx, rates, unit_cells, policy, 2, st, 8) == 0);
    for (i = 0; i < 3; i++) {
        /* good(r=0.5), a(tiny), b(tiny, release=1.0) */
        row(rows + (size_t)i * TASK_STRIDE, (uint64_t)(i + 1), 1, 0, 1,
            (uint64_t)i, 0.0, 0.0, i == 2 ? 1.0 : 0.0);
        link(links + (size_t)i * LINK_STRIDE, (uint64_t)i, i == 0 ? 1U : 0U,
             i == 0 ? 0.5 : 5e-323, INFINITY, 0, 0);
    }
    assert(msim_ctx_submit(ctx, rows, 3, TASK_STRIDE, links, 3, LINK_STRIDE,
                           NULL, 0, NULL, NULL, 0, slots, 3, st, 8) == 0);
    (void)slots;
    hdr[0] = 2;
    hdr[2] = 1000;
    io[1] = INFINITY;
    io[3] = INFINITY;
    rc = msim_ctx_run(ctx, hdr, 16, st, 8, io, 8, comp, COMP_STRIDE * 64,
                      svc, SVC_STRIDE * 128, started, 2 * 64);
    assert(rc != 0);
    assert(st[0] == (ST_FAILED | ST_PREFIX) && st[1] == 4 && st[7] == 1);
    assert(st[3] == 2);                     /* two accepted sub-steps */
    assert(st[4] == 1 && st[5] == 1);       /* good completed at 0.5 */
    assert(st[6] == 3);                     /* good, a, then b at 1.0 */
    assert(io[0] == 1.0);
    assert(comp[0] == 1 && from_bits(comp[1]) == 0.5);
    assert(svc[1] == 0.5 && svc[2] == 0.0);
    assert(started[4] == 3 && from_bits(started[5]) == 1.0);
    msim_ctx_free(ctx);
    puts("arithmetic_prefix_published: ok");
}
/* R4-6: a [valid, invalid] link batch is rejected before any edge commits. */
static void rejected_link_batch_is_atomic(void)
{
    uint64_t rows[2 * TASK_STRIDE];
    uint64_t links[2 * LINK_STRIDE];
    int32_t slots[2] = {0};
    int32_t st[8] = {0};
    int32_t pred[2], succ[2];
    uint64_t seqs[2];
    uint64_t comp[COMP_STRIDE * 16] = {0};
    double svc[SVC_STRIDE * 16] = {0};
    uint64_t started[2 * 16] = {0};
    uint64_t hdr[16] = {0};
    double io[8] = {0};
    double now[2] = {0.0, 0.0};
    int completions[2] = {0, 0};
    int ctx_i;
    for (ctx_i = 0; ctx_i < 2; ctx_i++) {
        void *ctx = context(1.0, 1);
        /* A: work 1, no predecessors.  B: work 1, waiting 2. */
        row(rows, 1, 1, 0, 1, 0, 0.0, 0.0, 0.0);
        row(rows + TASK_STRIDE, 2, 1, 2, 1, 1, 0.0, 0.0, 0.0);
        link(links, 0, 0, 1.0, INFINITY, 0, 0);
        link(links + LINK_STRIDE, 1, 0, 1.0, INFINITY, 0, 0);
        assert(msim_ctx_submit(ctx, rows, 2, TASK_STRIDE, links, 2,
                               LINK_STRIDE, NULL, 0, NULL, NULL, 0,
                               slots, 2, st, 8) == 0);
        if (ctx_i == 0) {
            pred[0] = slots[0]; succ[0] = slots[1]; seqs[0] = 2;
            pred[1] = slots[0]; succ[1] = 999999; seqs[1] = 3;
            assert(msim_ctx_link(ctx, pred, succ, seqs, 2, st, 8) != 0);
            assert(st[0] == ST_FAILED && st[1] == 1);
        }
        pred[0] = slots[0]; succ[0] = slots[1]; seqs[0] = 2;
        assert(msim_ctx_link(ctx, pred, succ, seqs, 1, st, 8) == 0);
        hdr[0] = 2;
        hdr[2] = 100;
        io[0] = 0.0;
        io[1] = INFINITY;
        io[3] = INFINITY;
        assert(msim_ctx_run(ctx, hdr, 16, st, 8, io, 8, comp,
                            COMP_STRIDE * 16, svc, SVC_STRIDE * 16,
                            started, 2 * 16) == 0);
        now[ctx_i] = io[0];
        completions[ctx_i] = st[4];
        assert(st[4] == 1);
        assert(comp[0] == 1 && from_bits(comp[1]) == 1.0);
        msim_ctx_free(ctx);
    }
    assert(now[0] == 1.0 && now[1] == 1.0);
    assert(completions[0] == completions[1]);
    puts("rejected_link_batch_is_atomic: ok");
}

/* R5-2: a compute_rates() zero-scale abort after 64 accepted completions and
 * one delayed start publishes the coherent accepted prefix exactly once. */
static void compute_rates_prefix_published(void)
{
    const double tiny = 5e-324;
    double rates[2] = {1.0, 1e308};
    int32_t unit_cells[2] = {1, 1};
    int32_t policy[2] = {0, 0};
    int32_t st[8] = {0};
    uint64_t rows[65 * TASK_STRIDE];
    uint64_t links[2 * LINK_STRIDE];
    int32_t slots[65] = {0};
    uint64_t comp[COMP_STRIDE * 64] = {0};
    double svc[SVC_STRIDE * 64] = {0};
    uint64_t started[2 * 65] = {0};
    uint64_t hdr[16] = {0};
    double io[8] = {0};
    double busy[2] = {0}, serv[2] = {0};
    int32_t i, rc;
    void *ctx = msim_ctx_create();
    assert(ctx != NULL);
    assert(msim_ctx_config(ctx, rates, unit_cells, policy, 2, st, 8) == 0);
    for (i = 0; i < 63; i++)
        row(rows + (size_t)i * TASK_STRIDE, (uint64_t)(i + 1), 1, 0, 0, 0,
            0.0, 0.0, 0.0);
    row(rows + 63 * TASK_STRIDE, 64, 1, 0, 1, 0, 0.0, 0.0, 0.0);
    link(links, 63, 0, 0.5, INFINITY, 0, 0);
    row(rows + 64 * TASK_STRIDE, 65, 1, 0, 1, 1, 0.0, 0.0, 1.0);
    rows[64 * TASK_STRIDE + 1] = 1;         /* MSIM_F_COUPLED */
    link(links + LINK_STRIDE, 64, 1, tiny, INFINITY, 0, 0);
    assert(msim_ctx_submit(ctx, rows, 65, TASK_STRIDE, links, 2, LINK_STRIDE,
                           NULL, 0, NULL, NULL, 0, slots, 65, st, 8) == 0);
    hdr[0] = 2;
    hdr[2] = 100000;
    io[1] = INFINITY;
    io[3] = INFINITY;
    rc = msim_ctx_run(ctx, hdr, 16, st, 8, io, 8, comp, COMP_STRIDE * 64,
                      svc, SVC_STRIDE * 64, started, 2 * 65);
    assert(rc != 0);
    assert(st[0] == (ST_FAILED | ST_PREFIX));
    assert(st[1] == 4 && st[7] == 0);       /* compute_rates, no resource id */
    assert(st[3] == 2);                     /* two accepted sub-steps */
    assert(st[4] == 64 && st[5] == 1);      /* 63 empty + good; one service */
    assert(st[6] == 65);                    /* 64 admissions + late at 1.0 */
    assert(io[0] == 1.0);
    assert(comp[63 * COMP_STRIDE] == 64);
    assert(from_bits(comp[63 * COMP_STRIDE + 1]) == 0.5);
    assert(svc[0] == 0.0 && svc[1] == 0.5 && svc[2] == 0.0);
    assert(started[64 * 2] == 65 && from_bits(started[64 * 2 + 1]) == 1.0);
    assert(msim_ctx_counters(ctx, busy, serv, 2) == 0);
    assert(busy[0] == 0.5 && busy[1] == 0.0);
    assert(serv[0] == 0.5 && serv[1] == 0.0);
    msim_ctx_free(ctx);
    puts("compute_rates_prefix_published: ok");
}
/* R5-2: the same compute_rates() failure on the very next epoch (no clock
 * advance, start, completion or step) must not fabricate a prefix. */
static void step_zero_prefix_absent(void)
{
    const double tiny = 5e-324;
    double rates[2] = {1.0, 1e308};
    int32_t unit_cells[2] = {1, 1};
    int32_t policy[2] = {0, 0};
    int32_t st[8] = {0};
    uint64_t rows[64 * TASK_STRIDE];
    uint64_t links[LINK_STRIDE];
    int32_t slots[64] = {0};
    uint64_t comp[COMP_STRIDE * 64] = {0};
    double svc[SVC_STRIDE * 64] = {0};
    uint64_t started[2 * 64] = {0};
    uint64_t hdr[16] = {0};
    double io[8] = {0};
    double busy[2] = {0}, serv[2] = {0};
    int32_t i, rc;
    void *ctx = msim_ctx_create();
    assert(ctx != NULL);
    assert(msim_ctx_config(ctx, rates, unit_cells, policy, 2, st, 8) == 0);
    for (i = 0; i < 63; i++)
        row(rows + (size_t)i * TASK_STRIDE, (uint64_t)(i + 1), 1, 0, 0, 0,
            0.0, 0.0, 0.0);
    row(rows + 63 * TASK_STRIDE, 64, 1, 0, 1, 0, 0.0, 0.0, 0.0);
    rows[63 * TASK_STRIDE + 1] = 1;         /* MSIM_F_COUPLED */
    link(links, 63, 1, tiny, INFINITY, 0, 0);
    assert(msim_ctx_submit(ctx, rows, 64, TASK_STRIDE, links, 1, LINK_STRIDE,
                           NULL, 0, NULL, NULL, 0, slots, 64, st, 8) == 0);
    hdr[0] = 2;
    hdr[2] = 100000;
    io[1] = 0.0;                            /* admission only, no rates */
    io[3] = INFINITY;
    assert(msim_ctx_run(ctx, hdr, 16, st, 8, io, 8, comp, COMP_STRIDE * 64,
                        svc, SVC_STRIDE * 64, started, 2 * 64) == 0);
    assert(st[4] == 63 && st[6] == 64 && io[0] == 0.0);
    io[1] = INFINITY;
    rc = msim_ctx_run(ctx, hdr, 16, st, 8, io, 8, comp, COMP_STRIDE * 64,
                      svc, SVC_STRIDE * 64, started, 2 * 64);
    assert(rc != 0);
    assert(st[0] == ST_FAILED);
    assert(st[1] == 4);
    assert(st[3] == 0 && st[4] == 0 && st[6] == 0);
    assert(io[0] == 0.0);
    assert(msim_ctx_counters(ctx, busy, serv, 2) == 0);
    assert(busy[0] == 0.0 && busy[1] == 0.0);
    assert(serv[0] == 0.0 && serv[1] == 0.0);
    msim_ctx_free(ctx);
    puts("step_zero_prefix_absent: ok");
}

#include "accel_sanitizer_handback.h"
#include "accel_sanitizer_wide.h"

int main(void)
{

    overlap_rejected();
    start_buffer_exhaustion();
    subnormal_underflow();
    subnormal_zero_frontier_horizon_none();
    subnormal_single_ok();
    sequential_placement_reclaim();
    wide_sequence_and_cohort();
    incremental_placement();
    malformed_preflight();
    malformed_run_cells();
    arithmetic_prefix_published();
    compute_rates_prefix_published();
    step_zero_prefix_absent();
    rejected_link_batch_is_atomic();
    r7_probes();
    wide_probes();
    puts("sanitizer_harness: ok");
    return 0;
}
