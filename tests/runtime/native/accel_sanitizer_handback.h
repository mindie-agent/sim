/* Round-7 native probes for the executed ABI/sanitizer harness.
 *
 * [Role] Embed one instrumented copy of the unchanged kernel (renamed symbols,
 *   one-shot failing ``realloc``) so the three R7-1 drain-failure prefixes are
 *   verified from C, and drive the real kernel through the R7-2 completion and
 *   service-buffer continuations.
 * [Invariants] The instrumented copy is the production source verbatim; only
 *   the allocator and exported symbol names differ.  Every check reads a kernel
 *   status/completion/counter cell, never reimplemented arithmetic.
 * [Timeouts & Locks] No threads or I/O; every loop is bounded by a table size.
 * [Side-Effect Policy] Allocates and frees only its own buffers and contexts.
 */
#ifndef MSIM_ACCEL_SANITIZER_R7_H
#define MSIM_ACCEL_SANITIZER_R7_H

#include <stdlib.h>
#include <string.h>

static int r7_armed;
static int r7_hits;
static void *r7_target;

static void *r7_realloc(void *ptr, size_t bytes)
{
    if (r7_armed && ptr == r7_target) {
        r7_armed = 0;
        r7_hits++;
        return NULL;
    }
    return realloc(ptr, bytes);
}

#define realloc r7_realloc
#define msim_abi_version r7_abi_version
#define msim_abi_probe r7_abi_probe
#define msim_ctx_create r7_ctx_create
#define msim_ctx_free r7_ctx_free
#define msim_ctx_reset r7_ctx_reset
#define msim_ctx_config r7_ctx_config
#define msim_ctx_seed r7_ctx_seed
#define msim_ctx_counters r7_ctx_counters
#define msim_ctx_activate r7_ctx_activate
#define msim_ctx_link r7_ctx_link
#define msim_ctx_deliver r7_ctx_deliver
#define msim_ctx_drop r7_ctx_drop
#define msim_ctx_mark_external r7_ctx_mark_external
#define msim_ctx_export r7_ctx_export
#define msim_ctx_run r7_ctx_run
#define msim_ctx_submit r7_ctx_submit
#include "_engine_accel.c"
#undef msim_ctx_submit
#undef msim_ctx_run
#undef msim_ctx_export
#undef msim_ctx_mark_external
#undef msim_ctx_drop
#undef msim_ctx_deliver
#undef msim_ctx_link
#undef msim_ctx_activate
#undef msim_ctx_counters
#undef msim_ctx_seed
#undef msim_ctx_config
#undef msim_ctx_reset
#undef msim_ctx_free
#undef msim_ctx_create
#undef msim_abi_probe
#undef msim_abi_version
#undef realloc

static void r7_arm(void *ctx, int kind)
{
    msim_ctx *c = (msim_ctx *)ctx;
    r7_target = kind == 1 ? (void *)c->cheap
              : kind == 2 ? (void *)c->heap : (void *)c->act;
    r7_armed = 1;
}

/* One raw run; returns the kernel's rc and leaves the status cells intact. */
static int32_t r7_run(void *ctx, double *now, double horizon, int64_t budget,
                      int32_t *st, uint64_t *hdr, double *io, uint64_t *comp,
                      double *svc, uint64_t *started)
{
    hdr[0] = 2; hdr[1] = 0; hdr[2] = (uint64_t)budget; hdr[15] = 0;
    io[0] = *now; io[1] = horizon; io[2] = 0.0; io[3] = INFINITY;
    memset(st, 0, 8 * sizeof(int32_t));
    {
        int32_t rc = r7_ctx_run(ctx, hdr, 16, st, 8, io, 8,
                                comp, COMP_STRIDE * 4096, svc,
                                SVC_STRIDE * 32768, started, 2 * 4096);
        *now = io[0];
        return rc;
    }
}

static void r7_fault_admission(void)
{
    const size_t n = 4097;
    uint64_t *rows = (uint64_t *)calloc(n * TASK_STRIDE, sizeof(uint64_t));
    int32_t *slots = (int32_t *)calloc(n, sizeof(int32_t));
    uint64_t comp[COMP_STRIDE * 8] = {0};
    double svc[SVC_STRIDE * 8] = {0};
    uint64_t started[2 * 4096] = {0};
    uint64_t hdr[16] = {0};
    int32_t st[8] = {0};
    double io[8] = {0};
    double now = 0.0, busy[1] = {0}, serv[1] = {0};
    size_t i;
    void *ctx = r7_ctx_create();
    {
        double rates[1] = {1.0};
        int32_t units[1] = {1}, policy[1] = {0};
        assert(ctx != NULL);
        assert(r7_ctx_config(ctx, rates, units, policy, 1, st, 8) == 0);
    }
    for (i = 0; i < n; i++)
        row(rows + i * TASK_STRIDE, i + 1, 1, 0, 0, 0, 0.0, 0.0, 0.0);
    assert(r7_ctx_submit(ctx, rows, n, TASK_STRIDE, NULL, 0, LINK_STRIDE,
                         NULL, 0, NULL, NULL, 0, slots, n, st, 8) == 0);
    assert(r7_run(ctx, &now, 0.0, 100000, st, hdr, io, comp, svc, started) == 0);
    assert(st[2] == B_OUTPUT && st[6] == 4096);
    r7_arm(ctx, 1);
    assert(r7_run(ctx, &now, 0.0, 0, st, hdr, io, comp, svc, started) != 0);
    assert(r7_hits == 1 && r7_armed == 0);
    assert(st[0] == (ST_FAILED | ST_PREFIX) && st[1] == 1);
    assert(st[3] == 0 && st[4] == 0 && st[6] == 1);
    assert(io[0] == 0.0 && now == 0.0);
    assert(r7_ctx_counters(ctx, busy, serv, 1) == 0);
    assert(busy[0] == 0.0 && serv[0] == 0.0);
    r7_ctx_free(ctx);
    free(rows); free(slots);
    puts("r7_fault_admission_prefix: ok");
}

static void r7_fault_completion(void)
{
    const size_t roots = 4097, children = 8193, n = roots + children;
    uint64_t *rows = (uint64_t *)calloc(n * TASK_STRIDE, sizeof(uint64_t));
    uint64_t *links = (uint64_t *)calloc(n * LINK_STRIDE, sizeof(uint64_t));
    int32_t *slots = (int32_t *)calloc(n, sizeof(int32_t));
    uint64_t comp[COMP_STRIDE * 4096] = {0};
    double svc[SVC_STRIDE * 8192] = {0};
    uint64_t started[2 * 4096] = {0};
    uint64_t hdr[16] = {0};
    int32_t st[8] = {0};
    double io[8] = {0};
    double now = 0.0, busy[1] = {0}, serv[1] = {0};
    size_t i;
    int32_t *pred = (int32_t *)calloc(children, sizeof(int32_t));
    int32_t *succ = (int32_t *)calloc(children, sizeof(int32_t));
    uint64_t *sseq = (uint64_t *)calloc(children, sizeof(uint64_t));
    void *ctx = r7_ctx_create();
    {
        double rates[1] = {4097.0};
        int32_t units[1] = {1}, policy[1] = {0};
        assert(ctx != NULL);
        assert(r7_ctx_config(ctx, rates, units, policy, 1, st, 8) == 0);
    }
    for (i = 0; i < n; i++) {
        row(rows + i * TASK_STRIDE, i + 1, 1, i < roots ? 0 : 1, 1, i,
            0.0, 0.0, 0.0);
        link(links + i * LINK_STRIDE, i, 0, 1.0, INFINITY, 0, 0);
    }
    assert(r7_ctx_submit(ctx, rows, n, TASK_STRIDE, links, n, LINK_STRIDE,
                         NULL, 0, NULL, NULL, 0, slots, n, st, 8) == 0);
    for (i = 0; i < children; i++) {
        pred[i] = slots[roots - 1];
        succ[i] = slots[roots + i];
        sseq[i] = roots + 1 + i;
    }
    assert(r7_ctx_link(ctx, pred, succ, sseq, children, st, 8) == 0);
    assert(r7_run(ctx, &now, 0.0, 100000, st, hdr, io, comp, svc, started) == 0);
    assert(st[2] == B_OUTPUT && st[6] == 4096);
    assert(r7_run(ctx, &now, 0.0, 0, st, hdr, io, comp, svc, started) == 0);
    assert(st[2] == B_HORIZON && st[6] == 1 && now == 0.0);
    assert(r7_run(ctx, &now, 1.0, 100000, st, hdr, io, comp, svc, started) == 0);
    assert(st[2] == B_OUTPUT && st[3] == 1 && st[4] == 4096 && st[6] == 0);
    assert(now == 1.0);
    r7_arm(ctx, 2);
    assert(r7_run(ctx, &now, 1.0, 0, st, hdr, io, comp, svc, started) != 0);
    assert(r7_hits == 2 && r7_armed == 0);
    assert(st[0] == (ST_FAILED | ST_PREFIX) && st[1] == 1);
    assert(st[3] == 0 && st[4] == 1 && st[6] == 0 && io[0] == 1.0);
    assert(r7_ctx_counters(ctx, busy, serv, 1) == 0);
    assert(busy[0] == 1.0 && serv[0] == 4097.0);
    r7_ctx_free(ctx);
    free(rows); free(links); free(slots); free(pred); free(succ); free(sseq);
    puts("r7_fault_completion_prefix: ok");
}

static void r7_fault_no_prefix(void)
{
    const size_t n = 4098;
    uint64_t *rows = (uint64_t *)calloc(n * TASK_STRIDE, sizeof(uint64_t));
    uint64_t *links = (uint64_t *)calloc(n * LINK_STRIDE, sizeof(uint64_t));
    int32_t *slots = (int32_t *)calloc(n, sizeof(int32_t));
    uint64_t comp[COMP_STRIDE * 8] = {0};
    double svc[SVC_STRIDE * 8] = {0};
    uint64_t started[2 * 4096] = {0};
    uint64_t hdr[16] = {0};
    int32_t st[8] = {0};
    double io[8] = {0};
    double now = 0.0, busy[1] = {0}, serv[1] = {0};
    size_t i;
    void *ctx = r7_ctx_create();
    {
        double rates[1] = {1.0};
        int32_t units[1] = {1}, policy[1] = {0};
        assert(ctx != NULL);
        assert(r7_ctx_config(ctx, rates, units, policy, 1, st, 8) == 0);
    }
    for (i = 0; i < n; i++) {
        double release = i == 0 ? 0.0 : 1.0;
        row(rows + i * TASK_STRIDE, i + 1, 1, 0, 1, i, 0.0, 0.0, release);
        link(links + i * LINK_STRIDE, i, 0, i == 0 ? 0.5 : 1.0,
             INFINITY, 0, 0);
    }
    assert(r7_ctx_submit(ctx, rows, n, TASK_STRIDE, links, n, LINK_STRIDE,
                         NULL, 0, NULL, NULL, 0, slots, n, st, 8) == 0);
    assert(r7_run(ctx, &now, 0.0, 1, st, hdr, io, comp, svc, started) == 0);
    assert(st[2] == B_HORIZON && st[6] == 1 && now == 0.0);
    assert(r7_run(ctx, &now, 1.0, 100000, st, hdr, io, comp, svc, started) == 0);
    assert(st[2] == B_OUTPUT && st[3] == 2 && st[4] == 1 && st[6] == 4096);
    assert(now == 1.0);
    r7_arm(ctx, 3);
    assert(r7_run(ctx, &now, 1.0, 0, st, hdr, io, comp, svc, started) != 0);
    assert(r7_hits == 3 && r7_armed == 0);
    assert(st[0] == ST_FAILED && st[1] == 1);
    assert(st[3] == 0 && st[4] == 0 && st[6] == 0 && io[0] == 1.0);
    assert(r7_ctx_counters(ctx, busy, serv, 1) == 0);
    assert(busy[0] == 0.5 && serv[0] == 0.5);
    r7_ctx_free(ctx);
    free(rows); free(links); free(slots);
    puts("r7_fault_no_prefix_counters: ok");
}

/* R7-2: 4,096 completions fill the record buffer.  The continuation must emit
 * the remaining root before the zero-work ``child`` awakened by root 0. */
static void r7_completion_snapshot_order(void)
{
    const int32_t roots = 4096;
    const size_t n = (size_t)roots + 2, links_n = (size_t)roots + 1;
    uint64_t *rows = (uint64_t *)calloc(n * TASK_STRIDE, sizeof(uint64_t));
    uint64_t *links = (uint64_t *)calloc(links_n * LINK_STRIDE, sizeof(uint64_t));
    int32_t *slots = (int32_t *)calloc(n, sizeof(int32_t));
    uint64_t comp[COMP_STRIDE * 4096] = {0};
    double svc[SVC_STRIDE * 4096] = {0};
    uint64_t started[2 * 4096] = {0};
    uint64_t hdr[16] = {0};
    int32_t st[8] = {0};
    double io[8] = {0};
    size_t i;
    int32_t pred[1], succ[1];
    uint64_t sseq[1] = {roots + 1};
    void *ctx = context(4097.0, 1);
    for (i = 0; i < (size_t)roots; i++) {
        row(rows + i * TASK_STRIDE, i + 1, 1, 0, 1, i, 0.0, 0.0, 0.0);
        link(links + i * LINK_STRIDE, i, 0, 1.0, INFINITY, 0, 0);
    }
    row(rows + roots * TASK_STRIDE, roots + 1, 1, 1, 0, links_n, 0.0, 0.0, 0.0);
    row(rows + (roots + 1) * TASK_STRIDE, roots + 2, 1, 0, 1, roots,
        0.0, 0.0, 0.0);
    link(links + roots * LINK_STRIDE, roots + 1, 0, 1.0, INFINITY, 0, 0);
    assert(msim_ctx_submit(ctx, rows, n, TASK_STRIDE, links, links_n,
                           LINK_STRIDE, NULL, 0, NULL, NULL, 0, slots, n,
                           st, 8) == 0);
    pred[0] = slots[0]; succ[0] = slots[roots];
    assert(msim_ctx_link(ctx, pred, succ, sseq, 1, st, 8) == 0);
    hdr[0] = 2; hdr[1] = 0; hdr[2] = 100000;
    io[0] = 0.0; io[1] = 0.0; io[3] = INFINITY;
    assert(msim_ctx_run(ctx, hdr, 16, st, 8, io, 8, comp, COMP_STRIDE * 4096,
                        svc, SVC_STRIDE * 4096, started, 2 * 4096) == 0);
    assert(st[2] == B_OUTPUT && st[6] == 4096);
    hdr[2] = 100000; io[0] = 0.0; io[1] = 1.0; io[3] = INFINITY;
    assert(msim_ctx_run(ctx, hdr, 16, st, 8, io, 8, comp, COMP_STRIDE * 4096,
                        svc, SVC_STRIDE * 4096, started, 2 * 4096) == 0);
    assert(st[2] == B_OUTPUT && st[3] == 1 && st[4] == 4096 && st[6] == 1);
    hdr[2] = 0; io[0] = 1.0; io[1] = 1.0;
    assert(msim_ctx_run(ctx, hdr, 16, st, 8, io, 8, comp, COMP_STRIDE * 4096,
                        svc, SVC_STRIDE * 4096, started, 2 * 4096) == 0);
    assert(st[2] == B_HORIZON && st[3] == 0 && st[4] == 2 && st[6] == 1);
    assert(comp[0] == roots + 2 && comp[COMP_STRIDE] == roots + 1);
    msim_ctx_free(ctx);
    free(rows); free(links); free(slots);
    puts("r7_completion_snapshot_order: ok");
}

/* R7-2: the service-record buffer fills after 1,024 x 32 links. */
static void r7_service_snapshot_order(void)
{
    const int32_t roots = 1025, nres = 32;
    const size_t n = (size_t)roots + 1, links_n = (size_t)roots * nres;
    uint64_t *rows = (uint64_t *)calloc(n * TASK_STRIDE, sizeof(uint64_t));
    uint64_t *links = (uint64_t *)calloc(links_n * LINK_STRIDE, sizeof(uint64_t));
    int32_t *slots = (int32_t *)calloc(n, sizeof(int32_t));
    uint64_t comp[COMP_STRIDE * 4096] = {0};
    double svc[SVC_STRIDE * 32768] = {0};
    uint64_t started[2 * 4096] = {0};
    uint64_t hdr[16] = {0};
    int32_t st[8] = {0};
    double io[8] = {0};
    double rates[32], busy[32], serv[32];
    int32_t units[32], policy[32];
    int32_t r, i;
    int32_t pred[1], succ[1];
    uint64_t sseq[1] = {roots + 1};
    void *ctx;
    for (r = 0; r < nres; r++) {
        rates[r] = 1025.0; units[r] = 1; policy[r] = 0;
    }
    ctx = msim_ctx_create();
    assert(ctx != NULL);
    assert(msim_ctx_config(ctx, rates, units, policy, nres, st, 8) == 0);
    for (i = 0; i < roots; i++) {
        row(rows + (size_t)i * TASK_STRIDE, (uint64_t)i + 1, 1, 0, nres,
            (uint64_t)i * nres, 0.0, 0.0, 0.0);
        for (r = 0; r < nres; r++)
            link(links + ((size_t)i * nres + r) * LINK_STRIDE, i, r, 1.0,
                 INFINITY, 0, 0);
    }
    row(rows + roots * TASK_STRIDE, roots + 1, 1, 1, 0, links_n, 0.0, 0.0, 0.0);
    assert(msim_ctx_submit(ctx, rows, n, TASK_STRIDE, links, links_n,
                           LINK_STRIDE, NULL, 0, NULL, NULL, 0, slots, n,
                           st, 8) == 0);
    pred[0] = slots[0]; succ[0] = slots[roots];
    assert(msim_ctx_link(ctx, pred, succ, sseq, 1, st, 8) == 0);
    hdr[0] = 2; hdr[1] = 0; hdr[2] = 1;
    io[0] = 0.0; io[1] = 0.0; io[3] = INFINITY;
    assert(msim_ctx_run(ctx, hdr, 16, st, 8, io, 8, comp, COMP_STRIDE * 4096,
                        svc, SVC_STRIDE * 32768, started, 2 * 4096) == 0);
    assert(st[2] == B_HORIZON && st[6] == roots);
    hdr[2] = 100000;
    io[0] = 0.0; io[1] = 1.0; io[3] = INFINITY;
    assert(msim_ctx_run(ctx, hdr, 16, st, 8, io, 8, comp, COMP_STRIDE * 4096,
                        svc, SVC_STRIDE * 32768, started, 2 * 4096) == 0);
    assert(st[2] == B_OUTPUT && st[3] == 1 && st[4] == 1024 && st[6] == 0);
    hdr[2] = 0; io[0] = 1.0; io[1] = 1.0;
    assert(msim_ctx_run(ctx, hdr, 16, st, 8, io, 8, comp, COMP_STRIDE * 4096,
                        svc, SVC_STRIDE * 32768, started, 2 * 4096) == 0);
    assert(st[2] == B_HORIZON && st[3] == 0 && st[4] == 2 && st[6] == 1);
    assert(comp[0] == (uint64_t)roots && comp[COMP_STRIDE] == (uint64_t)roots + 1);
    assert(msim_ctx_counters(ctx, busy, serv, nres) == 0);
    for (r = 0; r < nres; r++) {
        assert(busy[r] == 1.0 && serv[r] == 1025.0);
    }
    msim_ctx_free(ctx);
    free(rows); free(links); free(slots);
    puts("r7_service_snapshot_order: ok");
}

static void r7_probes(void)
{
    r7_fault_admission();
    r7_fault_completion();
    r7_fault_no_prefix();
    r7_completion_snapshot_order();
    r7_service_snapshot_order();
}

#endif /* MSIM_ACCEL_SANITIZER_R7_H */
