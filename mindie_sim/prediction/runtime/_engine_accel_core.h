/* Per-Engine state for the client-compiled C11 arena kernel (engine-arena/2).
 *
 * [Role] Declare the one heap-owned context that carries every mutable cell of
 *   an Engine's numeric accelerator: resource table, incremental pure-DAG task
 *   table, link table, placement pool, per-resource accumulators and the
 *   bounded progressive-filling scratch.  No object here is a global.
 * [Invariants] Pure ISO C11 <stdint.h>/<stddef.h>/<stdbool.h>/<math.h>/
 *   <float.h>/<string.h>/<stdlib.h>.  Each ``msim_ctx`` is reachable only from
 *   its owning Engine, so two Engines share zero mutable memory even while
 *   ``ctypes`` releases the GIL.  Every length is a checked element count and
 *   every capacity is validated before the first write.
 * [Timeouts & Locks] No threads, locks, I/O or unbounded loops: growth is
 *   geometric and every solver loop carries an explicit structural guard.
 * [Side-Effect Policy] Only the owning context and caller-provided output
 *   buffers are written; no pointer outlives the call that supplied it.
 */
#ifndef MSIM_ENGINE_ACCEL_CORE_H
#define MSIM_ENGINE_ACCEL_CORE_H
#ifdef _WIN32
#define MSIM_EXPORT __declspec(dllexport)
#else
#define MSIM_EXPORT
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Wire layout.  ``msim_abi_probe`` publishes this exact tuple so a mismatched
 * Python packer is rejected before the first numeric call. */
enum {
    MSIM_ABI_VERSION = 2,
    MSIM_HDR_CELLS = 16,      /* u64 header cells */
    MSIM_ST_CELLS = 8,        /* i32 status cells */
    MSIM_IO_CELLS = 8,        /* f64 request/reply cells */
    MSIM_TASK_STRIDE = 12,    /* u64 cells per submitted task */
    MSIM_LINK_STRIDE = 6,     /* u64 cells per submitted link */
    MSIM_COMP_STRIDE = 3,     /* u64 cells per completion record */
    MSIM_SVC_STRIDE = 3,      /* f64 cells per per-link service record */
    MSIM_MAX_T = 65536,       /* reusable task slots per context */
    MSIM_MAX_R = 128,
    MSIM_MAX_U = 64,
    MSIM_MAX_L = 262144,      /* live link cells per context */
    MSIM_MAX_F = 65536,       /* one flow per live link at most */
    MSIM_MAX_COEF = 262144,   /* one coefficient per live link at most */
    MSIM_MAX_MROW = 65536,    /* one solver row per live task at most */
    MSIM_MAX_SLOT = 8192,     /* structurally <= MAX_R * MAX_U + MAX_R */
    MSIM_MAX_SENT = 2097152,  /* dynamic ceiling for owner/entry scratch */
    MSIM_MAX_OWN = 32768,     /* mirrored dependency-edge quota */
    MSIM_MAX_UENT = 2097152   /* dynamic ceiling for effective unit memberships:
                               * every live work link contributes its explicit
                               * ``plen`` or, when unplaced, ``res.units`` */
};

/* ``st[0]`` is a status word: bit 0 is set on failure, bit 1 is set exactly
 * when the failing call also published a trustworthy accepted prefix.  A
 * success leaves both bits clear. */
enum {
    MSIM_ST_FAILED = 1,
    MSIM_ST_PREFIX = 2
};

/* Task flags (u64 row cell 1). */
enum {
    MSIM_F_COUPLED = 1,
    MSIM_F_LATENCY = 2,
    MSIM_F_CALLBACK = 4,      /* completion needs a Python callback */
    MSIM_F_STOP = 8,          /* observed ``until`` target: stop after it */
    MSIM_F_OCCURRENCE = 16,   /* completion updates occurrence metrics */
    MSIM_F_EXT_START = 32     /* Python admits the start (allocations) */
};

/* Error codes carried by a negative i32[0] plus i32[1]. */
enum {
    MSIM_E_RANGE = 1,
    MSIM_E_STALL = 2,
    MSIM_E_VALUE = 3,
    MSIM_E_ARITH = 4,
    MSIM_E_FULL = 5,
    MSIM_E_STATE = 6,
    MSIM_E_ABI = 7
};

/* Boundary kinds returned in i32[2].  ``MSIM_B_OUTPUT`` is a
 * continuation inside the current instant: an output wire buffer filled while
 * same-instant starts/completions were still pending.  ``MSIM_B_BUDGET`` is
 * reserved for a fully settled instant that needs another positive-dt step. */
enum {
    MSIM_B_NONE = 0,
    MSIM_B_CALLBACK = 1,
    MSIM_B_HORIZON = 2,
    MSIM_B_LATENCY = 3,
    MSIM_B_BUDGET = 4,
    MSIM_B_IDLE = 5,
    MSIM_B_RELEASE = 6,
    MSIM_B_STOP = 7,
    MSIM_B_OUTPUT = 8
};

typedef struct {
    uint64_t seq;            /* full unsigned 64-bit domain */
    int64_t prio;
    int64_t waiting;        /* pending predecessors not yet delivered */
    int64_t nlink;
    int64_t lstart;
    double start;
    double latency;
    double release;
    int32_t flags;
    uint64_t cohort;         /* 1..UINT64_MAX-1; UINT64_MAX is reserved */
    int32_t ext_waiters;    /* non-mirrored successors: force a boundary */
    int32_t state;          /* 0 pending, 1 active, 2 free */
    int32_t ready;          /* queued in the priority ready heap */
    int32_t compl;          /* queued in the ascending-sequence heap */
    int32_t act_slot;       /* position in the active list, -1 when inactive */
    int32_t link_block;     /* first reusable link cell owned by this slot */
    int32_t link_block_len; /* cells reserved at ``link_block`` */
    int32_t first_succ;     /* head of the mirrored-successor edge list */
} msim_task;

typedef struct {
    int32_t task;           /* owning slot */
    int32_t res;
    int32_t plen;
    int32_t pstart;
    int32_t mode;           /* partition group id, -1 otherwise */
    double rem;
    double cap;
    double rate;
    double svc;             /* accumulated ``task.service`` for this link */
} msim_link;

typedef struct {
    double rate;
    int32_t units;
    int32_t policy;
    int32_t mode;           /* branch used for the binary64 accumulation order */
} msim_res;

typedef struct { int32_t res; int32_t link; double amount; } msim_coef;
typedef struct { int32_t task, coeff_start, coeff_count; double limit, assigned; } msim_flow;
typedef struct { int32_t flow; double amount; } msim_entry;
typedef struct { double cap; int32_t start, count; } msim_slot;
typedef struct { int32_t flow, pstart, plen; double amount; } msim_row;

typedef struct msim_ctx {
    uint32_t magic;
    int32_t alive;
    msim_res *res;
    int32_t nres, res_cap;
    double *busy;
    double *serv;
    int32_t counter_cap;
    uint8_t resume_compl;     /* resume an interrupted completion snapshot */
    int32_t pending_hold;     /* preserve admission hold across B_OUTPUT */
    /* Task slots: fixed cap, free list, append-only link cells. */
    msim_task *task;
    int32_t task_cap, live_n;
    int32_t *free;
    int32_t free_n, free_cap;
    msim_link *link;
    int32_t nl, link_cap;
    uint32_t *units;
    int32_t nu, units_cap;
    int32_t *heap;
    int32_t heap_n, heap_cap;
    int32_t *rheap;
    int32_t rheap_n, rheap_cap;
    int32_t *cheap;
    int32_t cheap_n, cheap_cap;
    int32_t *act;
    int32_t act_n, act_cap;
    /* Mirrored successor edges (head per predecessor slot). */
    int32_t *sidx;
    int32_t *snext;
    uint64_t *sseq;
    int32_t sn, s_cap;
    /* Scratch sized once from the documented structural caps. */
    int32_t *idx;
    int32_t *gid;
    int32_t *rep;
    int32_t *pend;
    int32_t *surv;
    int32_t *fifo_first;
    int32_t *uhead;
    int32_t *unext;
    int32_t *uflow;
    double *uamt;
    int32_t uent_cap;         /* grown unit-membership scratch capacity */
    double *over;
    double *remu;
    double *bounds;
    double *slope;
    bool *active;
    bool *active2;
    bool *changed;
    msim_coef *coef;
    msim_flow *flow;
    msim_entry *entry;
    msim_slot *slot;
    msim_row *row;
    int32_t *ohead;
    int32_t *onext;
    int32_t *oslot;
    int32_t own_cap;          /* grown owner/entry scratch capacity */
} msim_ctx;

/* -- validated helpers -------------------------------------------------- */
static double msim_from_bits(uint64_t v)
{
    double d;
    memcpy(&d, &v, sizeof d);
    return d;
}

static uint64_t msim_to_bits(double d)
{
    uint64_t v;
    memcpy(&v, &d, sizeof v);
    return v;
}

/* Python ``math.ulp``: zero -> minimum subnormal; largest finite -> 2**971. */
static double msim_ulp(double x)
{
    if (isnan(x) || isinf(x)) return x;
    if (x == 0.0) return 0x1p-1074;
    if (x < 0.0) x = -x;
    if (x < DBL_MAX) return nextafter(x, INFINITY) - x;
    return DBL_MAX - nextafter(DBL_MAX, 0.0);
}

static double msim_min(double a, double b) { return a <= b ? a : b; }

/* Overflow-free ``idx + len <= cap`` for non-negative values. */
static bool msim_fits(size_t idx, size_t len, size_t cap)
{
    return idx <= cap && len <= cap - idx;
}

static bool msim_grow(void **block, int32_t *cap, int32_t need, size_t elem)
{
    int32_t target;
    void *next;
    if (need < 0) return false;
    if (need <= *cap) return true;
    target = *cap > 0 ? *cap : 64;
    while (target < need) {
        if (target > 0x3fffffff) { target = need; break; }
        target *= 2;
    }
    if ((size_t)target > (size_t)-1 / elem) return false;
    next = realloc(*block, (size_t)target * elem);
    if (next == NULL) return false;
    *block = next;
    *cap = target;
    return true;
}

static void *msim_zalloc(size_t count, size_t elem)
{
    if (elem != 0 && count > (size_t)-1 / elem) return NULL;
    return calloc(count ? count : 1, elem);
}

/* Demand-grown unit-membership scratch: ``uent`` bounds effective unit
 * memberships of the live work links (each contributes its explicit ``plen`` or
 * the declared resource-unit count), not explicit placement-pool storage.  It
 * is not a fixed 64K buffer; the ceiling only stops geometric growth from
 * reaching the signed 32-bit offset domain. */
static bool msim_uent_reserve(msim_ctx *c, int32_t need)
{
    int32_t target;
    int32_t *next_flow, *next_link;
    double *next_amount;
    if (need < 0) return false;
    if (need <= c->uent_cap) return true;
    target = c->uent_cap > 0 ? c->uent_cap : 64;
    while (target < need) {
        if (target > MSIM_MAX_UENT / 2) { target = need; break; }
        target *= 2;
    }
    if (target > MSIM_MAX_UENT) target = MSIM_MAX_UENT;
    if (target < need) return false;
    next_flow = (int32_t *)realloc(c->uflow, (size_t)target * sizeof(int32_t));
    if (next_flow == NULL) return false;
    c->uflow = next_flow;
    next_link = (int32_t *)realloc(c->unext, (size_t)target * sizeof(int32_t));
    if (next_link == NULL) return false;
    c->unext = next_link;
    next_amount = (double *)realloc(c->uamt, (size_t)target * sizeof(double));
    if (next_amount == NULL) return false;
    c->uamt = next_amount;
    c->uent_cap = target;
    return true;
}

/* The coupled solver's owner/entry scratch obeys the same demand-grown rule. */
static bool msim_own_reserve(msim_ctx *c, int32_t need)
{
    int32_t target;
    msim_entry *next_entry;
    int32_t *next_link, *next_slot;
    if (need < 0) return false;
    if (need <= c->own_cap) return true;
    target = c->own_cap > 0 ? c->own_cap : 64;
    while (target < need) {
        if (target > MSIM_MAX_SENT / 2) { target = need; break; }
        target *= 2;
    }
    if (target > MSIM_MAX_SENT) target = MSIM_MAX_SENT;
    if (target < need) return false;
    next_entry = (msim_entry *)realloc(c->entry, (size_t)target * sizeof(msim_entry));
    if (next_entry == NULL) return false;
    c->entry = next_entry;
    next_link = (int32_t *)realloc(c->onext, (size_t)target * sizeof(int32_t));
    if (next_link == NULL) return false;
    c->onext = next_link;
    next_slot = (int32_t *)realloc(c->oslot, (size_t)target * sizeof(int32_t));
    if (next_slot == NULL) return false;
    c->oslot = next_slot;
    c->own_cap = target;
    return true;
}

static void msim_ctx_zero(msim_ctx *c)
{
    memset(c, 0, sizeof *c);
    c->magic = 0x4d53494dU;
    c->alive = 1;
}

static void msim_ctx_release(msim_ctx *c)
{
    if (c == NULL) return;
    free(c->res); free(c->busy); free(c->serv);
    free(c->task); free(c->free); free(c->link); free(c->units);
    free(c->heap); free(c->rheap); free(c->cheap); free(c->act);
    free(c->sidx); free(c->snext); free(c->sseq);
    free(c->idx); free(c->gid); free(c->rep); free(c->pend); free(c->surv);
    free(c->fifo_first); free(c->uhead); free(c->unext); free(c->uflow);
    free(c->uamt); free(c->over); free(c->remu); free(c->bounds); free(c->slope);
    free(c->active); free(c->active2); free(c->changed);
    free(c->coef); free(c->flow); free(c->entry); free(c->slot); free(c->row);
    free(c->ohead); free(c->onext); free(c->oslot);
    memset(c, 0, sizeof *c);
    free(c);
}

/* One-time allocation of the bounded solver workspace. */
static bool msim_ctx_scratch(msim_ctx *c)
{
    c->coef = (msim_coef *)msim_zalloc((size_t)MSIM_MAX_COEF, sizeof(msim_coef));
    c->flow = (msim_flow *)msim_zalloc((size_t)MSIM_MAX_F, sizeof(msim_flow));
    c->entry = NULL;
    c->slot = (msim_slot *)msim_zalloc((size_t)MSIM_MAX_SLOT, sizeof(msim_slot));
    c->row = (msim_row *)msim_zalloc((size_t)MSIM_MAX_MROW, sizeof(msim_row));
    c->ohead = (int32_t *)msim_zalloc((size_t)MSIM_MAX_F, sizeof(int32_t));
    c->onext = NULL;
    c->oslot = NULL;
    c->idx = (int32_t *)msim_zalloc((size_t)MSIM_MAX_L, sizeof(int32_t));
    c->gid = (int32_t *)msim_zalloc((size_t)MSIM_MAX_L, sizeof(int32_t));
    c->rep = (int32_t *)msim_zalloc((size_t)MSIM_MAX_L, sizeof(int32_t));
    c->pend = (int32_t *)msim_zalloc((size_t)MSIM_MAX_L, sizeof(int32_t));
    c->surv = (int32_t *)msim_zalloc((size_t)MSIM_MAX_L, sizeof(int32_t));
    c->fifo_first = (int32_t *)msim_zalloc((size_t)MSIM_MAX_R, sizeof(int32_t));
    c->uhead = (int32_t *)msim_zalloc((size_t)MSIM_MAX_U, sizeof(int32_t));
    c->unext = NULL;
    c->uflow = NULL;
    c->uamt = NULL;
    c->over = (double *)msim_zalloc((size_t)MSIM_MAX_L, sizeof(double));
    c->remu = (double *)msim_zalloc((size_t)MSIM_MAX_U, sizeof(double));
    c->bounds = (double *)msim_zalloc((size_t)MSIM_MAX_U, sizeof(double));
    c->slope = (double *)msim_zalloc((size_t)MSIM_MAX_U, sizeof(double));
    c->active = (bool *)msim_zalloc((size_t)MSIM_MAX_L, sizeof(bool));
    c->active2 = (bool *)msim_zalloc((size_t)MSIM_MAX_U, sizeof(bool));
    c->changed = (bool *)msim_zalloc((size_t)MSIM_MAX_SLOT, sizeof(bool));
    c->task = (msim_task *)msim_zalloc((size_t)MSIM_MAX_T, sizeof(msim_task));
    c->task_cap = MSIM_MAX_T;
    return c->coef && c->flow && c->slot && c->row && c->ohead
        && c->idx && c->gid && c->rep && c->pend
        && c->surv && c->fifo_first && c->uhead && c->over
        && c->remu && c->bounds && c->slope
        && c->active && c->active2 && c->changed && c->task;
}

#endif /* MSIM_ENGINE_ACCEL_CORE_H */
