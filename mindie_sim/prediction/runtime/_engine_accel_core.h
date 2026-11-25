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
    MSIM_MAX_R = 2048,        /* ABI probe wire compatibility constant */
    MSIM_MAX_RES = 65536,     /* dynamic growable resource-table ceiling */
    MSIM_MAX_U = 128,         /* 36-AIC / 72-AIV and wider owner groups */
    MSIM_MAX_L = 262144,      /* live link cells per context */
    MSIM_MAX_F = 65536,       /* one flow per live link at most */
    MSIM_MAX_COEF = 262144,   /* one coefficient per live link at most */
    MSIM_MAX_MROW = 65536,    /* one solver row per live task at most */
    MSIM_MAX_SLOT = 2097152,  /* dynamic ceiling matching MSIM_MAX_UENT for growable nres */
    MSIM_MAX_SENT = 2097152,  /* dynamic ceiling for owner/entry scratch */
    MSIM_MAX_OWN = 262144,    /* mirrored dependency-edge quota */
    MSIM_MAX_UENT = 2097152   /* dynamic ceiling for effective unit memberships:
                               * every live work link contributes its explicit
                               * ``plen`` or, when unplaced, ``res.units`` */
};

/* A bounded unit bitset wide enough for every declared owner id.  Two 64-bit
 * words cover ``MSIM_MAX_U == 128`` without ever shifting by 64 or more, which
 * would be undefined; a single word is still the fast path for ``<= 64``. */
#define MSIM_UNIT_WORDS ((MSIM_MAX_U + 63) / 64)
typedef struct { uint64_t w[MSIM_UNIT_WORDS]; } msim_ubits;

static void msim_ubits_clear(msim_ubits *b)
{
    int32_t i;
    for (i = 0; i < (int32_t)MSIM_UNIT_WORDS; i++) b->w[i] = 0;
}

static void msim_ubits_set(msim_ubits *b, int32_t u)
{
    if (u < 0 || u >= MSIM_MAX_U) return;
    b->w[u >> 6] |= (uint64_t)1 << (u & 63);
}

static bool msim_ubits_test(const msim_ubits *b, int32_t u)
{
    if (u < 0 || u >= MSIM_MAX_U) return false;
    return ((b->w[u >> 6] >> (u & 63)) & (uint64_t)1) != 0;
}

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
    uint8_t coupled_comp;   /* scratch: 1 if in a coupled owner component */
} msim_task;

typedef struct {
    int32_t task;           /* owning slot */
    int32_t res;
    int32_t plen;
    int32_t pstart;
    int32_t mode;           /* partition group id, -1 otherwise */
    uint8_t widened;        /* 1 when placement is disjointly widened over cohort */
    double rem;
    double cap;
    double rate;
    double svc;             /* accumulated ``task.service`` for this link */
} msim_link;

typedef struct {
    uint64_t t_start_ns;
    uint64_t t_end_ns;
    uint64_t task_id;
    uint32_t pipe_id;
    uint32_t owner_id;
    uint16_t kind;
    uint16_t reason_code;
    double work_done;
} msim_obs;

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
    uint32_t fifo_epoch, coef_epoch, mode_epoch, seen_epoch, comp_epoch, ord_epoch;
    uint32_t *res_fifo_ep;
    uint32_t *res_coef_ep;
    uint32_t *res_mode_ep;
    uint32_t *res_seen_ep;
    uint32_t *res_comp_ep;
    uint32_t *res_ord_ep;
    int32_t *res_chead;
    int32_t *res_ctail;
    int32_t *touched_res;
    int32_t *res_lhead;
    int32_t *res_ltail;
    msim_ubits *res_hot_units;
    int32_t *coef_next;
    int32_t *coef_flow;
    int32_t *link_rnext;
    int32_t link_rnext_cap;
    int32_t *act_links;
    int32_t act_links_cap;
    int32_t *comp_q;
    int32_t comp_q_cap;
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
    int32_t *new_ready;
    int32_t new_ready_n, new_ready_cap;
    int32_t *act_res;
    uint8_t *res_seen;
    int32_t res_seen_cap;
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
    /* Demand-grown slot solver scratch.  Never on the C stack and never a
     * reuse of the ``MSIM_MAX_U``-sized per-resource accumulators: a bounded
     * stack worker must be able to run the coupled solver safely. */
    double *slot_rem;
    double *slot_slope;
    double *slot_bounds;
    int32_t slot_cap;
    msim_row *row;
    int32_t *ohead;
    int32_t *onext;
    int32_t *oslot;
    int32_t own_cap;          /* grown owner/entry scratch capacity */
    /* Caller-owned observation buffer (non-owning pointer, never freed). */
    msim_obs *obs_buf;
    uint32_t obs_cap, obs_count, obs_overflow;
    /* Prepared-terms epoch solver tables (owned by ctx). */
    void *pt_rows;
    void *pt_progs;
    void *pt_terms;
    void *pt_dels;
    double *pt_ctrs;
    int32_t pt_nrows, pt_row_cap;
    int32_t pt_nprogs, pt_prog_cap;
    int32_t pt_nterms, pt_term_cap;
    int32_t pt_ndels, pt_del_cap;
    int32_t pt_nctrs, pt_ctr_cap;
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

/* Demand-grown coupled-solver slots.  ``need`` is the exact slot count of this
 * arbitration pass; every array is grown before any solver cell is written, so
 * an allocation failure leaves the previous context untouched and the caller
 * reports a bounded arithmetic status instead of faulting. */
static bool msim_slot_reserve(msim_ctx *c, int32_t need)
{
    int32_t target;
    msim_slot *next_slot;
    double *next_rem, *next_slope, *next_bounds;
    bool *next_changed;
    if (need < 0) return false;
    if (need <= c->slot_cap) return true;
    if (need > MSIM_MAX_SLOT) return false;
    target = c->slot_cap > 0 ? c->slot_cap : 64;
    while (target < need) {
        if (target > MSIM_MAX_SLOT / 2) { target = need; break; }
        target *= 2;
    }
    if (target > MSIM_MAX_SLOT) target = MSIM_MAX_SLOT;
    if (target < need) return false;
    next_slot = (msim_slot *)realloc(c->slot, (size_t)target * sizeof(msim_slot));
    if (next_slot == NULL) return false;
    c->slot = next_slot;
    next_rem = (double *)realloc(c->slot_rem, (size_t)target * sizeof(double));
    if (next_rem == NULL) return false;
    c->slot_rem = next_rem;
    next_slope = (double *)realloc(c->slot_slope, (size_t)target * sizeof(double));
    if (next_slope == NULL) return false;
    c->slot_slope = next_slope;
    next_bounds = (double *)realloc(c->slot_bounds, (size_t)target * sizeof(double));
    if (next_bounds == NULL) return false;
    c->slot_bounds = next_bounds;
    next_changed = (bool *)realloc(c->changed, (size_t)target * sizeof(bool));
    if (next_changed == NULL) return false;
    c->changed = next_changed;
    c->slot_cap = target;
    return true;
}

/* Amortized O(1) monotonic epoch reset: zeroes ``arr`` only on 32-bit wraparound. */
static void msim_next_epoch(uint32_t *epoch, uint32_t *arr, int32_t n)
{
    (*epoch)++;
    if (*epoch == 0u) {
        if (n > 0 && arr != NULL) memset(arr, 0, (size_t)n * sizeof(uint32_t));
        *epoch = 1u;
    }
}

static int32_t msim_res_mode_get(const msim_ctx *c, int32_t r)
{
    return (c->res_mode_ep[r] == c->mode_epoch) ? c->res[r].mode : -1;
}

static void msim_res_mode_set(msim_ctx *c, int32_t r, int32_t mode)
{
    c->res_mode_ep[r] = c->mode_epoch;
    c->res[r].mode = mode;
}

static void msim_reset_epochs(msim_ctx *c)
{
    int32_t n = c->counter_cap;
    c->fifo_epoch = 0u;
    c->coef_epoch = 0u;
    c->mode_epoch = 0u;
    c->seen_epoch = 0u;
    c->comp_epoch = 0u;
    c->ord_epoch = 0u;
    if (n > 0) {
        if (c->res_fifo_ep) memset(c->res_fifo_ep, 0, (size_t)n * sizeof(uint32_t));
        if (c->res_coef_ep) memset(c->res_coef_ep, 0, (size_t)n * sizeof(uint32_t));
        if (c->res_mode_ep) memset(c->res_mode_ep, 0, (size_t)n * sizeof(uint32_t));
        if (c->res_seen_ep) memset(c->res_seen_ep, 0, (size_t)n * sizeof(uint32_t));
        if (c->res_comp_ep) memset(c->res_comp_ep, 0, (size_t)n * sizeof(uint32_t));
        if (c->res_ord_ep) memset(c->res_ord_ep, 0, (size_t)n * sizeof(uint32_t));
    }
}

/* Demand-grow all per-resource arrays cleanly and zero-init newly grown epoch tails. */
static bool msim_res_reserve(msim_ctx *c, int32_t need)
{
    int32_t old_cap = c->counter_cap, target;
    double *nbusy, *nserv;
    int32_t *nact_res, *nfifo, *nchead, *nctail, *ntouched, *nlhead, *nltail;
    uint8_t *nseen;
    uint32_t *efifo, *ecoef, *emode, *eseen, *ecomp, *eord;
    msim_ubits *nhot;
    if (need < 0 || need > MSIM_MAX_RES) return false;
    if (!msim_grow((void **)&c->res, &c->res_cap, need, sizeof(msim_res)))
        return false;
    if (need <= old_cap) return true;
    target = old_cap > 0 ? old_cap : 64;
    while (target < need) {
        if (target > MSIM_MAX_RES / 2) { target = need; break; }
        target *= 2;
    }
    if (target > MSIM_MAX_RES) target = MSIM_MAX_RES;
    if (target < need) return false;
    nbusy = (double *)realloc(c->busy, (size_t)target * sizeof(double));
    if (!nbusy) return false;
    c->busy = nbusy;
    nserv = (double *)realloc(c->serv, (size_t)target * sizeof(double));
    if (!nserv) return false;
    c->serv = nserv;
    nact_res = (int32_t *)realloc(c->act_res, (size_t)target * sizeof(int32_t));
    if (!nact_res) return false;
    c->act_res = nact_res;
    nseen = (uint8_t *)realloc(c->res_seen, (size_t)target * sizeof(uint8_t));
    if (!nseen) return false;
    c->res_seen = nseen;
    nfifo = (int32_t *)realloc(c->fifo_first, (size_t)target * sizeof(int32_t));
    if (!nfifo) return false;
    c->fifo_first = nfifo;
    nchead = (int32_t *)realloc(c->res_chead, (size_t)target * sizeof(int32_t));
    if (!nchead) return false;
    c->res_chead = nchead;
    nctail = (int32_t *)realloc(c->res_ctail, (size_t)target * sizeof(int32_t));
    if (!nctail) return false;
    c->res_ctail = nctail;
    ntouched = (int32_t *)realloc(c->touched_res, (size_t)target * sizeof(int32_t));
    if (!ntouched) return false;
    c->touched_res = ntouched;
    nlhead = (int32_t *)realloc(c->res_lhead, (size_t)target * sizeof(int32_t));
    if (!nlhead) return false;
    c->res_lhead = nlhead;
    nltail = (int32_t *)realloc(c->res_ltail, (size_t)target * sizeof(int32_t));
    if (!nltail) return false;
    c->res_ltail = nltail;
    nhot = (msim_ubits *)realloc(c->res_hot_units, (size_t)target * sizeof(msim_ubits));
    if (!nhot) return false;
    c->res_hot_units = nhot;
    efifo = (uint32_t *)realloc(c->res_fifo_ep, (size_t)target * sizeof(uint32_t));
    if (!efifo) return false;
    c->res_fifo_ep = efifo;
    ecoef = (uint32_t *)realloc(c->res_coef_ep, (size_t)target * sizeof(uint32_t));
    if (!ecoef) return false;
    c->res_coef_ep = ecoef;
    emode = (uint32_t *)realloc(c->res_mode_ep, (size_t)target * sizeof(uint32_t));
    if (!emode) return false;
    c->res_mode_ep = emode;
    eseen = (uint32_t *)realloc(c->res_seen_ep, (size_t)target * sizeof(uint32_t));
    if (!eseen) return false;
    c->res_seen_ep = eseen;
    ecomp = (uint32_t *)realloc(c->res_comp_ep, (size_t)target * sizeof(uint32_t));
    if (!ecomp) return false;
    c->res_comp_ep = ecomp;
    eord = (uint32_t *)realloc(c->res_ord_ep, (size_t)target * sizeof(uint32_t));
    if (!eord) return false;
    c->res_ord_ep = eord;
    {
        size_t tail = (size_t)(target - old_cap);
        memset(c->res_fifo_ep + old_cap, 0, tail * sizeof(uint32_t));
        memset(c->res_coef_ep + old_cap, 0, tail * sizeof(uint32_t));
        memset(c->res_mode_ep + old_cap, 0, tail * sizeof(uint32_t));
        memset(c->res_seen_ep + old_cap, 0, tail * sizeof(uint32_t));
        memset(c->res_comp_ep + old_cap, 0, tail * sizeof(uint32_t));
        memset(c->res_ord_ep + old_cap, 0, tail * sizeof(uint32_t));
    }
    c->res_seen_cap = target;
    c->counter_cap = target;
    return true;
}

/* In-place O(n log n) ascending sort with zero recursion. */
static void msim_heapsort_i32(int32_t *a, int32_t n)
{
    int32_t i;
    if (n <= 1) return;
    if (n <= 16) {
        for (i = 1; i < n; i++) {
            int32_t key = a[i], j = i - 1;
            while (j >= 0 && a[j] > key) { a[j + 1] = a[j]; j--; }
            a[j + 1] = key;
        }
        return;
    }
    for (i = n / 2 - 1; i >= 0; i--) {
        int32_t root = i;
        for (;;) {
            int32_t ch = 2 * root + 1, sw = root;
            if (ch < n && a[ch] > a[sw]) sw = ch;
            if (ch + 1 < n && a[ch + 1] > a[sw]) sw = ch + 1;
            if (sw == root) break;
            { int32_t tmp = a[root]; a[root] = a[sw]; a[sw] = tmp; }
            root = sw;
        }
    }
    for (i = n - 1; i > 0; i--) {
        int32_t tmp = a[0], root = 0;
        a[0] = a[i]; a[i] = tmp;
        for (;;) {
            int32_t ch = 2 * root + 1, sw = root;
            if (ch < i && a[ch] > a[sw]) sw = ch;
            if (ch + 1 < i && a[ch + 1] > a[sw]) sw = ch + 1;
            if (sw == root) break;
            { int32_t t2 = a[root]; a[root] = a[sw]; a[sw] = t2; }
            root = sw;
        }
    }
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
    free(c->res_fifo_ep); free(c->res_coef_ep); free(c->res_mode_ep);
    free(c->res_seen_ep); free(c->res_comp_ep); free(c->res_ord_ep);
    free(c->res_chead); free(c->res_ctail); free(c->touched_res);
    free(c->res_lhead); free(c->res_ltail); free(c->res_hot_units);
    free(c->coef_next); free(c->coef_flow);
    free(c->link_rnext); free(c->act_links); free(c->comp_q);
    free(c->task); free(c->free); free(c->link); free(c->units);
    free(c->heap); free(c->rheap); free(c->cheap); free(c->act);
    free(c->new_ready); free(c->act_res); free(c->res_seen);
    free(c->sidx); free(c->snext); free(c->sseq);
    free(c->idx); free(c->gid); free(c->rep); free(c->pend); free(c->surv);
    free(c->fifo_first); free(c->uhead); free(c->unext); free(c->uflow);
    free(c->uamt); free(c->over); free(c->remu); free(c->bounds); free(c->slope);
    free(c->active); free(c->active2); free(c->changed);
    free(c->coef); free(c->flow); free(c->entry); free(c->slot); free(c->row);
    free(c->slot_rem); free(c->slot_slope); free(c->slot_bounds);
    free(c->ohead); free(c->onext); free(c->oslot);
    free(c->pt_rows); free(c->pt_progs); free(c->pt_terms);
    free(c->pt_dels); free(c->pt_ctrs);
    memset(c, 0, sizeof *c);
    free(c);
}

/* One-time allocation of the bounded solver workspace. */
static bool msim_ctx_scratch(msim_ctx *c)
{
    c->coef = (msim_coef *)msim_zalloc((size_t)MSIM_MAX_COEF, sizeof(msim_coef));
    c->coef_next = (int32_t *)msim_zalloc((size_t)MSIM_MAX_COEF, sizeof(int32_t));
    c->coef_flow = (int32_t *)msim_zalloc((size_t)MSIM_MAX_COEF, sizeof(int32_t));
    c->flow = (msim_flow *)msim_zalloc((size_t)MSIM_MAX_F, sizeof(msim_flow));
    c->entry = NULL;
    c->slot = NULL;
    c->slot_rem = NULL;
    c->slot_slope = NULL;
    c->slot_bounds = NULL;
    c->changed = NULL;
    c->slot_cap = 0;
    c->row = (msim_row *)msim_zalloc((size_t)MSIM_MAX_MROW, sizeof(msim_row));
    c->ohead = (int32_t *)msim_zalloc((size_t)MSIM_MAX_F, sizeof(int32_t));
    c->onext = NULL;
    c->oslot = NULL;
    c->idx = (int32_t *)msim_zalloc((size_t)MSIM_MAX_L, sizeof(int32_t));
    c->gid = (int32_t *)msim_zalloc((size_t)MSIM_MAX_L, sizeof(int32_t));
    c->rep = (int32_t *)msim_zalloc((size_t)MSIM_MAX_L, sizeof(int32_t));
    c->pend = (int32_t *)msim_zalloc((size_t)MSIM_MAX_L, sizeof(int32_t));
    c->surv = (int32_t *)msim_zalloc((size_t)MSIM_MAX_L, sizeof(int32_t));
    c->fifo_first = NULL;
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
    c->task = (msim_task *)msim_zalloc((size_t)MSIM_MAX_T, sizeof(msim_task));
    c->task_cap = MSIM_MAX_T;
    return c->coef && c->coef_next && c->coef_flow && c->flow && c->row && c->ohead
        && c->idx && c->gid && c->rep && c->pend
        && c->surv && c->uhead && c->over
        && c->remu && c->bounds && c->slope
        && c->active && c->active2 && c->task;
}

#endif /* MSIM_ENGINE_ACCEL_CORE_H */
