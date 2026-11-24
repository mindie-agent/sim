/* Client-compiled C11 arena kernel for the one causal Engine (engine-arena/2).
 *
 * [Role] Maintain one Engine's live numeric frontier -- resource rates,
 *   incremental pure-DAG submission, exact next-event integration and
 *   multi-step settlement -- inside a heap-owned ``msim_ctx``.  Python keeps
 *   clock authority, callbacks, allocations and the memory ledger; the kernel
 *   only advances callback-free numeric windows and returns at every
 *   Python-visible boundary.
 * [Invariants] Pure ISO C11.  Zero mutable file-scope objects: every cell lives
 *   in the caller's context, so concurrent Engines are disjoint by
 *   construction.  Binary64 arithmetic follows the Python reference operation
 *   order exactly (no fast-math, contraction or reassociation), including the
 *   per-resource service accumulator order.  All capacities and spans are
 *   validated before the first mutation; a full output buffer is reported,
 *   never truncated.  64-bit task sequences are never truncated.
 * [Timeouts & Locks] No threads, locks, sleeps or I/O.  Every loop is bounded by
 *   the task/link tables, a structural guard or the caller's step budget.
 * [Side-Effect Policy] Writes only the context and the caller-provided output
 *   buffers; no pointer is retained past the call that supplied it.
 */
#include "_engine_accel_core.h"

MSIM_EXPORT int32_t msim_ctx_reset(void *ctx);
static void remove_active(msim_ctx *c, msim_task *t);

/* -- validated accessors ------------------------------------------------ */
static bool ctx_ok(const msim_ctx *c)
{
    return c != NULL && c->magic == 0x4d53494dU && c->alive == 1;
}

static int32_t reject(int32_t *st, size_t nst, int32_t code)
{
    if (st != NULL && nst >= 2) { st[0] = MSIM_ST_FAILED; st[1] = code; }
    return -1;
}

static bool slot_ok(const msim_ctx *c, int64_t slot)
{
    return slot >= 0 && slot < (int64_t)c->task_cap
        && c->task[slot].state != 2;
}

/* Effective membership: an absent placement owns every declared unit. */
static int32_t unit_len(const msim_ctx *c, const msim_link *l)
{
    return l->plen > 0 ? l->plen : c->res[l->res].units;
}

static int32_t unit_at(const msim_ctx *c, const msim_link *l, int32_t k)
{
    return l->plen > 0 ? (int32_t)c->units[l->pstart + k] : k;
}

/* Equal placement tuples compare by value, never by CSR offsets. */
static bool same_units(const msim_ctx *c, const msim_link *a, const msim_link *b)
{
    int32_t n = unit_len(c, a), i;
    if (n != unit_len(c, b)) return false;
    for (i = 0; i < n; i++) if (unit_at(c, a, i) != unit_at(c, b, i)) return false;
    return true;
}

/* Python fifo_winner proxy: (actual start, priority, sequence). */
static bool proxy_less(const msim_ctx *c, int32_t a, int32_t b)
{
    const msim_task *x = &c->task[a], *y = &c->task[b];
    if (x->start != y->start) return x->start < y->start;
    if (x->prio != y->prio) return x->prio < y->prio;
    return x->seq < y->seq;
}

/* -- bounded index heaps ------------------------------------------------ */
enum { HEAP_READY = 0, HEAP_RELEASE = 1, HEAP_COMPL = 2 };

static bool heap_less(const msim_ctx *c, int which, int32_t a, int32_t b)
{
    const msim_task *x = &c->task[a], *y = &c->task[b];
    if (which == HEAP_READY) {
        if (x->prio != y->prio) return x->prio < y->prio;
        return x->seq < y->seq;
    }
    if (which == HEAP_RELEASE) {
        if (x->release != y->release) return x->release < y->release;
        return x->seq < y->seq;
    }
    return x->seq < y->seq;
}

static int32_t **heap_ptr(msim_ctx *c, int which, int32_t **n, int32_t **cap)
{
    if (which == HEAP_READY) { *n = &c->heap_n; *cap = &c->heap_cap; return &c->heap; }
    if (which == HEAP_RELEASE) { *n = &c->rheap_n; *cap = &c->rheap_cap; return &c->rheap; }
    *n = &c->cheap_n; *cap = &c->cheap_cap; return &c->cheap;
}

static bool heap_push(msim_ctx *c, int which, int32_t idx)
{
    int32_t *n, *cap, **slot = heap_ptr(c, which, &n, &cap);
    int32_t i;
    if (!msim_grow((void **)slot, cap, *n + 1, sizeof(int32_t))) return false;
    i = (*n)++;
    (*slot)[i] = idx;
    while (i > 0) {
        int32_t parent = (i - 1) / 2;
        if (!heap_less(c, which, (*slot)[i], (*slot)[parent])) break;
        { int32_t tmp = (*slot)[i]; (*slot)[i] = (*slot)[parent]; (*slot)[parent] = tmp; }
        i = parent;
    }
    return true;
}

static int32_t heap_pop(msim_ctx *c, int which)
{
    int32_t *n, *cap, **slot = heap_ptr(c, which, &n, &cap);
    int32_t top, i = 0;
    if (*n <= 0) return -1;
    top = (*slot)[0];
    (*n)--;
    if (*n == 0) return top;
    (*slot)[0] = (*slot)[*n];
    for (;;) {
        int32_t l = 2 * i + 1, r = l + 1, best = i;
        if (l < *n && heap_less(c, which, (*slot)[l], (*slot)[best])) best = l;
        if (r < *n && heap_less(c, which, (*slot)[r], (*slot)[best])) best = r;
        if (best == i) break;
        { int32_t tmp = (*slot)[i]; (*slot)[i] = (*slot)[best]; (*slot)[best] = tmp; }
        i = best;
    }
    return top;
}

#include "_engine_accel_rates.h"

/* Match Python ``components.items()`` order per resource so binary64 service
 * totals are bit-identical to ``Engine.advance``. */
static bool is_consumer(const msim_ctx *c, const msim_link *l)
{
    if (l->rem <= 0.0) return false;
    if (c->task[l->task].state != 1) return false;
    if (c->res[l->res].mode == 4 && l->rate <= 0.0) return false;
    return true;
}

static int32_t build_order(msim_ctx *c, int32_t r, int32_t *n_out)
{
    /* LEGACY_PURGED(S6-W5-23): deleted per-resource O(nres*nactive) active-task scan
     * and insertion-sort order rebuild; implement incremental sparse active-resource
     * order buckets updated on task admission/completion */
    (void)c;
    (void)r;
    *n_out = 0;
    return MSIM_E_STATE;
}

#include "_engine_accel_submit.h"
#include "_engine_accel_terms.h"
#include "_engine_accel_coupled.h"
#include "_engine_accel_abi.h"
