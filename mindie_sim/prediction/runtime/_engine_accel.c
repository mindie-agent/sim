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

/* Python fifo_winner(caps, tasks, resource_name): active holder on this
 * resource (svc > 0), then priority, then actual start, then sequence. */
static bool fifo_link_less(const msim_ctx *c, int32_t la, int32_t lb)
{
    const msim_link *xa = &c->link[la], *xb = &c->link[lb];
    bool ha = xa->svc > 0.0, hb = xb->svc > 0.0;
    const msim_task *ta = &c->task[xa->task], *tb = &c->task[xb->task];
    if (ha != hb) return ha;
    if (ta->prio != tb->prio) return ta->prio < tb->prio;
    if (ta->start != tb->start) return ta->start < tb->start;
    return ta->seq < tb->seq;
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
    if (l->coupled_comp && l->rate <= 0.0) return false;
    return true;
}

static bool build_order_greater(const msim_ctx *c, int32_t rmode, int32_t a, int32_t b)
{
    const msim_link *pa = &c->link[a], *pb = &c->link[b];
    if (rmode == 4 || rmode == 3) {
        if (c->task[pa->task].seq != c->task[pb->task].seq)
            return c->task[pa->task].seq > c->task[pb->task].seq;
        return a > b;
    }
    if (rmode == 2) {
        int32_t ga = pa->mode >= 0 ? pa->mode : 0;
        int32_t gb = pb->mode >= 0 ? pb->mode : 0;
        if (ga != gb) return ga > gb;
        if (pa->cap != pb->cap) return pa->cap > pb->cap;
        if (c->task[pa->task].seq != c->task[pb->task].seq)
            return c->task[pa->task].seq > c->task[pb->task].seq;
        return a > b;
    }
    if (pa->cap != pb->cap) return pa->cap > pb->cap;
    if (c->task[pa->task].seq != c->task[pb->task].seq)
        return c->task[pa->task].seq > c->task[pb->task].seq;
    return a > b;
}

/* Sparse per-resource consumer ordering over ``res_lhead[r]`` (no scan of ``act_n``). */
static int32_t build_order(msim_ctx *c, int32_t r, int32_t *n_out)
{
    int32_t rmode = msim_res_mode_get(c, r);
    int32_t *order = c->idx, n = 0, li, i;
    if (c->res_ord_ep[r] == c->ord_epoch) {
        for (li = c->res_lhead[r]; li >= 0; li = c->link_rnext[li]) {
            if (n >= MSIM_MAX_L) return MSIM_E_ARITH;
            order[n++] = li;
        }
    }
    if (n <= 16) {
        for (i = 1; i < n; i++) {
            int32_t key = order[i], j = i - 1;
            while (j >= 0 && build_order_greater(c, rmode, order[j], key)) {
                order[j + 1] = order[j];
                j--;
            }
            order[j + 1] = key;
        }
    } else {
        for (i = n / 2 - 1; i >= 0; i--) {
            int32_t root = i;
            for (;;) {
                int32_t ch = 2 * root + 1, sw = root;
                if (ch < n && build_order_greater(c, rmode, order[ch], order[sw])) sw = ch;
                if (ch + 1 < n && build_order_greater(c, rmode, order[ch + 1], order[sw])) sw = ch + 1;
                if (sw == root) break;
                { int32_t tmp = order[root]; order[root] = order[sw]; order[sw] = tmp; }
                root = sw;
            }
        }
        for (i = n - 1; i > 0; i--) {
            int32_t tmp = order[0], root = 0;
            order[0] = order[i]; order[i] = tmp;
            for (;;) {
                int32_t ch = 2 * root + 1, sw = root;
                if (ch < i && build_order_greater(c, rmode, order[ch], order[sw])) sw = ch;
                if (ch + 1 < i && build_order_greater(c, rmode, order[ch + 1], order[sw])) sw = ch + 1;
                if (sw == root) break;
                { int32_t t2 = order[root]; order[root] = order[sw]; order[sw] = t2; }
                root = sw;
            }
        }
    }
    *n_out = n;
    return 0;
}

#include "_engine_accel_submit.h"
#include "_engine_accel_terms.h"
#include "_engine_accel_coupled.h"
#include "_engine_accel_abi.h"
