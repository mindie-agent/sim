/* Exported ABI surface of the client-compiled C11 arena kernel (engine-arena/2).
 *
 * [Role] Context lifecycle, incremental submission, epoch execution and the
 *   bounded counter/export accessors for one Engine.
 * [Invariants] Every writable span and every table length is validated before
 *   the first mutation; a full buffer is reported, never truncated; 64-bit
 *   sequences are preserved end to end.  All state is reachable only through
 *   the caller's ``msim_ctx``.
 * [Timeouts & Locks] No threads, locks, sleeps or I/O; every loop is bounded by
 *   a table, a structural guard or the caller's step budget.
 * [Side-Effect Policy] Writes only the context and the caller-provided output
 *   buffers.
 */
#ifndef MSIM_ENGINE_ACCEL_ABI_H
#define MSIM_ENGINE_ACCEL_ABI_H

/* -- ABI ----------------------------------------------------------------- */
MSIM_EXPORT uint32_t msim_abi_version(void) { return (uint32_t)MSIM_ABI_VERSION; }

MSIM_EXPORT int32_t msim_abi_probe(uint64_t *out, size_t nout)
{
    static const double one = 1.0;
    int64_t negative = -1;
    if (out == NULL || nout < 16) return -1;
    out[0] = (uint64_t)MSIM_ABI_VERSION;
    out[1] = (uint64_t)sizeof(double);
    out[2] = (uint64_t)sizeof(int32_t);
    out[3] = (uint64_t)sizeof(uint64_t);
    out[4] = (uint64_t)sizeof(void *);
    out[5] = (uint64_t)MSIM_HDR_CELLS;
    out[6] = (uint64_t)MSIM_TASK_STRIDE;
    out[7] = (uint64_t)MSIM_LINK_STRIDE;
    out[8] = (uint64_t)MSIM_COMP_STRIDE;
    out[9] = (uint64_t)MSIM_SVC_STRIDE;
    out[10] = msim_to_bits(msim_ulp(1.0));
    out[11] = msim_to_bits(0x1p-1074);
    out[12] = msim_to_bits(nextafter(one, 2.0) - one);
    out[13] = msim_to_bits(one / 3.0);
    out[14] = (uint64_t)(negative == -1 ? 1 : 0);
    out[15] = (uint64_t)MSIM_MAX_R;
    return 0;
}

MSIM_EXPORT void *msim_ctx_create(void)
{
    msim_ctx *c = (msim_ctx *)calloc(1, sizeof(msim_ctx));
    if (c == NULL) return NULL;
    msim_ctx_zero(c);
    if (!msim_ctx_scratch(c)) { msim_ctx_release(c); return NULL; }
    if (msim_ctx_reset(c) != 0) { msim_ctx_release(c); return NULL; }
    return c;
}

MSIM_EXPORT void msim_ctx_free(void *ctx)
{
    msim_ctx *c = (msim_ctx *)ctx;
    if (!ctx_ok(c)) return;
    c->alive = 0;
    msim_ctx_release(c);
}

MSIM_EXPORT int32_t msim_ctx_reset(void *ctx)
{
    msim_ctx *c = (msim_ctx *)ctx;
    int32_t i;
    if (!ctx_ok(c)) return -1;
    c->nl = 0; c->nu = 0; c->live_n = 0; c->free_n = 0; c->sn = 0;
    c->heap_n = 0; c->rheap_n = 0; c->cheap_n = 0; c->act_n = 0;
    c->new_ready_n = 0;
    c->resume_compl = 0;
    c->pending_hold = 0;
    c->obs_count = 0; c->obs_overflow = 0;
    c->pt_nprogs = 0; c->pt_nterms = 0; c->pt_ndels = 0; c->pt_nctrs = 0;
    if (!msim_grow((void **)&c->free, &c->free_cap, c->task_cap, sizeof(int32_t)))
        return -1;
    for (i = 0; i < c->task_cap; i++) {
        c->task[i].state = 2;
        c->task[i].link_block = -1;
        c->task[i].link_block_len = 0;
        c->task[i].act_slot = -1;
        c->free[i] = i;
    }
    c->free_n = c->task_cap;
    for (i = 0; i < c->nres; i++) { c->busy[i] = 0.0; c->serv[i] = 0.0; }
    return 0;
}

MSIM_EXPORT int32_t msim_ctx_config(void *ctx, const double *rates, const int32_t *units,
                        const int32_t *policy, size_t nres, int32_t *st, size_t nst)
{
    msim_ctx *c = (msim_ctx *)ctx;
    size_t i;
    if (st != NULL && nst >= 2) { st[0] = 0; st[1] = 0; }
    if (!ctx_ok(c)) return reject(st, nst, MSIM_E_STATE);
    if (nres < 1 || nres > (size_t)MSIM_MAX_R) return reject(st, nst, MSIM_E_RANGE);
    if (rates == NULL || units == NULL || policy == NULL) return reject(st, nst, MSIM_E_RANGE);
    for (i = 0; i < nres; i++) {
        if (!isfinite(rates[i]) || rates[i] <= 0.0) return reject(st, nst, MSIM_E_VALUE);
        if (policy[i] != 0 && policy[i] != 1) return reject(st, nst, MSIM_E_VALUE);
        if (units[i] < 1 || units[i] > MSIM_MAX_U) return reject(st, nst, MSIM_E_VALUE);
    }
    if (!msim_grow((void **)&c->res, &c->res_cap, (int32_t)nres, sizeof(msim_res)))
        return reject(st, nst, MSIM_E_RANGE);
    if (c->counter_cap < (int32_t)nres) {
        int32_t target = nres < 64 ? 64 : (int32_t)nres;
        double *next_busy = (double *)realloc(c->busy, (size_t)target * sizeof(double));
        double *next_serv;
        int32_t *next_act_res;
        uint8_t *next_res_seen;
        if (next_busy == NULL) return reject(st, nst, MSIM_E_RANGE);
        c->busy = next_busy;
        next_serv = (double *)realloc(c->serv, (size_t)target * sizeof(double));
        if (next_serv == NULL) return reject(st, nst, MSIM_E_RANGE);
        c->serv = next_serv;
        next_act_res = (int32_t *)realloc(c->act_res, (size_t)target * sizeof(int32_t));
        if (next_act_res == NULL) return reject(st, nst, MSIM_E_RANGE);
        c->act_res = next_act_res;
        next_res_seen = (uint8_t *)realloc(c->res_seen, (size_t)target * sizeof(uint8_t));
        if (next_res_seen == NULL) return reject(st, nst, MSIM_E_RANGE);
        c->res_seen = next_res_seen;
        c->res_seen_cap = target;
        c->counter_cap = target;
    }
    for (i = 0; i < nres; i++) {
        c->res[i].rate = rates[i];
        c->res[i].units = units[i];
        c->res[i].policy = policy[i];
        c->res[i].mode = -1;
        c->busy[i] = 0.0;
        c->serv[i] = 0.0;
    }
    c->nres = (int32_t)nres;
    return 0;
}

MSIM_EXPORT int32_t msim_ctx_seed(void *ctx, const double *busy, const double *serv, size_t nres)
{
    msim_ctx *c = (msim_ctx *)ctx;
    size_t i;
    if (!ctx_ok(c) || (int32_t)nres != c->nres) return -1;
    if (busy == NULL || serv == NULL) return -1;
    for (i = 0; i < nres; i++) {
        if (!isfinite(busy[i]) || !isfinite(serv[i])) return -1;
        c->busy[i] = busy[i];
        c->serv[i] = serv[i];
    }
    return 0;
}

MSIM_EXPORT int32_t msim_ctx_counters(void *ctx, double *busy, double *serv, size_t nres)
{
    msim_ctx *c = (msim_ctx *)ctx;
    size_t i;
    if (!ctx_ok(c) || (int32_t)nres != c->nres) return -1;
    if (busy == NULL || serv == NULL) return -1;
    for (i = 0; i < nres; i++) { busy[i] = c->busy[i]; serv[i] = c->serv[i]; }
    return 0;
}

/* Admit a task the Python reference started (allocation admission owner). */
MSIM_EXPORT int32_t msim_ctx_activate(void *ctx, const int32_t *slots, size_t n, double now,
                          int32_t *st, size_t nst)
{
    msim_ctx *c = (msim_ctx *)ctx;
    size_t i;
    if (st != NULL && nst >= 2) { st[0] = 0; st[1] = 0; }
    if (!ctx_ok(c)) return reject(st, nst, MSIM_E_STATE);
    if (!isfinite(now)) return reject(st, nst, MSIM_E_VALUE);
    if (n > 0 && slots == NULL) return reject(st, nst, MSIM_E_RANGE);
    for (i = 0; i < n; i++) {
        msim_task *t;
        if (!slot_ok(c, slots[i])) continue;
        t = &c->task[slots[i]];
        if (t->state != 0) continue;
        t->state = 1;
        t->start = now;
        t->act_slot = -1;
        if (!msim_grow((void **)&c->act, &c->act_cap, c->act_n + 1, sizeof(int32_t)))
            return reject(st, nst, MSIM_E_RANGE);
        t->act_slot = c->act_n;
        c->act[c->act_n++] = slots[i];
    }
    return 0;
}

/* Post-submission edge registration: the flush assigns slots first, then wires
 * every mirrored predecessor -> successor pair in one bounded call.
 *
 * Strict two-phase commit:  Phase 1 validates every tuple,
 * pointer, capacity and sequence with zero graph mutation; phase 2 commits the
 * whole batch only after phase 1 accepted it.  A rejected suffix therefore
 * leaves the context byte-equivalent to one that never saw the batch. */
MSIM_EXPORT int32_t msim_ctx_link(void *ctx, const int32_t *pred, const int32_t *succ,
                      const uint64_t *sseq, size_t n, int32_t *st, size_t nst)
{
    msim_ctx *c = (msim_ctx *)ctx;
    size_t i;
    if (st != NULL && nst >= 2) { st[0] = 0; st[1] = 0; }
    if (!ctx_ok(c)) return reject(st, nst, MSIM_E_STATE);
    if (n == 0) return 0;
    if (pred == NULL || succ == NULL || sseq == NULL)
        return reject(st, nst, MSIM_E_RANGE);
    if (n > (size_t)INT32_MAX) return reject(st, nst, MSIM_E_RANGE);
    if (!msim_fits((size_t)c->sn, n, (size_t)MSIM_MAX_OWN))
        return reject(st, nst, MSIM_E_RANGE);
    /* Phase 1: read-only preflight of the entire batch. */
    for (i = 0; i < n; i++) {
        if (!slot_ok(c, pred[i]) || !slot_ok(c, succ[i]))
            return reject(st, nst, MSIM_E_RANGE);
        if (pred[i] == succ[i]) return reject(st, nst, MSIM_E_RANGE);
        if (c->task[succ[i]].seq != sseq[i])
            return reject(st, nst, MSIM_E_RANGE);
    }
    /* Phase 2: commit.  Capacity growth is scratch-only; no graph cell is
     * written until every realloc below has succeeded. */
    if (c->sn + (int32_t)n > c->s_cap) {
        int32_t target = c->s_cap > 0 ? c->s_cap : 64;
        int32_t *next_sidx, *next_snext;
        uint64_t *next_sseq;
        while (target < c->sn + (int32_t)n) target *= 2;
        next_sidx = (int32_t *)realloc(c->sidx, (size_t)target * sizeof(int32_t));
        if (next_sidx == NULL) return reject(st, nst, MSIM_E_RANGE);
        c->sidx = next_sidx;
        next_snext = (int32_t *)realloc(c->snext, (size_t)target * sizeof(int32_t));
        if (next_snext == NULL) return reject(st, nst, MSIM_E_RANGE);
        c->snext = next_snext;
        next_sseq = (uint64_t *)realloc(c->sseq, (size_t)target * sizeof(uint64_t));
        if (next_sseq == NULL) return reject(st, nst, MSIM_E_RANGE);
        c->sseq = next_sseq;
        c->s_cap = target;
    }
    for (i = 0; i < n; i++) {
        c->sidx[c->sn] = succ[i];
        c->sseq[c->sn] = sseq[i];
        c->snext[c->sn] = c->task[pred[i]].first_succ;
        c->task[pred[i]].first_succ = c->sn;
        c->sn++;
    }
    return 0;
}

MSIM_EXPORT int32_t msim_ctx_deliver(void *ctx, const int32_t *slots, size_t n, int32_t *st, size_t nst)
{
    msim_ctx *c = (msim_ctx *)ctx;
    size_t i;
    if (st != NULL && nst >= 2) { st[0] = 0; st[1] = 0; }
    if (!ctx_ok(c)) return reject(st, nst, MSIM_E_STATE);
    if (n > 0 && slots == NULL) return reject(st, nst, MSIM_E_RANGE);
    for (i = 0; i < n; i++) {
        msim_task *t;
        if (!slot_ok(c, slots[i])) continue;
        t = &c->task[slots[i]];
        if (t->state != 0) continue;
        if (t->waiting > 0) t->waiting--;
        if (t->waiting == 0 && !t->ready) {
            t->ready = 1;
            if (!heap_push(c, HEAP_READY, slots[i]))
                return reject(st, nst, MSIM_E_RANGE);
        }
    }
    return 0;
}

MSIM_EXPORT int32_t msim_ctx_drop(void *ctx, const int32_t *slots, size_t n, int32_t *st, size_t nst)
{
    msim_ctx *c = (msim_ctx *)ctx;
    size_t i;
    if (st != NULL && nst >= 2) { st[0] = 0; st[1] = 0; }
    if (!ctx_ok(c)) return reject(st, nst, MSIM_E_STATE);
    if (n > 0 && slots == NULL) return reject(st, nst, MSIM_E_RANGE);
    for (i = 0; i < n; i++) {
        msim_task *t;
        int32_t slot = slots[i];
        if (!slot_ok(c, slot)) continue;
        t = &c->task[slot];
        remove_active(c, t);
        t->state = 2;
        t->ready = 0;
        t->compl = 0;
        if (c->live_n > 0) c->live_n--;
        if (!msim_grow((void **)&c->free, &c->free_cap, c->free_n + 1, sizeof(int32_t)))
            return reject(st, nst, MSIM_E_RANGE);
        c->free[c->free_n++] = slot;
    }
    return 0;
}

MSIM_EXPORT int32_t msim_ctx_mark_external(void *ctx, const int32_t *slots, size_t n,
                               int32_t *st, size_t nst)
{
    msim_ctx *c = (msim_ctx *)ctx;
    size_t i;
    if (st != NULL && nst >= 2) { st[0] = 0; st[1] = 0; }
    if (!ctx_ok(c)) return reject(st, nst, MSIM_E_STATE);
    if (n > 0 && slots == NULL) return reject(st, nst, MSIM_E_RANGE);
    for (i = 0; i < n; i++)
        if (slot_ok(c, slots[i])) c->task[slots[i]].ext_waiters = 1;
    return 0;
}

#ifndef msim_abi_version
MSIM_EXPORT int32_t msim_ctx_mark_callback(void *ctx, const int32_t *slots, size_t n,
                               int32_t *st, size_t nst)
{
    msim_ctx *c = (msim_ctx *)ctx;
    size_t i;
    if (st != NULL && nst >= 2) { st[0] = 0; st[1] = 0; }
    if (!ctx_ok(c)) return reject(st, nst, MSIM_E_STATE);
    if (n > 0 && slots == NULL) return reject(st, nst, MSIM_E_RANGE);
    for (i = 0; i < n; i++)
        if (slot_ok(c, slots[i])) c->task[slots[i]].flags |= MSIM_F_CALLBACK;
    return 0;
}
#endif

MSIM_EXPORT int32_t msim_ctx_export(void *ctx, int32_t slot, double *out, size_t nout)
{
    msim_ctx *c = (msim_ctx *)ctx;
    int32_t k;
    if (!ctx_ok(c) || !slot_ok(c, slot) || out == NULL) return -1;
    {
        msim_task *t = &c->task[slot];
        if (!msim_fits(0, (size_t)t->nlink * 2, nout)) return -1;
        for (k = 0; k < (int32_t)t->nlink; k++) {
            out[k] = c->link[t->lstart + k].rem;
            out[(size_t)t->nlink + k] = c->link[t->lstart + k].svc;
        }
        return (int32_t)t->nlink;
    }
}

static bool task_completable(const msim_ctx *c, const msim_task *t, double now)
{
    int32_t k;
    if (t->state != 1) return false;
    for (k = 0; k < (int32_t)t->nlink; k++)
        if (c->link[t->lstart + k].rem != 0.0) return false;
    return t->start + t->latency <= now;
}

static bool emit_completion(msim_ctx *c, int32_t idx, double now,
                            uint64_t *comp, size_t comp_cap,
                            double *svc, size_t svc_cap,
                            int32_t *ncomp_out, int32_t *nsvc_out)
{
    msim_task *t = &c->task[idx];
    size_t need = (size_t)t->nlink, k;
    if (!msim_fits((size_t)(*ncomp_out), (size_t)1, comp_cap)) return false;
    if (!msim_fits((size_t)(*nsvc_out), need, svc_cap)) return false;
    comp[(size_t)(*ncomp_out) * MSIM_COMP_STRIDE + 0] = (uint64_t)t->seq;
    comp[(size_t)(*ncomp_out) * MSIM_COMP_STRIDE + 1] = msim_to_bits(now);
    comp[(size_t)(*ncomp_out) * MSIM_COMP_STRIDE + 2] = (uint64_t)(*nsvc_out);
    for (k = 0; k < need; k++) {
        msim_link *l = &c->link[t->lstart + k];
        svc[((size_t)(*nsvc_out) + k) * MSIM_SVC_STRIDE + 0] = (double)l->res;
        svc[((size_t)(*nsvc_out) + k) * MSIM_SVC_STRIDE + 1] = l->svc;
        svc[((size_t)(*nsvc_out) + k) * MSIM_SVC_STRIDE + 2] = l->rem;
    }
    (*ncomp_out)++;
    (*nsvc_out) += (int32_t)need;
    return true;
}

static void remove_active(msim_ctx *c, msim_task *t)
{
    int32_t slot = t->act_slot, last;
    if (slot < 0) return;
    last = c->act_n - 1;
    if (slot != last) {
        int32_t moved = c->act[last];
        c->act[slot] = moved;
        c->task[moved].act_slot = slot;
    }
    c->act_n = last;
    t->act_slot = -1;
}

/* One Engine.advance epoch: settle the mirrored frontier, then integrate.
 *
 * Status word (``st[0]``): 0 on success; ``MSIM_ST_FAILED`` on every
 * failure; additionally ``MSIM_ST_PREFIX`` exactly when this failing call
 * published ``io[0]``, ``st[2..7]`` (counts, records, fault resource) and
 * ``hdr[15]`` as one coherent accepted prefix.  A call with no accepted clock
 * advance, start, completion or step never sets ``MSIM_ST_PREFIX``.
 *
 * Step budget: ``hdr[2]`` counts positive-dt integration steps only.  The
 * whole zero-time settlement at the current instant (admissions, completed
 * tasks and their completions/service records) is always finished before the
 * boundary is decided.  ``MSIM_B_OUTPUT`` reports an output wire buffer that
 * filled mid-settlement; ``MSIM_B_BUDGET`` is reserved for a fully settled
 * instant whose next interval would exceed the caller's step budget.  A call
 * with ``hdr[2] == 0`` therefore drains every same-instant record without
 * integrating and never advances the clock. */
MSIM_EXPORT int32_t msim_ctx_run(void *ctx, uint64_t *hdr, size_t nhdr, int32_t *st, size_t nst,
                     double *io, size_t nio, uint64_t *comp, size_t ncomp,
                     double *svc, size_t nsvc, uint64_t *started, size_t nstarted)
{
    msim_ctx *c = (msim_ctx *)ctx;
    int32_t steps = 0, ncomp_out = 0, nsvc_out = 0, nstart = 0, boundary = MSIM_B_NONE;
    int32_t i, round, rc = 0, fault_res = 0, accepted, hold = 0;
    double now, horizon, next_rel, initial_now;
    int64_t budget;
    size_t comp_cap, svc_cap, start_cap;

    /* Clear every caller-visible status cell before validation
     * check, so a rejected call can never expose stale buffers as a prefix. */
    if (st != NULL) {
        for (i = 0; i < (int32_t)nst && i < (int32_t)MSIM_ST_CELLS; i++) st[i] = 0;
    }
    if (st == NULL || nst < (size_t)MSIM_ST_CELLS || hdr == NULL
            || nhdr < (size_t)MSIM_HDR_CELLS) return -1;
    if (!ctx_ok(c)) return reject(st, nst, MSIM_E_STATE);
    if (hdr[0] != (uint64_t)MSIM_ABI_VERSION) return reject(st, nst, MSIM_E_ABI);
    if (hdr[1] != 0) return reject(st, nst, MSIM_E_VALUE);
    if (io == NULL || nio < (size_t)MSIM_IO_CELLS) return reject(st, nst, MSIM_E_RANGE);
    if (comp == NULL || svc == NULL || started == NULL)
        return reject(st, nst, MSIM_E_RANGE);
    /* Validate every writable output span before the first mutation. */
    comp_cap = ncomp / MSIM_COMP_STRIDE;
    svc_cap = nsvc / MSIM_SVC_STRIDE;
    start_cap = nstarted / 2;
    if (comp_cap < 1 || svc_cap < 1 || start_cap < 1)
        return reject(st, nst, MSIM_E_FULL);
    /* Carry forward any admission hold latched before an output-buffer split;
     * still in force on this continuation call.  ``pending_hold`` is cleared
     * only once the interrupted completion snapshot has fully drained below. */
    hold = c->pending_hold;
    now = io[0]; horizon = io[1]; next_rel = io[3];
    initial_now = now;
    budget = (int64_t)hdr[2];
    if (!isfinite(now) || isnan(horizon) || isnan(next_rel))
        return reject(st, nst, MSIM_E_VALUE);
    if (next_rel < now) next_rel = now;
    if (budget < 0) budget = 0;
    if (budget > 1000000) budget = 1000000;
    for (i = 0; i < c->new_ready_n; i++) {
        int32_t slot = c->new_ready[i];
        msim_task *t = &c->task[slot];
        if (t->state != 0 || t->waiting != 0 || t->ready) continue;
        t->ready = 1;
        if (t->release <= now) {
            if (!heap_push(c, HEAP_READY, slot)) { rc = MSIM_E_RANGE; goto fail; }
        } else {
            if (!heap_push(c, HEAP_RELEASE, slot)) { rc = MSIM_E_RANGE; goto fail; }
        }
    }
    c->new_ready_n = 0;
    for (;;) {
        for (round = 0; round < 100000; round++) {
            int32_t idx;
            /* Output buffer filled mid-snapshot: resume draining that exact
             * snapshot.  Resume that exact batch before release/ready scanning
             * can admit a zero-work successor and merge it into HEAP_COMPL
             * ahead of an unpublished predecessor. */
            if (c->resume_compl) {
                c->resume_compl = 0;
                goto drain_compl_snapshot;
            }
            while ((idx = heap_pop(c, HEAP_RELEASE)) >= 0) {
                msim_task *t = &c->task[idx];
                if (t->state != 0) { t->ready = 0; continue; }
                if (t->waiting != 0) { t->ready = 0; continue; }
                if (t->release > now) { heap_push(c, HEAP_RELEASE, idx); break; }
                t->ready = 1;
                if (!heap_push(c, HEAP_READY, idx)) { rc = MSIM_E_RANGE; goto fail; }
            }
            while ((idx = heap_pop(c, HEAP_READY)) >= 0) {
                msim_task *t = &c->task[idx];
                if (t->state != 0) { t->ready = 0; continue; }
                if (t->waiting != 0) { t->ready = 0; continue; }
                if (t->release > now) {
                    if (!heap_push(c, HEAP_RELEASE, idx))
                        { rc = MSIM_E_RANGE; goto fail; }
                    continue;
                }
                if (!msim_fits((size_t)nstart, (size_t)1, start_cap)) {
                    /* Output-buffer continuation rather than a step-budget stop.
                     * Preserve the pending record: the task is not activated,
                     * so the next call emits its start event exactly once. */
                    if (!heap_push(c, HEAP_READY, idx))
                        { rc = MSIM_E_RANGE; goto fail; }
                    boundary = MSIM_B_OUTPUT;
                    goto done;
                }
                if (!msim_grow((void **)&c->act, &c->act_cap, c->act_n + 1, sizeof(int32_t))) {
                    if (!heap_push(c, HEAP_READY, idx))
                        { rc = MSIM_E_RANGE; goto fail; }
                    { rc = MSIM_E_RANGE; goto fail; }
                }
                t->state = 1;
                t->start = now;
                t->act_slot = c->act_n;
                c->act[c->act_n++] = idx;
                started[(size_t)nstart * 2 + 0] = t->seq;
                started[(size_t)nstart * 2 + 1] = msim_to_bits(now);
                nstart++;
            }
            for (i = 0; i < c->act_n; i++) {
                int32_t ai = c->act[i];
                msim_task *t = &c->task[ai];
                if (t->compl || !task_completable(c, t, now)) continue;
                t->compl = 1;
                if (!heap_push(c, HEAP_COMPL, ai)) { rc = MSIM_E_RANGE; goto fail; }
            }
            if (c->cheap_n == 0) break;
drain_compl_snapshot:
            while (c->cheap_n > 0) {
                int32_t ci = heap_pop(c, HEAP_COMPL);
                msim_task *t = &c->task[ci];
                int32_t k, emit;
                if (t->compl == 0) continue;
                t->compl = 0;
                if (t->state != 1 || !task_completable(c, t, now)) continue;
                if (!emit_completion(c, ci, now, comp, comp_cap, svc, svc_cap,
                                     &ncomp_out, &nsvc_out)) {
                    /* Completion or per-link service buffer full.
                     * The completion was not published; re-mark and restore its
                     * heap cell so a budget-0 drain emits the rest of this
                     * snapshot in sequence order before any awakened successor
                     * can start. */
                    t->compl = 1;
                    if (!heap_push(c, HEAP_COMPL, ci)) { rc = MSIM_E_RANGE; goto fail; }
                    c->resume_compl = 1;
                    /* Keep the admission hold latched across the wire split while
                     * admission prohibition must survive the wire split.  The
                     * continuation call starts with ``hold`` already set. */
                    c->pending_hold = hold;
                    boundary = MSIM_B_OUTPUT;
                    goto done;
                }
                emit = (t->flags & (MSIM_F_CALLBACK | MSIM_F_STOP)) != 0
                       || t->ext_waiters != 0;
                /* A callback-capable completion halts further admission at this
                 * instant.  The awakened successors stay ready in the C heaps
                 * but are not started until Python has published this
                 * completion and released its ``_Task`` reference, so a
                 * finalizer/callback observes exactly the reference order. */
                if ((t->flags & MSIM_F_CALLBACK) != 0) hold = 1;
                remove_active(c, t);
                t->state = 2;
                if (!msim_grow((void **)&c->free, &c->free_cap, c->free_n + 1,
                               sizeof(int32_t)))
                    { rc = MSIM_E_RANGE; goto fail; }
                c->free[c->free_n++] = ci;
                if (c->live_n > 0) c->live_n--;
                for (k = 0; k < (int32_t)t->nlink; k++) {
                    msim_link *l = &c->link[t->lstart + k];
                    l->rate = 0.0;
                    l->rem = 0.0;
                }
                for (k = t->first_succ; k >= 0; k = c->snext[k]) {
                    int32_t succ = c->sidx[k];
                    msim_task *s2 = &c->task[succ];
                    if (s2->state != 0 || s2->seq != c->sseq[k]) continue;
                    if (s2->waiting > 0) s2->waiting--;
                    if (s2->waiting == 0 && !s2->ready) {
                        s2->ready = 1;
                        if (!heap_push(c, HEAP_READY, succ))
                            { rc = MSIM_E_RANGE; goto fail; }
                    }
                }
                t->first_succ = -1;
                if (emit)
                    boundary = (t->flags & MSIM_F_STOP) ? MSIM_B_STOP
                                                        : MSIM_B_CALLBACK;
            }
            /* Hand back before the next admission round when this
             * instant published a callback-capable completion.  Python
             * publishes the whole completion snapshot and drops those task
             * references first, then performs same-instant ready admission.
             * A non-callback boundary settles the full instant: every
             * start/completion woken at ``now`` is published in this call. */
            if (hold) break;
        }
        /* The interrupted completion snapshot has now fully drained.
         * The next call may admit the rows awakened by it; this call still
         * replies with the hold set so Python publishes every record first. */
        if (hold) {
            c->pending_hold = 0;
            if (boundary == MSIM_B_NONE) boundary = MSIM_B_CALLBACK;
            break;
        }
        if (boundary != MSIM_B_NONE) break;
        /* Inner rounds return early only on output continuations;
         * reaching here means ``now`` is fully settled.  Decide the public
         * boundary first, so a settled horizon/idle state is never reported as
         * a step-budget stop, then reserve ``MSIM_B_BUDGET`` for the case where
         * one more positive-dt interval really is required. */
        if (now >= horizon) { boundary = MSIM_B_HORIZON; break; }
        if (c->act_n == 0) {
            /* Nothing is running yet, but a future release may still promote a
             * task: advance the clock exactly as the reference does.  An empty
             * context is idle immediately; the Python Engine stops at the last
             * completion rather than reporting the caller's horizon. */
            double rel = next_rel;
            int32_t j;
            if (c->live_n == 0) { boundary = MSIM_B_IDLE; break; }
            for (j = 0; j < c->rheap_n; j++) {
                msim_task *t = &c->task[c->rheap[j]];
                if (t->state != 0 || t->waiting != 0) continue;
                if (t->release < rel) rel = t->release;
            }
            /* ``INFINITY > INFINITY`` is false: an exhausted graph must be idle.
             * A release beyond a finite horizon still advances to that horizon. */
            if (!(rel > now) || (!isfinite(rel) && !isfinite(horizon))) {
                boundary = MSIM_B_IDLE; break;
            }
        }
        if (steps >= (int32_t)budget) { boundary = MSIM_B_BUDGET; break; }
        rc = compute_rates(c);
        if (rc != 0) goto fail;
        {
            double dt = horizon - now, rel = next_rel, next_time;
            int32_t a, k, n_act_res = 0, bad_li = -1, bad_res = 0;
            for (i = 0; i < c->rheap_n; i++) {
                msim_task *t = &c->task[c->rheap[i]];
                if (t->waiting == 0 && t->state == 0 && t->release < rel) rel = t->release;
            }
            if (rel - now < dt) dt = rel - now;
            for (a = 0; a < c->act_n; a++) {
                msim_task *t = &c->task[c->act[a]];
                if (t->state != 1) continue;
                for (k = 0; k < (int32_t)t->nlink; k++) {
                    msim_link *l = &c->link[t->lstart + k];
                    if (l->rate > 0.0 && l->rem > 0.0) {
                        double b = l->rem / l->rate;
                        if (b < dt) dt = b;
                    }
                }
                if ((t->flags & MSIM_F_LATENCY) && t->start + t->latency > now) {
                    double b = t->start + t->latency - now;
                    if (b < dt) dt = b;
                }
            }
            /* Python's representability rule is checked per consumer, never as a
             * per-resource aggregate.  Pick the lowest link index on tie so
             * ``fault_res`` matches the ascending link scan bit-for-bit. */
            for (a = 0; a < c->act_n; a++) {
                msim_task *t = &c->task[c->act[a]];
                if (t->state != 1) continue;
                for (k = 0; k < (int32_t)t->nlink; k++) {
                    int32_t li = (int32_t)t->lstart + k;
                    msim_link *l = &c->link[li];
                    int32_t mode;
                    if (l->rem <= 0.0 || l->rate > 0.0) continue;
                    mode = c->res[l->res].mode;
                    if (mode == 1) continue;
                    if (mode == 4 && l->mode != 1) continue;
                    if (bad_li < 0 || li < bad_li) {
                        bad_li = li;
                        bad_res = l->res + 1;
                    }
                }
            }
            if (bad_li >= 0) {
                fault_res = bad_res;
                rc = MSIM_E_ARITH;
                goto fail;
            }
            if (!isfinite(dt)) { rc = MSIM_E_STALL; goto fail; }
            next_time = now + dt;
            if (dt > 0.0 && next_time == now) next_time = nextafter(now, INFINITY);
            if (!(next_time > now)) { rc = MSIM_E_STALL; goto fail; }
            if (c->obs_buf != NULL && dt > 0.0) {
                uint32_t need_obs = 0;
                for (a = 0; a < c->act_n; a++) {
                    msim_task *t = &c->task[c->act[a]];
                    if (t->state != 1) continue;
                    for (k = 0; k < (int32_t)t->nlink; k++) {
                        msim_link *l = &c->link[t->lstart + k];
                        if (l->rate > 0.0 && l->rem > 0.0) need_obs++;
                    }
                }
                if (c->obs_count + need_obs > c->obs_cap) {
                    if (need_obs > c->obs_cap) c->obs_overflow = need_obs;
                    boundary = MSIM_B_OUTPUT; goto done;
                }
            }
            memset(c->res_seen, 0, (size_t)c->nres);
            for (a = 0; a < c->act_n; a++) {
                msim_task *t = &c->task[c->act[a]];
                if (t->state != 1) continue;
                for (k = 0; k < (int32_t)t->nlink; k++) {
                    msim_link *l = &c->link[t->lstart + k];
                    if (is_consumer(c, l) && !c->res_seen[l->res]) {
                        c->res_seen[l->res] = 1;
                        c->busy[l->res] += dt;
                        c->act_res[n_act_res++] = l->res;
                    }
                }
            }
            for (i = 1; i < n_act_res; i++) {
                int32_t key = c->act_res[i], j = i - 1;
                while (j >= 0 && c->act_res[j] > key) {
                    c->act_res[j + 1] = c->act_res[j];
                    j--;
                }
                c->act_res[j + 1] = key;
            }
            for (i = 0; i < n_act_res; i++) {
                int32_t r = c->act_res[i], n = 0, k2;
                rc = build_order(c, r, &n);
                if (rc != 0) goto fail;
                for (k2 = 0; k2 < n; k2++) {
                    msim_link *l = &c->link[c->idx[k2]];
                    double remaining, scaled, used, residual;
                    if (l->rate <= 0.0 || l->rem <= 0.0) continue;
                    remaining = l->rem;
                    scaled = dt * l->rate;
                    if (scaled >= remaining) { used = remaining; residual = 0.0; }
                    else {
                        residual = remaining - scaled;
                        if (residual <= 4.0 * msim_ulp(remaining)) {
                            used = remaining; residual = 0.0;
                        } else used = scaled;
                    }
                    l->svc += used;
                    c->serv[l->res] += used;
                    l->rem = residual;
                    if (c->obs_buf != NULL && dt > 0.0) {
                        uint32_t owner_u = l->plen > 0 ? c->units[l->pstart] : 0u;
                        msim_emit_obs_service(
                            c, now, next_time, c->task[l->task].seq,
                            (uint32_t)l->res, owner_u, used);
                    }
                }
            }
            now = next_time;
            io[2] = dt;
            steps++;
        }
    }
done:
    io[0] = now;
    st[2] = boundary;
    st[3] = steps;
    st[4] = ncomp_out;
    st[5] = nsvc_out;
    st[6] = nstart;
    st[7] = 0;
    /* hdr[3] carries the one-bit "same-instant admission held" reply.  It is
     * set only when a callback-capable completion ended this call before the
     * ready heaps were admitted; Python then publishes the completions and
     * runs a zero-time settlement for the awakened rows. */
    hdr[3] = (uint64_t)hold;
    hdr[15] = (uint64_t)c->live_n;
    return 0;
fail:
    /* Unified publication epilogue for post-validation failures.  The
     * accepted flag is explicit: no clock advance, start, completion or step
     * means no prefix exists, so no stale buffer can be mistaken for one. */
    rc = rc != 0 ? rc : MSIM_E_STATE;
    accepted = steps > 0 || nstart > 0 || ncomp_out > 0 || now != initial_now;
    if (accepted) {
        io[0] = now;
        st[2] = boundary;
        st[3] = steps;
        st[4] = ncomp_out;
        st[5] = nsvc_out;
        st[6] = nstart;
        st[7] = fault_res;
        st[1] = rc;
        hdr[3] = (uint64_t)hold;
        hdr[15] = (uint64_t)c->live_n;
        st[0] = MSIM_ST_FAILED | MSIM_ST_PREFIX;
        return -1;
    }
    /* No prefix exists, but the failing resource is still fault detail. */
    st[7] = fault_res;
    return reject(st, nst, rc);
}
#endif /* MSIM_ENGINE_ACCEL_ABI_H */
