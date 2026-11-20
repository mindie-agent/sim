/* Incremental pure-DAG submission for the C11 arena kernel (engine-arena/2).
 *
 * [Role] Validate and mirror one batch of task/link rows plus placement units
 *   into the owning context, appending to the existing incremental frontier.
 * [Invariants] Every span, ownership slice, aggregate demand and dependency is
 *   checked before the first table mutation.  Link slices must be disjoint:
 *   a task's ``nlink`` rows are owned exactly once, so ``sum(nlink) <= nlinks``
 *   and no two tasks may cite the same link cell.  Placement slices are bounded
 *   per link and rebased into the growable context pool; ``c->nu`` advances by
 *   exactly the new placement count.  The pool is not a lifetime quota: it
 *   grows geometrically up to the signed 32-bit offset domain, and the Python
 *   bridge reclaims it at a safe empty boundary.
 * [Timeouts & Locks] No threads, locks, I/O or unbounded loops; every loop is
 *   bounded by the caller's row/link/dep counts.
 * [Side-Effect Policy] Writes only the owning context and the caller-provided
 *   slot/status outputs, and only after the complete preflight succeeds.
 */
#ifndef MSIM_ENGINE_ACCEL_SUBMIT_H
#define MSIM_ENGINE_ACCEL_SUBMIT_H

#include "_engine_accel_core.h"

/* Read-only ownership and scalar-bounds preflight for submitted task rows. */
static int32_t submit_rows_check(msim_ctx *c, const uint64_t *rows, size_t nrows,
                                 size_t row_stride, size_t nlinks,
                                 size_t *sum_nlink, int32_t *st, size_t nst)
{
    size_t i, total = 0;
    if (nlinks > 0) memset(c->gid, 0, nlinks * sizeof(int32_t));
    for (i = 0; i < nrows; i++) {
        const uint64_t *row = rows + i * row_stride;
        int64_t waiting = (int64_t)row[4];
        int64_t nlink = (int64_t)row[9], lstart = (int64_t)row[10];
        double start = msim_from_bits(row[6]);
        double latency = msim_from_bits(row[7]);
        double release = msim_from_bits(row[8]);
        if (row[1] > 63u) return reject(st, nst, MSIM_E_RANGE);
        if (row[2] > (uint64_t)INT64_MAX) return reject(st, nst, MSIM_E_RANGE);
        if (row[3] < 1u || row[3] == UINT64_MAX)
            return reject(st, nst, MSIM_E_RANGE);
        if (waiting < 0) return reject(st, nst, MSIM_E_RANGE);
        if (nlink < 0 || lstart < 0) return reject(st, nst, MSIM_E_RANGE);
        if (row[5] != 0u) {
            if (row[3] <= 1u) return reject(st, nst, MSIM_E_RANGE);
            if (nlink < 64 && (row[5] >> (unsigned)nlink) != 0u)
                return reject(st, nst, MSIM_E_RANGE);
        }
        if ((uint64_t)lstart > (uint64_t)nlinks
            || (uint64_t)nlink > (uint64_t)nlinks - (uint64_t)lstart)
            return reject(st, nst, MSIM_E_RANGE);
        if (!isfinite(start) || !isfinite(latency) || latency < 0.0)
            return reject(st, nst, MSIM_E_VALUE);
        if (!isfinite(release)) return reject(st, nst, MSIM_E_VALUE);
        if (row[11] > 1u) return reject(st, nst, MSIM_E_RANGE);
        if ((size_t)nlink > SIZE_MAX - total)
            return reject(st, nst, MSIM_E_RANGE);
        total += (size_t)nlink;
        if (total > nlinks) return reject(st, nst, MSIM_E_RANGE);
        {
            int64_t k;
            for (k = 0; k < nlink; k++) {
                size_t cell = (size_t)lstart + (size_t)k;
                if (c->gid[cell] != 0) return reject(st, nst, MSIM_E_RANGE);
                c->gid[cell] = 1;
            }
        }
    }
    *sum_nlink = total;
    return 0;
}

/* Per-link preflight: owner/resource/remaining/capacity and bounded units. */
static int32_t submit_links_check(msim_ctx *c, const uint64_t *links,
                                  size_t nlinks, size_t link_stride,
                                  const uint32_t *place, size_t nplace,
                                  size_t nrows, int32_t *st, size_t nst)
{
    size_t i;
    for (i = 0; i < nlinks; i++) {
        const uint64_t *row = links + i * link_stride;
        int64_t owner = (int64_t)row[0], res = (int64_t)row[1];
        int64_t ps = (int64_t)row[4], pl = (int64_t)row[5];
        double rem = msim_from_bits(row[2]), cap = msim_from_bits(row[3]);
        if (owner < 0 || (uint64_t)owner >= (uint64_t)nrows)
            return reject(st, nst, MSIM_E_RANGE);
        if (res < 0 || res >= c->nres) return reject(st, nst, MSIM_E_RANGE);
        if (!isfinite(rem) || rem <= 0.0) return reject(st, nst, MSIM_E_VALUE);
        if (isnan(cap) || cap <= 0.0) return reject(st, nst, MSIM_E_VALUE);
        if (ps < 0 || pl < 0 || pl > MSIM_MAX_U)
            return reject(st, nst, MSIM_E_RANGE);
        if ((uint64_t)ps > (uint64_t)nplace
            || (uint64_t)pl > (uint64_t)nplace - (uint64_t)ps)
            return reject(st, nst, MSIM_E_RANGE);
        {
            int64_t k;
            for (k = 0; k < pl; k++)
                if ((int64_t)place[ps + k] >= (int64_t)c->res[res].units)
                    return reject(st, nst, MSIM_E_VALUE);
        }
    }
    return 0;
}

/* Dependencies must cite this batch and an already-live predecessor slot. */
static int32_t submit_deps_check(msim_ctx *c, const int32_t *dep_task,
                                 const int32_t *dep_pred, size_t ndeps,
                                 size_t nrows, int32_t *st, size_t nst)
{
    size_t i;
    for (i = 0; i < ndeps; i++) {
        if (dep_task[i] < 0 || (size_t)dep_task[i] >= nrows)
            return reject(st, nst, MSIM_E_RANGE);
        if (!slot_ok(c, dep_pred[i])) return reject(st, nst, MSIM_E_RANGE);
    }
    return 0;
}

/* Incremental submission: every row/link/placement cell is validated before the
 * first table mutation.  On success ``out_slots`` receives each assigned slot. */
MSIM_EXPORT int32_t msim_ctx_submit(void *ctx, const uint64_t *rows, size_t nrows,
                        size_t row_stride, const uint64_t *links, size_t nlinks,
                        size_t link_stride, const uint32_t *place, size_t nplace,
                        const int32_t *dep_task, const int32_t *dep_pred,
                        size_t ndeps, int32_t *out_slots, size_t nslots,
                        int32_t *st, size_t nst)
{
    msim_ctx *c = (msim_ctx *)ctx;
    size_t i, sum_nlink = 0, need_l, base;
    if (st != NULL && nst >= 2) { st[0] = 0; st[1] = 0; }
    if (!ctx_ok(c)) return reject(st, nst, MSIM_E_STATE);
    if (nrows == 0) {
        if (nlinks != 0 || nplace != 0 || ndeps != 0)
            return reject(st, nst, MSIM_E_RANGE);
        return 0;
    }
    if (out_slots == NULL || nslots < nrows) return reject(st, nst, MSIM_E_RANGE);
    if (ndeps > 0 && (dep_task == NULL || dep_pred == NULL))
        return reject(st, nst, MSIM_E_RANGE);
    if (!msim_fits((size_t)c->sn, ndeps, (size_t)MSIM_MAX_OWN))
        return reject(st, nst, MSIM_E_RANGE);
    if (rows == NULL) return reject(st, nst, MSIM_E_RANGE);
    if (links == NULL && nlinks > 0) return reject(st, nst, MSIM_E_RANGE);
    if (place == NULL && nplace > 0) return reject(st, nst, MSIM_E_RANGE);
    if (row_stride < (size_t)MSIM_TASK_STRIDE) return reject(st, nst, MSIM_E_RANGE);
    if (nlinks > 0 && link_stride < (size_t)MSIM_LINK_STRIDE)
        return reject(st, nst, MSIM_E_RANGE);
    if (row_stride > SIZE_MAX / nrows) return reject(st, nst, MSIM_E_RANGE);
    if (nlinks > 0 && link_stride > SIZE_MAX / nlinks)
        return reject(st, nst, MSIM_E_RANGE);
    if ((size_t)nrows > (size_t)c->free_n) return reject(st, nst, MSIM_E_FULL);
    if (nlinks > (size_t)MSIM_MAX_L) return reject(st, nst, MSIM_E_FULL);
    if (!msim_fits((size_t)c->nu, nplace, (size_t)INT32_MAX))
        return reject(st, nst, MSIM_E_RANGE);
    if (submit_rows_check(c, rows, nrows, row_stride, nlinks, &sum_nlink, st, nst))
        return -1;
    if (submit_links_check(c, links, nlinks, link_stride, place, nplace,
                           nrows, st, nst))
        return -1;
    for (i = 0; i < nrows; i++) {
        const uint64_t *row = rows + i * row_stride;
        uint64_t wmask = row[5];
        if (wmask != 0u) {
            uint64_t cohort = row[3];
            int64_t k, nlink = (int64_t)row[9], lstart = (int64_t)row[10];
            for (k = 0; k < nlink && k < 64; k++) {
                if ((wmask >> (unsigned)k) & 1u) {
                    const uint64_t *lrow = links + (size_t)(lstart + k) * link_stride;
                    int64_t pl = (int64_t)lrow[5];
                    if (pl <= 0 || (uint64_t)pl < cohort || ((uint64_t)pl % cohort) != 0u)
                        return reject(st, nst, MSIM_E_RANGE);
                }
            }
        }
    }
    if (submit_deps_check(c, dep_task, dep_pred, ndeps, nrows, st, nst))
        return -1;
    /* Reserve the worst-case new link cells before mutating any task slot. */
    need_l = (size_t)c->nl + sum_nlink;
    if (!msim_fits((size_t)c->nl, sum_nlink, (size_t)MSIM_MAX_L))
        return reject(st, nst, MSIM_E_FULL);
    if (!msim_grow((void **)&c->link, &c->link_cap, (int32_t)need_l,
                   sizeof(msim_link)))
        return reject(st, nst, MSIM_E_RANGE);
    if (ndeps > 0 && c->sn + (int32_t)ndeps > c->s_cap) {
        int32_t target = c->s_cap > 0 ? c->s_cap : 64;
        int32_t *next_sidx, *next_snext;
        uint64_t *next_sseq;
        while (target < c->sn + (int32_t)ndeps) target *= 2;
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
    if (nplace > 0 && !msim_grow((void **)&c->units, &c->units_cap,
                                 (int32_t)(c->nu + nplace), sizeof(uint32_t)))
        return reject(st, nst, MSIM_E_RANGE);
    base = (size_t)c->nu;
    if (nplace > 0) memcpy(c->units + base, place, nplace * sizeof(uint32_t));
    for (i = 0; i < nrows; i++) {
        const uint64_t *row = rows + i * row_stride;
        int32_t slot = c->free[--c->free_n];
        msim_task *t = &c->task[slot];
        int64_t nlink = (int64_t)row[9];
        int32_t block = t->link_block, block_len = t->link_block_len;
        if (block < 0 || block_len < (int32_t)nlink) {
            block = c->nl;
            block_len = (int32_t)nlink;
            c->nl += (int32_t)nlink;
        }
        memset(t, 0, sizeof *t);
        t->seq = row[0];
        t->flags = (int32_t)(row[1] & 63u);
        t->prio = (int64_t)row[2];
        t->cohort = row[3];
        t->waiting = (int64_t)row[4];
        t->start = msim_from_bits(row[6]);
        t->latency = msim_from_bits(row[7]);
        t->release = msim_from_bits(row[8]);
        t->nlink = nlink;
        t->ext_waiters = (int32_t)row[11];
        t->state = 0;
        t->act_slot = -1;
        t->link_block = block;
        t->link_block_len = block_len;
        t->lstart = block;
        t->ready = 0;
        t->compl = 0;
        t->first_succ = -1;
        c->idx[i] = slot;
        out_slots[i] = slot;
        c->live_n++;
    }
    for (i = 0; i < nrows; i++) {
        const uint64_t *row = rows + i * row_stride;
        int32_t slot = c->idx[i];
        msim_task *t = &c->task[slot];
        int64_t k, lstart = (int64_t)row[10];
        uint64_t wmask = row[5];
        for (k = 0; k < t->nlink; k++) {
            const uint64_t *lrow = links + (size_t)(lstart + k) * link_stride;
            msim_link *l = &c->link[t->lstart + k];
            memset(l, 0, sizeof *l);
            l->task = slot;
            l->res = (int32_t)(int64_t)lrow[1];
            l->rem = msim_from_bits(lrow[2]);
            l->cap = msim_from_bits(lrow[3]);
            l->pstart = (int32_t)(base + (size_t)lrow[4]);
            l->plen = (int32_t)(int64_t)lrow[5];
            l->mode = -1;
            l->widened = (k < 64 && ((wmask >> (unsigned)k) & 1u)) ? 1u : 0u;
        }
    }
    c->nu = (int32_t)(base + nplace);
    for (i = 0; i < ndeps; i++) {
        int32_t succ = c->idx[dep_task[i]];
        c->sidx[c->sn] = succ;
        c->sseq[c->sn] = c->task[succ].seq;
        c->snext[c->sn] = c->task[dep_pred[i]].first_succ;
        c->task[dep_pred[i]].first_succ = c->sn;
        c->sn++;
    }
    return 0;
}

#endif /* MSIM_ENGINE_ACCEL_SUBMIT_H */
