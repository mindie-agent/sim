/* Rate arbitration for the client-compiled C11 arena kernel (engine-arena/2).
 *
 * [Role] Reproduce the canonical Python arbitration exactly: capped weighted
 *   max-min fair share, non-preemptive FIFO ownership, disjoint/overlapping
 *   physical-unit partitions and the coupled joint-progress fixed point.
 * [Invariants] Pure ISO C11, no state outside the passed ``msim_ctx``; the
 *   progressive-filling loops carry explicit structural guards; equal
 *   placement tuples are compared by unit value (never by CSR offset).
 * [Timeouts & Locks] Finite guarded filling only; no I/O, locks or threads.
 * [Side-Effect Policy] Writes only ``ctx->link[].rate``/``mode`` and the
 *   context's bounded solver scratch.
 */
#ifndef MSIM_ENGINE_ACCEL_RATES_H
#define MSIM_ENGINE_ACCEL_RATES_H

#include "_engine_accel_core.h"

/* -- rate arbitration (exact Python arbiter.py order) ------------------- */
static void rates_fair(msim_ctx *c, const int32_t *idx, int32_t n, double capacity)
{
    int32_t i, j, *order = c->pend;
    double remaining = capacity;
    if (n == 1) {
        double cap = c->link[idx[0]].cap;
        c->link[idx[0]].rate = cap < capacity ? cap : capacity;
        return;
    }
    if (n == 2) {
        int32_t a = idx[0], b = idx[1];
        double ca = c->link[a].cap, cb = c->link[b].cap, share, rest;
        uint64_t ka = c->task[c->link[a].task].seq, kb = c->task[c->link[b].task].seq;
        if (cb < ca || (cb == ca && kb < ka)) {
            int32_t t = a; a = b; b = t;
            { double d = ca; ca = cb; cb = d; }
        }
        share = capacity * 0.5;
        c->link[a].rate = ca < share ? ca : share;
        rest = capacity - c->link[a].rate;
        c->link[b].rate = cb < rest ? cb : rest;
        return;
    }
    for (i = 0; i < n; i++) order[i] = idx[i];
    for (i = 1; i < n; i++) {
        int32_t key = order[i];
        double kc = c->link[key].cap;
        uint64_t ks = c->task[c->link[key].task].seq;
        j = i - 1;
        while (j >= 0 && (c->link[order[j]].cap > kc
                          || (c->link[order[j]].cap == kc
                              && c->task[c->link[order[j]].task].seq > ks))) {
            order[j + 1] = order[j]; j--;
        }
        order[j + 1] = key;
    }
    for (i = 0; i < n; i++) {
        double share = remaining / (double)(n - i), cap = c->link[order[i]].cap;
        if (cap >= share) {
            int32_t k;
            for (k = i; k < n; k++) c->link[order[k]].rate = share;
            break;
        }
        c->link[order[i]].rate = cap;
        remaining -= cap;
    }
}

static void rates_fifo(msim_ctx *c, const int32_t *idx, int32_t n, double capacity)
{
    int32_t best = idx[0], i;
    for (i = 1; i < n; i++) {
        int32_t cand = idx[i];
        if (proxy_less(c, c->link[cand].task, c->link[best].task)) best = cand;
    }
    c->link[best].rate = capacity <= c->link[best].cap ? capacity : c->link[best].cap;
}

/* Disjoint owner partitions: equal pools; overlap declines to the full solver. */
static bool rates_partition(msim_ctx *c, const int32_t *idx, int32_t n,
                            const msim_res *res)
{
    int32_t *gid = c->gid, *rep = c->rep, *members = c->pend;
    int32_t gcount = 0, i, g, k, w;
    uint64_t occ[MSIM_UNIT_WORDS];
    for (w = 0; w < (int32_t)MSIM_UNIT_WORDS; w++) occ[w] = 0;
    for (i = 0; i < n; i++) {
        const msim_link *l = &c->link[idx[i]];
        int32_t len = unit_len(c, l), found = -1;
        if (len < 1 || len > MSIM_MAX_U) return false;
        for (g = 0; g < gcount; g++)
            if (same_units(c, &c->link[rep[g]], l)) { found = g; break; }
        if (found >= 0) { gid[i] = found; continue; }
        {
            msim_ubits mask, seen_occ;
            bool overlaps = false;
            msim_ubits_clear(&mask);
            for (k = 0; k < len; k++) {
                int32_t u = unit_at(c, l, k);
                if (u < 0 || u >= res->units) return false;
                msim_ubits_set(&mask, u);
            }
            for (w = 0; w < (int32_t)MSIM_UNIT_WORDS; w++) seen_occ.w[w] = occ[w];
            for (w = 0; w < (int32_t)MSIM_UNIT_WORDS; w++)
                if ((seen_occ.w[w] & mask.w[w]) != 0) overlaps = true;
            if (overlaps) return false;
            for (w = 0; w < (int32_t)MSIM_UNIT_WORDS; w++) occ[w] |= mask.w[w];
        }
        rep[gcount] = idx[i];
        gid[i] = gcount++;
    }
    for (g = 0; g < gcount; g++) {
        int32_t m = 0;
        double capacity = res->rate / (double)res->units
                          * (double)unit_len(c, &c->link[rep[g]]);
        for (i = 0; i < n; i++) if (gid[i] == g) members[m++] = idx[i];
        for (i = 0; i < m; i++) c->link[members[i]].mode = g;
        if (m == 1) {
            c->link[members[0]].rate = msim_min(capacity, c->link[members[0]].cap);
            continue;
        }
        for (i = 1; i < m; i++) {
            int32_t key = members[i], j = i - 1;
            while (j >= 0 && c->link[members[j]].cap > c->link[key].cap) {
                members[j + 1] = members[j]; j--;
            }
            members[j + 1] = key;
        }
        for (i = 0; i < m; i++) {
            double share = capacity / (double)(m - i), cap = c->link[members[i]].cap;
            if (cap >= share) {
                int32_t k;
                for (k = i; k < m; k++) c->link[members[k]].rate = share;
                break;
            }
            c->link[members[i]].rate = cap;
            capacity -= cap;
        }
    }
    return true;
}

/* Overlapping units: the full capped max-min fixed point of ownership.py. */
static bool rates_overlap(msim_ctx *c, const int32_t *idx, int32_t n,
                          const msim_res *res)
{
    int32_t count = res->units, i, k, u, e, used = 0, guard = 0, nactive = n;
    int32_t need = 0;
    double *rem_u = c->remu, *bounds = c->bounds, *slope = c->slope;
    double *result = c->over;
    bool *active = c->active;
    if (count < 1 || count > MSIM_MAX_U || n > MSIM_MAX_L) return false;
    /* Demand-grown membership scratch: 1,025 x 64 overlapping placements are a
     * valid frontier even though they exceed the former fixed 65,536 entries.
     * ``unit_len`` is the effective membership: an absent placement owns every
     * declared unit, exactly as the Python capacity preflight now accounts. */
    for (i = 0; i < n; i++) {
        int32_t len = unit_len(c, &c->link[idx[i]]);
        if (len < 1 || len > MSIM_MAX_U) return false;
        if (need > INT32_MAX - len) return false;
        need += len;
    }
    if (!msim_uent_reserve(c, need)) return false;
    for (u = 0; u < count; u++) { c->uhead[u] = -1; rem_u[u] = res->rate / (double)count; }
    for (i = 0; i < n; i++) {
        const msim_link *l = &c->link[idx[i]];
        int32_t len = unit_len(c, l);
        double w = 1.0 / (double)len;
        if (len < 1 || len > MSIM_MAX_U) return false;
        result[i] = 0.0;
        active[i] = true;
        for (k = 0; k < len; k++) {
            int32_t prev;
            u = unit_at(c, l, k);
            if (u < 0 || u >= count) return false;
            if (used >= c->uent_cap) return false;
            c->uflow[used] = i; c->uamt[used] = w; c->unext[used] = -1;
            prev = c->uhead[u];
            if (prev < 0) c->uhead[u] = used;
            else { while (c->unext[prev] >= 0) prev = c->unext[prev]; c->unext[prev] = used; }
            used++;
        }
    }
    while (nactive > 0 && guard++ <= n + 1) {
        double delta = INFINITY;
        for (u = 0; u < count; u++) {
            slope[u] = 0.0;
            for (e = c->uhead[u]; e >= 0; e = c->unext[e])
                if (active[c->uflow[e]]) slope[u] += c->uamt[e];
            bounds[u] = slope[u] != 0.0 ? rem_u[u] / slope[u] : INFINITY;
            if (bounds[u] < delta) delta = bounds[u];
        }
        for (i = 0; i < n; i++) if (active[i]) {
            double b = c->link[idx[i]].cap - result[i];
            if (b < delta) delta = b;
        }
        if (!isfinite(delta)) return false;
        for (i = 0; i < n; i++) if (active[i]) result[i] += delta;
        for (u = 0; u < count; u++) {
            if (slope[u] != 0.0) {
                double v = rem_u[u] - delta * slope[u];
                rem_u[u] = v > 0.0 ? v : 0.0;
            }
            c->active2[u] = bounds[u] <= delta;
        }
        for (i = 0; i < n; i++) if (active[i]) {
            const msim_link *l = &c->link[idx[i]];
            bool gone = result[i] >= l->cap;
            for (k = 0; k < unit_len(c, l) && !gone; k++)
                if (c->active2[unit_at(c, l, k)]) gone = true;
            if (gone) { active[i] = false; nactive--; }
        }
    }
    if (nactive != 0) return false;
    for (i = 0; i < n; i++) c->link[idx[i]].rate = result[i];
    return true;
}

/* Normalize a coupled span by one cohort member's bottleneck duration:
 * explicitly widened disjoint placements already span ``cohort`` members'
 * units, while unplaced or shared-unit links multiply denominator by
 * ``cohort`` once so demand is never scaled twice. */
static double coupled_scale(const msim_ctx *c, const msim_link *l, uint64_t cohort)
{
    const msim_res *res = &c->res[l->res];
    int32_t len = unit_len(c, l);
    double denom = (res->rate / (double)res->units) * (double)len;
    if (cohort > 1 && !l->widened) denom *= (double)cohort;
    if (denom <= 0.0) return INFINITY;
    return l->rem / denom;
}

/* -- coupled / pinned joint arbitration (coupled.py order) -------------- */
static int32_t rates_coupled(msim_ctx *c)
{
    /* LEGACY_PURGED(S6-W5-22): deleted dense O(nres)/O(nres*nflows) scans and
     * stack int32_t act[MSIM_MAX_R]; implement sparse touched_res + tail-insert
     * res_ctail + amortized O(1) epoch generation-tag reset */
    (void)c;
    return MSIM_E_STATE;
}

static int32_t compute_rates(msim_ctx *c)
{
    /* LEGACY_PURGED(S6-W5-22): deleted dense O(nres) scans & memset(res_seen, 0, nres);
     * implement sparse touched_res + amortized O(1) epoch generation-tag reset */
    (void)c;
    return MSIM_E_STATE;
}

#endif /* MSIM_ENGINE_ACCEL_RATES_H */
