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
        if (fifo_link_less(c, cand, best)) best = cand;
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
    int32_t nf = 0, nc = 0, nslot = 0, nent = 0, owner_n = 0, ti, k, r, i;
    int32_t total_u = 0, nact = 0, n_touched = 0, tr;
    int32_t *ohead = c->ohead, *onext = c->onext, *oslot = c->oslot;
    msim_task *T = c->task;
    msim_link *L = c->link;
    msim_coef *coef = c->coef;
    msim_flow *flow = c->flow;
    msim_entry *entry = c->entry;
    msim_slot *slot = c->slot;
    msim_row *row = c->row;

    msim_next_epoch(&c->fifo_epoch, c->res_fifo_ep, c->nres);
    /* Sort active coupled task slots by ascending ``seq`` to match Python
     * ``sorted(active)`` binary64 summation order bit-for-bit. */
    for (i = 0; i < c->act_n; i++) {
        int32_t cur = c->act[i], pos;
        if (T[cur].state != 1 || !T[cur].coupled_comp) continue;
        pos = nact;
        while (pos > 0 && T[c->idx[pos - 1]].seq > T[cur].seq) {
            c->idx[pos] = c->idx[pos - 1];
            pos--;
        }
        c->idx[pos] = cur;
        nact++;
    }
    for (i = 0; i < nact; i++) {
        msim_task *t = &T[c->idx[i]];
        ti = c->idx[i];
        if (t->nlink <= 0) continue;
        for (k = 0; k < (int32_t)t->nlink; k++) {
            int32_t li = (int32_t)t->lstart + k, rr;
            if (L[li].rem <= 0.0 || !L[li].coupled_comp) continue;
            rr = L[li].res;
            if (c->res[rr].policy == 1) {
                int32_t prev = (c->res_fifo_ep[rr] == c->fifo_epoch)
                    ? c->fifo_first[rr] : -1;
                if (prev < 0 || proxy_less(c, ti, prev)) {
                    c->fifo_first[rr] = ti;
                    c->res_fifo_ep[rr] = c->fifo_epoch;
                }
            }
        }
    }
    msim_next_epoch(&c->coef_epoch, c->res_coef_ep, c->nres);
    for (i = 0; i < nact; i++) {
        msim_task *t;
        int32_t na = 0, groups, gi;
        ti = c->idx[i];
        t = &T[ti];
        if (t->nlink > 0 && !msim_grow((void **)&c->act_links, &c->act_links_cap,
                                       (int32_t)t->nlink, sizeof(int32_t)))
            return MSIM_E_ARITH;
        for (k = 0; k < (int32_t)t->nlink; k++) {
            int32_t li = (int32_t)t->lstart + k;
            if (L[li].rem > 0.0 && L[li].coupled_comp) c->act_links[na++] = li;
        }
        if (na == 0) continue;
        groups = (t->flags & MSIM_F_COUPLED) ? 1 : na;
        for (gi = 0; gi < groups; gi++) {
            int32_t cnt = groups == 1 ? na : 1;
            double scale;
            msim_flow *f;
            uint64_t cohort = (t->flags & MSIM_F_COUPLED) ? t->cohort : 1u;
            if (nf >= MSIM_MAX_F) return MSIM_E_ARITH;
            if (!msim_fits((size_t)nc, (size_t)cnt, (size_t)MSIM_MAX_COEF))
                return MSIM_E_ARITH;
            f = &flow[nf];
            f->task = ti; f->coeff_start = nc; f->coeff_count = cnt;
            f->limit = INFINITY; f->assigned = 0.0; ohead[nf] = -1;
            if (cnt == 1) {
                scale = coupled_scale(c, &L[c->act_links[gi]], cohort);
            } else {
                scale = -INFINITY;
                for (k = 0; k < cnt; k++) {
                    double sm = coupled_scale(c, &L[c->act_links[k]], cohort);
                    if (sm > scale) scale = sm;
                }
            }
            if (!isfinite(scale) || scale <= 0.0) return MSIM_E_ARITH;
            for (k = 0; k < cnt; k++) {
                int32_t li = groups == 1 ? c->act_links[k] : c->act_links[gi];
                int32_t rr = L[li].res;
                int32_t ci = nc;
                double amount = L[li].rem / scale;
                int32_t winner;
                coef[ci].res = rr; coef[ci].link = li; coef[ci].amount = amount;
                c->coef_flow[ci] = nf;
                c->coef_next[ci] = -1;
                if (c->res_coef_ep[rr] != c->coef_epoch) {
                    c->res_coef_ep[rr] = c->coef_epoch;
                    c->touched_res[n_touched++] = rr;
                    c->res_chead[rr] = ci;
                    c->res_ctail[rr] = ci;
                } else {
                    c->coef_next[c->res_ctail[rr]] = ci;
                    c->res_ctail[rr] = ci;
                }
                nc++;
                if (isfinite(L[li].cap)) {
                    double lim = L[li].cap / amount;
                    if (lim < f->limit) f->limit = lim;
                }
                winner = (c->res_fifo_ep[rr] == c->fifo_epoch) ? c->fifo_first[rr] : -1;
                if (winner >= 0 && winner != ti) f->limit = 0.0;
            }
            nf++;
        }
    }
    msim_heapsort_i32(c->touched_res, n_touched);
    for (tr = 0; tr < n_touched; tr++) {
        int32_t rows = 0, common = 1, len0, ci;
        double capacity;
        r = c->touched_res[tr];
        capacity = c->res[r].rate / (double)c->res[r].units;
        for (ci = c->res_chead[r]; ci >= 0; ci = c->coef_next[ci]) {
            msim_coef *cf = &coef[ci];
            if (rows >= MSIM_MAX_MROW) return MSIM_E_ARITH;
            row[rows].flow = c->coef_flow[ci];
            row[rows].pstart = L[cf->link].pstart;
            row[rows].plen = L[cf->link].plen;
            row[rows].amount = cf->amount / (double)unit_len(c, &L[cf->link]);
            rows++;
        }
        if (rows == 0) continue;
        len0 = row[0].plen > 0 ? row[0].plen : c->res[r].units;
        for (i = 1; i < rows; i++) {
            int32_t li = row[i].plen > 0 ? row[i].plen : c->res[r].units, k2;
            if (li != len0) { common = 0; break; }
            for (k2 = 0; k2 < li; k2++) {
                int32_t av = row[i].plen > 0
                    ? (int32_t)c->units[row[i].pstart + k2] : k2;
                int32_t bv = row[0].plen > 0
                    ? (int32_t)c->units[row[0].pstart + k2] : k2;
                if (av != bv) { common = 0; break; }
            }
            if (!common) break;
        }
        if (common) {
            int32_t e;
            if (rows == 1) {
                if (row[0].amount != 0.0) {
                    double lim = capacity / row[0].amount;
                    if (lim < flow[row[0].flow].limit) flow[row[0].flow].limit = lim;
                }
                continue;
            }
            if (nslot >= MSIM_MAX_SLOT) return MSIM_E_ARITH;
            if (!msim_own_reserve(c, nent + rows)) return MSIM_E_ARITH;
            if (!msim_slot_reserve(c, nslot + 1)) return MSIM_E_ARITH;
            entry = c->entry; onext = c->onext; oslot = c->oslot;
            slot = c->slot;
            slot[nslot].cap = capacity; slot[nslot].start = nent; slot[nslot].count = rows;
            for (e = 0; e < rows; e++) {
                if (owner_n >= c->own_cap) return MSIM_E_ARITH;
                entry[nent].flow = row[e].flow;
                entry[nent].amount = row[e].amount;
                oslot[owner_n] = nslot;
                onext[owner_n] = ohead[row[e].flow];
                ohead[row[e].flow] = owner_n++;
                nent++;
            }
            nslot++;
            continue;
        }
        {
            int32_t distinct = 0;
            msim_ubits seen;
            msim_ubits_clear(&seen);
            total_u = 0;
            for (i = 0; i < rows; i++) {
                int32_t len = row[i].plen > 0 ? row[i].plen : c->res[r].units, k2;
                total_u += len;
                for (k2 = 0; k2 < len; k2++) {
                    int32_t u = row[i].plen > 0
                        ? (int32_t)c->units[row[i].pstart + k2] : k2;
                    if (u < 0 || u >= MSIM_MAX_U) return MSIM_E_ARITH;
                    if (!msim_ubits_test(&seen, u)) { msim_ubits_set(&seen, u); distinct++; }
                }
            }
            if (distinct == total_u) {
                for (i = 0; i < rows; i++) if (row[i].amount != 0.0) {
                    double lim = capacity / row[i].amount;
                    if (lim < flow[row[i].flow].limit) flow[row[i].flow].limit = lim;
                }
                continue;
            }
        }
        {
            int32_t nu = c->res[r].units, order[MSIM_MAX_U], norder = 0, ue = 0;
            if (!msim_uent_reserve(c, total_u)) return MSIM_E_ARITH;
            if (!msim_own_reserve(c, nent + total_u)) return MSIM_E_ARITH;
            entry = c->entry; onext = c->onext; oslot = c->oslot;
            for (i = 0; i < nu; i++) c->uhead[i] = -1;
            for (i = 0; i < rows; i++) {
                int32_t len = row[i].plen > 0 ? row[i].plen : nu, k2;
                for (k2 = 0; k2 < len; k2++) {
                    int32_t u = row[i].plen > 0
                        ? (int32_t)c->units[row[i].pstart + k2] : k2;
                    int32_t prev;
                    if (u < 0 || u >= MSIM_MAX_U || ue >= c->uent_cap)
                        return MSIM_E_ARITH;
                    prev = c->uhead[u];
                    c->uflow[ue] = row[i].flow;
                    c->uamt[ue] = row[i].amount;
                    c->unext[ue] = -1;
                    if (prev < 0) { c->uhead[u] = ue; order[norder++] = u; }
                    else {
                        while (c->unext[prev] >= 0) prev = c->unext[prev];
                        c->unext[prev] = ue;
                    }
                    ue++;
                }
            }
            for (i = 0; i < norder; i++) {
                int32_t e = c->uhead[order[i]], count = 0, k2;
                for (k2 = e; k2 >= 0; k2 = c->unext[k2]) count++;
                if (count == 1) {
                    if (c->uamt[e] != 0.0) {
                        double lim = capacity / c->uamt[e];
                        msim_flow *f = &flow[c->uflow[e]];
                        if (lim < f->limit) f->limit = lim;
                    }
                    continue;
                }
                if (nslot >= MSIM_MAX_SLOT) return MSIM_E_ARITH;
                if (!msim_own_reserve(c, nent + count)) return MSIM_E_ARITH;
                if (!msim_slot_reserve(c, nslot + 1)) return MSIM_E_ARITH;
                entry = c->entry; onext = c->onext; oslot = c->oslot;
                slot = c->slot;
                slot[nslot].cap = capacity; slot[nslot].start = nent;
                slot[nslot].count = count;
                for (k2 = c->uhead[order[i]]; k2 >= 0; k2 = c->unext[k2]) {
                    if (owner_n >= c->own_cap) return MSIM_E_ARITH;
                    entry[nent].flow = c->uflow[k2];
                    entry[nent].amount = c->uamt[k2];
                    oslot[owner_n] = nslot;
                    onext[owner_n] = ohead[c->uflow[k2]];
                    ohead[c->uflow[k2]] = owner_n++;
                    nent++;
                }
                nslot++;
            }
        }
    }
    if (nslot == 0) {
        for (i = 0; i < nf; i++) {
            msim_flow *f = &flow[i];
            if (f->limit > 0.0 && isfinite(f->limit))
                for (k = 0; k < f->coeff_count; k++) {
                    msim_coef *cf = &coef[f->coeff_start + k];
                    L[cf->link].rate = cf->amount * f->limit;
                    L[cf->link].mode = 1;
                }
        }
        return 0;
    }
    {
        bool *active = c->active, *changed;
        int32_t *pending = c->pend, *survivors = c->surv;
        double *rem_slot, *slope, *bounds;
        int32_t np = 0, ns = 0, guard = 0, o, s;
        if (!msim_slot_reserve(c, nslot)) return MSIM_E_ARITH;
        slot = c->slot;
        changed = c->changed;
        rem_slot = c->slot_rem; slope = c->slot_slope; bounds = c->slot_bounds;
        for (i = 0; i < nslot; i++) changed[i] = false;
        for (i = 0; i < nf; i++) {
            flow[i].assigned = 0.0;
            active[i] = flow[i].limit > 0.0;
            if (active[i]) pending[np++] = i;
        }
        for (i = 0; i < nslot; i++) {
            int32_t e2;
            double s0 = 0.0, c0 = 0.0;
            rem_slot[i] = slot[i].cap;
            for (e2 = slot[i].start; e2 < slot[i].start + slot[i].count; e2++)
                if (active[entry[e2].flow]) {
                    double x = entry[e2].amount, t = s0 + x;
                    c0 += (fabs(s0) >= fabs(x)) ? ((s0 - t) + x) : ((x - t) + s0);
                    s0 = t;
                }
            slope[i] = s0 + c0;
        }
        while (np > 0 && guard++ <= nf + 1) {
            double delta = INFINITY;
            ns = 0;
            for (i = 0; i < nslot; i++) {
                bounds[i] = slope[i] != 0.0 ? rem_slot[i] / slope[i] : INFINITY;
                if (bounds[i] < delta) delta = bounds[i];
            }
            for (i = 0; i < np; i++) {
                double b = flow[pending[i]].limit - flow[pending[i]].assigned;
                if (b < delta) delta = b;
            }
            if (!isfinite(delta)) return MSIM_E_ARITH;
            for (i = 0; i < nslot; i++) {
                if (slope[i] != 0.0) {
                    double v = rem_slot[i] - delta * slope[i];
                    rem_slot[i] = v > 0.0 ? v : 0.0;
                }
                if (bounds[i] <= delta)
                    for (o = slot[i].start; o < slot[i].start + slot[i].count; o++)
                        active[entry[o].flow] = false;
            }
            for (i = 0; i < np; i++) {
                int32_t fi = pending[i];
                double nv = flow[fi].assigned + delta;
                flow[fi].assigned = nv;
                if (active[fi] && nv < flow[fi].limit) survivors[ns++] = fi;
                else {
                    active[fi] = false;
                    for (o = ohead[fi]; o >= 0; o = onext[o]) changed[oslot[o]] = true;
                }
            }
            for (i = 0; i < ns; i++) pending[i] = survivors[i];
            np = ns;
            if (np == 0) break;
            for (s = 0; s < nslot; s++) if (changed[s]) {
                double s0 = 0.0, c0 = 0.0;
                for (i = slot[s].start; i < slot[s].start + slot[s].count; i++)
                    if (active[entry[i].flow]) {
                        double x = entry[i].amount, t = s0 + x;
                        c0 += (fabs(s0) >= fabs(x)) ? ((s0 - t) + x) : ((x - t) + s0);
                        s0 = t;
                    }
                slope[s] = s0 + c0; changed[s] = false;
            }
        }
        if (np != 0) return MSIM_E_ARITH;
        for (i = 0; i < nf; i++) {
            msim_flow *f = &flow[i];
            if (f->assigned != 0.0)
                for (k = 0; k < f->coeff_count; k++) {
                    msim_coef *cf = &coef[f->coeff_start + k];
                    L[cf->link].rate = cf->amount * f->assigned;
                    L[cf->link].mode = 1;
                }
        }
    }
    return 0;
}

static int32_t compute_rates(msim_ctx *c)
{
    int32_t *idx = c->idx, r, a, k, ri, n_act_res = 0, any_c = 0, any_u = 0;
    if (c->nl > 0 && !msim_grow((void **)&c->link_rnext, &c->link_rnext_cap,
                                c->nl, sizeof(int32_t)))
        return MSIM_E_ARITH;
    for (a = 0; a < c->act_n; a++) {
        msim_task *t = &c->task[c->act[a]];
        int32_t has_rem = 0, is_c = (t->flags & MSIM_F_COUPLED) ? 1 : 0;
        t->coupled_comp = 0;
        if (t->state != 1) continue;
        for (k = 0; k < (int32_t)t->nlink; k++) {
            msim_link *l = &c->link[t->lstart + k];
            l->rate = 0.0;
            l->mode = -1;
            l->coupled_comp = 0;
            if (l->rem > 0.0) {
                has_rem = 1;
                if (is_c) l->coupled_comp = 1;
            }
        }
        if (!has_rem) continue;
        if (is_c) { t->coupled_comp = 1; any_c = 1; }
        else any_u = 1;
    }
    if (any_c && any_u) {
        /* Per-link unit-granularity closure: uncoupled links only join a coupled
         * component when they overlap coupled-hot units (or FIFO) on the SAME
         * resource, never bridging across distinct resources of an uncoupled task. */
        int32_t n_ctouch = 0, ti;
        msim_next_epoch(&c->ord_epoch, c->res_ord_ep, c->nres);
        msim_next_epoch(&c->comp_epoch, c->res_comp_ep, c->nres);
        for (a = 0; a < c->act_n; a++) {
            int32_t slot = c->act[a];
            msim_task *t = &c->task[slot];
            if (t->state != 1) continue;
            for (k = 0; k < (int32_t)t->nlink; k++) {
                int32_t li = (int32_t)t->lstart + k;
                msim_link *l = &c->link[li];
                if (l->rem <= 0.0) continue;
                r = l->res;
                c->link_rnext[li] = -1;
                if (c->res_ord_ep[r] != c->ord_epoch) {
                    c->res_ord_ep[r] = c->ord_epoch;
                    c->res_lhead[r] = li;
                    c->res_ltail[r] = li;
                } else {
                    c->link_rnext[c->res_ltail[r]] = li;
                    c->res_ltail[r] = li;
                }
                if (l->coupled_comp && c->res_comp_ep[r] != c->comp_epoch) {
                    c->res_comp_ep[r] = c->comp_epoch;
                    c->touched_res[n_ctouch++] = r;
                }
            }
        }
        for (ti = 0; ti < n_ctouch; ti++) {
            msim_res *res;
            int32_t count, lj;
            r = c->touched_res[ti];
            res = &c->res[r];
            count = res->units;
            if (count < 1 || count > MSIM_MAX_U) return MSIM_E_ARITH;
            if (res->policy == 1) {
                for (lj = c->res_lhead[r]; lj >= 0; lj = c->link_rnext[lj]) {
                    c->link[lj].coupled_comp = 1;
                    c->task[c->link[lj].task].coupled_comp = 1;
                }
                continue;
            }
            msim_ubits_clear(&c->res_hot_units[r]);
            for (lj = c->res_lhead[r]; lj >= 0; lj = c->link_rnext[lj]) {
                msim_link *l = &c->link[lj];
                int32_t u, len;
                if (!l->coupled_comp) continue;
                len = unit_len(c, l);
                for (u = 0; u < len; u++) {
                    int32_t uid = unit_at(c, l, u);
                    if (uid < 0 || uid >= count) return MSIM_E_ARITH;
                    msim_ubits_set(&c->res_hot_units[r], uid);
                }
            }
            {
                bool grew = true;
                while (grew) {
                    grew = false;
                    for (lj = c->res_lhead[r]; lj >= 0; lj = c->link_rnext[lj]) {
                        msim_link *l2 = &c->link[lj];
                        int32_t u, len2;
                        bool hit = false;
                        if (l2->coupled_comp) continue;
                        len2 = unit_len(c, l2);
                        for (u = 0; u < len2; u++) {
                            int32_t uid = unit_at(c, l2, u);
                            if (uid < 0 || uid >= count) return MSIM_E_ARITH;
                            if (msim_ubits_test(&c->res_hot_units[r], uid)) {
                                hit = true;
                                break;
                            }
                        }
                        if (hit) {
                            l2->coupled_comp = 1;
                            c->task[l2->task].coupled_comp = 1;
                            for (u = 0; u < len2; u++) {
                                int32_t uid = unit_at(c, l2, u);
                                if (!msim_ubits_test(&c->res_hot_units[r], uid)) {
                                    msim_ubits_set(&c->res_hot_units[r], uid);
                                    grew = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    msim_next_epoch(&c->mode_epoch, c->res_mode_ep, c->nres);
    if (any_c) {
        int32_t rc;
        for (a = 0; a < c->act_n; a++) {
            msim_task *t = &c->task[c->act[a]];
            if (t->state != 1 || !t->coupled_comp) continue;
            for (k = 0; k < (int32_t)t->nlink; k++) {
                msim_link *l = &c->link[t->lstart + k];
                if (l->rem > 0.0 && l->coupled_comp) msim_res_mode_set(c, l->res, 4);
            }
        }
        rc = rates_coupled(c);
        if (rc != 0) return rc;
    }
    if (!any_u) return 0;
    msim_next_epoch(&c->seen_epoch, c->res_seen_ep, c->nres);
    msim_next_epoch(&c->ord_epoch, c->res_ord_ep, c->nres);
    for (a = 0; a < c->act_n; a++) {
        msim_task *t = &c->task[c->act[a]];
        if (t->state != 1 || (t->flags & MSIM_F_COUPLED)) continue;
        for (k = 0; k < (int32_t)t->nlink; k++) {
            int32_t li = (int32_t)t->lstart + k;
            msim_link *l = &c->link[li];
            if (l->rem <= 0.0 || l->coupled_comp) continue;
            r = l->res;
            if (c->res_seen_ep[r] != c->seen_epoch) {
                c->res_seen_ep[r] = c->seen_epoch;
                c->act_res[n_act_res++] = r;
            }
            c->link_rnext[li] = -1;
            if (c->res_ord_ep[r] != c->ord_epoch) {
                c->res_ord_ep[r] = c->ord_epoch;
                c->res_lhead[r] = li;
                c->res_ltail[r] = li;
            } else {
                c->link_rnext[c->res_ltail[r]] = li;
                c->res_ltail[r] = li;
            }
        }
    }
    for (ri = 0; ri < n_act_res; ri++) {
        msim_res *res;
        int32_t n = 0, placed = 0, li;
        r = c->act_res[ri];
        res = &c->res[r];
        for (li = c->res_lhead[r]; li >= 0; li = c->link_rnext[li]) {
            if (n >= MSIM_MAX_L) return MSIM_E_ARITH;
            idx[n++] = li;
            if (c->link[li].plen > 0) placed = 1;
        }
        if (n == 0) continue;
        if (res->policy == 1) {
            rates_fifo(c, idx, n, res->rate);
            msim_res_mode_set(c, r, 1);
            continue;
        }
        if (placed) {
            msim_heapsort_i32(idx, n);
            if (rates_partition(c, idx, n, res)) {
                if (msim_res_mode_get(c, r) != 4) msim_res_mode_set(c, r, 2);
                continue;
            }
            if (!rates_overlap(c, idx, n, res)) return MSIM_E_ARITH;
            if (msim_res_mode_get(c, r) != 4) msim_res_mode_set(c, r, 3);
            continue;
        }
        rates_fair(c, idx, n, res->rate);
        if (msim_res_mode_get(c, r) != 4) msim_res_mode_set(c, r, 0);
    }
    return 0;
}

#endif /* MSIM_ENGINE_ACCEL_RATES_H */
