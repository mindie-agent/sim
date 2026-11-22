/* Native C11 single-tier per-unit coupled progressive-filling arbiter.
 *
 * [Role] Execute ``arbiter_fast_coupled.fast_single_tier_coupled`` entirely in
 *   pure ISO C11: flow normalization, private-unit alpha folding, shared-unit
 *   binding deduplication, degree-1 closed-form shortcut, exact Shewchuk
 *   ``fsum`` progressive filling, and independent per-unit overallocation
 *   verification.
 * [Invariants] Pure ISO C11, zero mutable file-scope state (fully reentrant).
 *   Binary64 operation order, Shewchuk adaptive two-sum ``msim_fsum``,
 *   ``nextafter`` tightening, 16-ULP unit saturation, and 8-ULP cap retirement
 *   reproduce ``arbiter_fast_coupled.py`` bit-for-bit.  Every heap allocation
 *   is released on the single cleanup path; invalid dimensions or OOM return
 *   status 3 so Python falls back cleanly without side effects.
 * [Timeouts & Locks] Finite bounded progressive filling (<= 256 tightening
 *   attempts per step, strictly decreasing active set); no locks, threads or I/O.
 * [Side-Effect Policy] Writes only caller-provided ``out_rate`` and
 *   ``out_fault`` arrays; retains no pointers past return.
 */
#ifndef MSIM_ENGINE_ACCEL_COUPLED_H
#define MSIM_ENGINE_ACCEL_COUPLED_H

#include "_engine_accel_core.h"

/* Omit when embedded in the sanitizer harness under renamed symbols. */
#ifndef msim_abi_version

#define MSIM_FC_MIN_RATE 1e-12
#define MSIM_FC_HALF_STEP 0.5
#define MSIM_FC_CAP_ULPS 8.0
#define MSIM_FC_SAT_ULPS 16.0
#define MSIM_FC_NEXTAFTER_STEPS 64
#define MSIM_FC_MAX_ATTEMPTS 256

/* Exact Shewchuk adaptive floating-point sum matching CPython ``math.fsum``. */
static double msim_fsum(const double *vals, int32_t n)
{
    double p_stack[64];
    double *p = p_stack;
    int32_t cap = 64, np = 0, i;
    double hi = 0.0, lo = 0.0;

    if (n <= 0) return 0.0;
    if (n == 1) return vals[0];
    if (n == 2) return vals[0] + vals[1];

    for (i = 0; i < n; i++) {
        double x = vals[i];
        int32_t j = 0, k;
        if (!isfinite(x)) {
            double s = x;
            for (k = i + 1; k < n; k++) s += vals[k];
            if (p != p_stack) free(p);
            return s;
        }
        for (k = 0; k < np; k++) {
            double y = p[k];
            double sum, r;
            if (fabs(x) < fabs(y)) { double t = x; x = y; y = t; }
            sum = x + y;
            r = y - (sum - x);
            if (r != 0.0) p[j++] = r;
            x = sum;
        }
        np = j;
        if (x != 0.0) {
            if (np >= cap) {
                int32_t ncap = cap * 2;
                double *next = (double *)malloc((size_t)ncap * sizeof(double));
                if (next == NULL) {
                    /* Graceful Kahan fallback on extreme OOM. */
                    double s0 = 0.0, c0 = 0.0;
                    for (k = 0; k < n; k++) {
                        double v = vals[k], t = s0 + v;
                        c0 += (fabs(s0) >= fabs(v)) ? ((s0 - t) + v) : ((v - t) + s0);
                        s0 = t;
                    }
                    if (p != p_stack) free(p);
                    return s0 + c0;
                }
                memcpy(next, p, (size_t)np * sizeof(double));
                if (p != p_stack) free(p);
                p = next;
                cap = ncap;
            }
            p[np++] = x;
        }
    }
    if (np == 0) {
        if (p != p_stack) free(p);
        return 0.0;
    }
    hi = p[np - 1];
    np--;
    while (np > 0) {
        double x = hi;
        double y = p[--np];
        hi = x + y;
        lo = y - (hi - x);
        if (lo != 0.0) break;
    }
    if (np > 0 && ((lo < 0.0 && p[np - 1] < 0.0)
                   || (lo > 0.0 && p[np - 1] > 0.0))) {
        double y = lo * 2.0;
        double x = hi + y;
        if (y == x - hi) hi = x;
    }
    if (p != p_stack) free(p);
    return hi;
}

typedef struct {
    int32_t flow;
    double w;
} msim_fc_term;

typedef struct {
    int32_t res;
    int32_t unit;
    double cap;
    int32_t flow;
    double w;
} msim_fc_priv;

typedef struct {
    int32_t res;
    int32_t unit;
    double cap;
    double sat_tol;
    double bind_thresh;
    int32_t start;
    int32_t count;
} msim_fc_unit;

typedef struct {
    double cap;
    double sat_tol;
    int32_t start;
    int32_t count;
} msim_fc_bind;

typedef struct {
    int32_t flow;
    double w;
    int32_t next;
} msim_fc_node;

typedef struct {
    int32_t link_idx;
    double demand;
} msim_fc_span;

static bool msim_fc_push_node(msim_fc_node **arr, int32_t *n, int32_t *cap,
                              int32_t *head, int32_t *tail,
                              int32_t flow, double w)
{
    int32_t idx;
    if (*n >= *cap) {
        int32_t ncap = *cap > 0 ? (*cap * 2) : 64;
        msim_fc_node *next = (msim_fc_node *)realloc(*arr, (size_t)ncap * sizeof(msim_fc_node));
        if (next == NULL) return false;
        *arr = next;
        *cap = ncap;
    }
    idx = (*n)++;
    (*arr)[idx].flow = flow;
    (*arr)[idx].w = w;
    (*arr)[idx].next = -1;
    if (*head < 0) {
        *head = idx;
        *tail = idx;
    } else {
        (*arr)[*tail].next = idx;
        *tail = idx;
    }
    return true;
}

static bool msim_fc_push_term(msim_fc_term **arr, int32_t *n, int32_t *cap,
                              int32_t flow, double w)
{
    if (*n >= *cap) {
        int32_t ncap = *cap > 0 ? (*cap * 2) : 64;
        msim_fc_term *next = (msim_fc_term *)realloc(*arr, (size_t)ncap * sizeof(msim_fc_term));
        if (next == NULL) return false;
        *arr = next;
        *cap = ncap;
    }
    (*arr)[*n].flow = flow;
    (*arr)[*n].w = w;
    (*n)++;
    return true;
}

static void msim_fc_sort_terms(msim_fc_term *t, int32_t n)
{
    int32_t i, j;
    for (i = 1; i < n; i++) {
        msim_fc_term key = t[i];
        j = i - 1;
        while (j >= 0 && (t[j].flow > key.flow
                          || (t[j].flow == key.flow && t[j].w > key.w))) {
            t[j + 1] = t[j];
            j--;
        }
        t[j + 1] = key;
    }
}

static uint64_t msim_fc_hash_unit(double cap, const msim_fc_term *terms, int32_t cnt)
{
    uint64_t h = 14695981039346656037ULL;
    int32_t i;
    h ^= msim_to_bits(cap); h *= 1099511628211ULL;
    h ^= (uint64_t)(uint32_t)cnt; h *= 1099511628211ULL;
    for (i = 0; i < cnt; i++) {
        h ^= (uint64_t)(uint32_t)terms[i].flow; h *= 1099511628211ULL;
        h ^= msim_to_bits(terms[i].w); h *= 1099511628211ULL;
    }
    return h;
}

static bool msim_fc_same_unit(double cap_a, const msim_fc_term *ta, int32_t na,
                              double cap_b, const msim_fc_term *tb, int32_t nb)
{
    int32_t i;
    if (na != nb || cap_a != cap_b) return false;
    for (i = 0; i < na; i++) {
        if (ta[i].flow != tb[i].flow || ta[i].w != tb[i].w) return false;
    }
    return true;
}

static double msim_fc_unit_sum(const msim_fc_term *t, int32_t cnt,
                               const double *mult, double *scratch)
{
    int32_t i;
    if (cnt == 1) return mult[t[0].flow] * t[0].w;
    if (cnt == 2) return mult[t[0].flow] * t[0].w + mult[t[1].flow] * t[1].w;
    for (i = 0; i < cnt; i++) scratch[i] = mult[t[i].flow] * t[i].w;
    return msim_fsum(scratch, cnt);
}

static double msim_fc_delta_sum(const msim_fc_term *t, int32_t cnt,
                                double delta, double *scratch)
{
    int32_t i;
    if (cnt == 1) return delta * t[0].w;
    if (cnt == 2) return delta * t[0].w + delta * t[1].w;
    for (i = 0; i < cnt; i++) scratch[i] = delta * t[i].w;
    return msim_fsum(scratch, cnt);
}

static double msim_fc_sat_tol_exact(double cap, double res_v, double base_tol)
{
    double m;
    if (res_v >= 0.0 && res_v <= cap) return base_tol;
    m = fabs(cap);
    if (fabs(res_v) > m) m = fabs(res_v);
    if (1.0 > m) m = 1.0;
    m = MSIM_FC_SAT_ULPS * msim_ulp(m);
    return m > MSIM_FC_MIN_RATE ? m : MSIM_FC_MIN_RATE;
}

/* Native C11 implementation of ``fast_single_tier_coupled``.
 * ``out_fault`` has 3 int32 cells: [code, res_idx, unit].
 *   code == 0: success
 *   code == 1: coupled_scale_underflow (ZeroDivisionError)
 *   code == 2: unit_service_overallocated at (res_idx, unit) (AssertionError)
 *   code == 3: invalid args / allocation failure (delegate to Python)
 */
MSIM_EXPORT int32_t msim_fast_single_tier_coupled(
    int32_t n_res,
    const double *res_rate,
    const int32_t *res_units,
    int32_t n_tasks,
    const uint8_t *task_coupled,
    const int32_t *task_cohort,
    const int32_t *task_link_off,
    int32_t n_links,
    const int32_t *link_res,
    const double *link_rem,
    const double *link_cap,
    const uint8_t *link_widened,
    const uint8_t *link_unplaced,
    const int32_t *link_place_off,
    const int32_t *place_units,
    double *out_rate,
    int32_t *out_fault)
{
    int32_t rc = 0;
    int32_t r, t, l, i, u;
    int32_t total_res_units = 0;
    double *r_ucap = NULL, *r_sattol = NULL, *r_bthresh = NULL;
    int32_t *r_cnt = NULL, *r_uoff = NULL;
    int32_t *r_fhead = NULL, *r_ftail = NULL, *r_fcnt = NULL;
    int32_t *u_head = NULL, *u_tail = NULL, *u_cnt = NULL, *r_umap_n = NULL;
    msim_fc_node *nodes = NULL;
    int32_t n_nodes = 0, cap_nodes = 0;

    int32_t n_flows = 0, cap_flows = 0;
    double *flow_limits = NULL, *alphas = NULL;
    int32_t *flow_span_off = NULL;
    msim_fc_span *flow_spans = NULL;
    int32_t n_spans_tot = 0, cap_spans = 0;

    msim_fc_priv *privs = NULL;
    int32_t n_priv = 0, cap_priv = 0;
    msim_fc_unit *shared = NULL;
    int32_t n_shared = 0, cap_shared = 0;
    msim_fc_term *terms = NULL;
    int32_t n_terms = 0, cap_terms = 0;

    msim_fc_bind *binding = NULL;
    int32_t n_binding = 0;
    int32_t *htab = NULL;

    int32_t *deg = NULL;
    uint8_t *act_flag = NULL;
    msim_fc_bind *rem_units = NULL;
    int32_t n_rem = 0;

    int32_t *act_idx = NULL, *next_act = NULL;
    int32_t *open_sh = NULL, *still_open = NULL, *next_open = NULL;
    int32_t *comm_off = NULL;
    double *committed = NULL, *totals = NULL, *cand_tot = NULL;
    double *fsum_buf = NULL;
    int32_t fsum_cap = 0;

    int32_t *span_l = NULL, *span_nu = NULL;
    uint8_t *span_full = NULL;

    if (out_fault != NULL) { out_fault[0] = 0; out_fault[1] = 0; out_fault[2] = 0; }
    if (n_links > 0 && out_rate != NULL) {
        memset(out_rate, 0, (size_t)n_links * sizeof(double));
    }
    if (n_tasks <= 0 || n_links <= 0 || n_res <= 0) return 0;
    if (res_rate == NULL || res_units == NULL || task_coupled == NULL
            || task_cohort == NULL || task_link_off == NULL || link_res == NULL
            || link_rem == NULL || link_cap == NULL || link_widened == NULL
            || link_unplaced == NULL || link_place_off == NULL || out_rate == NULL
            || out_fault == NULL) {
        if (out_fault != NULL) out_fault[0] = 3;
        return -1;
    }

    r_ucap = (double *)malloc((size_t)n_res * sizeof(double));
    r_sattol = (double *)malloc((size_t)n_res * sizeof(double));
    r_bthresh = (double *)malloc((size_t)n_res * sizeof(double));
    r_cnt = (int32_t *)malloc((size_t)n_res * sizeof(int32_t));
    r_uoff = (int32_t *)malloc((size_t)n_res * sizeof(int32_t));
    r_fhead = (int32_t *)malloc((size_t)n_res * sizeof(int32_t));
    r_ftail = (int32_t *)malloc((size_t)n_res * sizeof(int32_t));
    r_fcnt = (int32_t *)calloc((size_t)n_res, sizeof(int32_t));
    r_umap_n = (int32_t *)calloc((size_t)n_res, sizeof(int32_t));
    if (!r_ucap || !r_sattol || !r_bthresh || !r_cnt || !r_uoff
            || !r_fhead || !r_ftail || !r_fcnt || !r_umap_n) {
        out_fault[0] = 3; rc = -1; goto cleanup;
    }

    for (r = 0; r < n_res; r++) {
        int32_t cnt = res_units[r] > 1 ? res_units[r] : 1;
        double u_cap = res_rate[r] / (double)cnt;
        double cap_scale = u_cap < 1.0 ? 1.0 : u_cap;
        double u16 = MSIM_FC_SAT_ULPS * msim_ulp(cap_scale);
        double u32 = MSIM_FC_SAT_ULPS * 2.0 * msim_ulp(cap_scale);
        r_cnt[r] = cnt;
        r_ucap[r] = u_cap;
        r_sattol[r] = u16 > MSIM_FC_MIN_RATE ? u16 : MSIM_FC_MIN_RATE;
        r_bthresh[r] = u_cap - (u32 > MSIM_FC_MIN_RATE ? u32 : MSIM_FC_MIN_RATE);
        r_fhead[r] = -1;
        r_ftail[r] = -1;
        r_uoff[r] = total_res_units;
        if (cnt > INT32_MAX - total_res_units) { out_fault[0] = 3; rc = -1; goto cleanup; }
        total_res_units += cnt;
    }

    u_head = (int32_t *)malloc((size_t)(total_res_units > 0 ? total_res_units : 1) * sizeof(int32_t));
    u_tail = (int32_t *)malloc((size_t)(total_res_units > 0 ? total_res_units : 1) * sizeof(int32_t));
    u_cnt = (int32_t *)calloc((size_t)(total_res_units > 0 ? total_res_units : 1), sizeof(int32_t));
    if (!u_head || !u_tail || !u_cnt) { out_fault[0] = 3; rc = -1; goto cleanup; }
    for (i = 0; i < total_res_units; i++) { u_head[i] = -1; u_tail[i] = -1; }

    cap_flows = n_links + 1;
    flow_limits = (double *)malloc((size_t)cap_flows * sizeof(double));
    flow_span_off = (int32_t *)malloc((size_t)(cap_flows + 1) * sizeof(int32_t));
    cap_spans = n_links + 1;
    flow_spans = (msim_fc_span *)malloc((size_t)cap_spans * sizeof(msim_fc_span));
    span_l = (int32_t *)malloc((size_t)n_links * sizeof(int32_t));
    span_nu = (int32_t *)malloc((size_t)n_links * sizeof(int32_t));
    span_full = (uint8_t *)malloc((size_t)n_links * sizeof(uint8_t));
    if (!flow_limits || !flow_span_off || !flow_spans
            || !span_l || !span_nu || !span_full) {
        out_fault[0] = 3; rc = -1; goto cleanup;
    }
    flow_span_off[0] = 0;

    /* Step 1: Build flows and per-resource/per-unit term lists. */
    for (t = 0; t < n_tasks; t++) {
        int32_t l0 = task_link_off[t];
        int32_t l1 = task_link_off[t + 1];
        if (task_coupled[t]) {
            int32_t cohort = task_cohort[t] > 1 ? task_cohort[t] : 1;
            int32_t ns = 0;
            bool blocked = false;
            double scale = 0.0;
            double limit = 1.0;
            int32_t f_idx;
            for (l = l0; l < l1; l++) {
                double amt = link_rem[l];
                int32_t ri, cnt, n_u = 0;
                double u_cap, denom;
                uint8_t is_full = 0;
                if (amt <= 0.0) continue;
                ri = link_res[l];
                if (ri < 0 || ri >= n_res) { out_fault[0] = 3; rc = -1; goto cleanup; }
                u_cap = r_ucap[ri];
                cnt = r_cnt[ri];
                if (u_cap <= MSIM_FC_MIN_RATE) { blocked = true; break; }
                if (link_unplaced[l]) {
                    n_u = cnt;
                    is_full = 1;
                } else {
                    int32_t p0 = link_place_off[l], p1 = link_place_off[l + 1], p;
                    bool seq_ok = true;
                    for (p = p0; p < p1; p++) {
                        int32_t pu = place_units[p];
                        if (pu >= 0 && pu < cnt) {
                            if (pu != n_u) seq_ok = false;
                            n_u++;
                        }
                    }
                    if (n_u == 0) { blocked = true; break; }
                    is_full = (seq_ok && n_u == cnt) ? 1 : 0;
                }
                denom = u_cap * (double)n_u;
                if (cohort > 1 && !link_widened[l]) denom *= (double)cohort;
                if (denom <= 0.0) {
                    scale = INFINITY;
                } else if (!isinf(scale)) {
                    double val = amt / denom;
                    if (val > scale) scale = val;
                }
                span_l[ns] = l;
                span_nu[ns] = n_u;
                span_full[ns] = is_full;
                ns++;
            }
            if (blocked || ns == 0) continue;
            if (scale == 0.0) { out_fault[0] = 1; rc = -3; goto cleanup; }
            if (!isfinite(scale) || scale < 0.0) continue;
            for (i = 0; i < ns; i++) {
                int32_t li = span_l[i];
                double cap = link_cap[li];
                if (!isnan(cap) && isfinite(cap)) {
                    double demand = link_rem[li] / scale;
                    double c_lim = (demand > 0.0) ? ((cap > 0.0 ? cap : 0.0) / demand) : 0.0;
                    if (c_lim < limit) limit = c_lim;
                }
            }
            if (limit <= 0.0) continue;
            f_idx = n_flows++;
            flow_limits[f_idx] = limit;
            for (i = 0; i < ns; i++) {
                int32_t li = span_l[i];
                int32_t ri = link_res[li];
                int32_t cnt = r_cnt[ri];
                double demand = link_rem[li] / scale;
                double weight = demand / (double)span_nu[i];
                flow_spans[n_spans_tot].link_idx = li;
                flow_spans[n_spans_tot].demand = demand;
                n_spans_tot++;
                if (span_full[i]) {
                    if (!msim_fc_push_node(&nodes, &n_nodes, &cap_nodes,
                                           &r_fhead[ri], &r_ftail[ri], f_idx, weight)) {
                        out_fault[0] = 3; rc = -1; goto cleanup;
                    }
                    r_fcnt[ri]++;
                } else {
                    int32_t p0 = link_place_off[li], p1 = link_place_off[li + 1], p;
                    int32_t base = r_uoff[ri];
                    for (p = p0; p < p1; p++) {
                        int32_t pu = place_units[p];
                        if (pu < 0 || pu >= cnt) continue;
                        if (u_cnt[base + pu] == 0) r_umap_n[ri]++;
                        if (!msim_fc_push_node(&nodes, &n_nodes, &cap_nodes,
                                               &u_head[base + pu], &u_tail[base + pu],
                                               f_idx, weight)) {
                            out_fault[0] = 3; rc = -1; goto cleanup;
                        }
                        u_cnt[base + pu]++;
                    }
                }
            }
            flow_span_off[n_flows] = n_spans_tot;
        } else {
            for (l = l0; l < l1; l++) {
                double amt = link_rem[l], u_cap, demand, cap, limit;
                int32_t ri, cnt, n_u = 0, f_idx;
                uint8_t is_full = 0;
                if (amt <= 0.0) continue;
                ri = link_res[l];
                if (ri < 0 || ri >= n_res) { out_fault[0] = 3; rc = -1; goto cleanup; }
                u_cap = r_ucap[ri];
                cnt = r_cnt[ri];
                if (u_cap <= MSIM_FC_MIN_RATE) continue;
                if (link_unplaced[l]) {
                    n_u = cnt;
                    is_full = 1;
                } else {
                    int32_t p0 = link_place_off[l], p1 = link_place_off[l + 1], p;
                    bool seq_ok = true;
                    for (p = p0; p < p1; p++) {
                        int32_t pu = place_units[p];
                        if (pu >= 0 && pu < cnt) {
                            if (pu != n_u) seq_ok = false;
                            n_u++;
                        }
                    }
                    if (n_u == 0) continue;
                    is_full = (seq_ok && n_u == cnt) ? 1 : 0;
                }
                demand = u_cap * (double)n_u;
                if (demand <= 0.0) continue;
                limit = 1.0;
                cap = link_cap[l];
                if (!isnan(cap) && isfinite(cap)) {
                    double c_lim = (cap > 0.0 ? cap : 0.0) / demand;
                    if (c_lim < limit) limit = c_lim;
                }
                if (limit <= 0.0) continue;
                f_idx = n_flows++;
                flow_limits[f_idx] = limit;
                flow_spans[n_spans_tot].link_idx = l;
                flow_spans[n_spans_tot].demand = demand;
                n_spans_tot++;
                flow_span_off[n_flows] = n_spans_tot;
                {
                    double weight = demand / (double)n_u;
                    if (is_full) {
                        if (!msim_fc_push_node(&nodes, &n_nodes, &cap_nodes,
                                               &r_fhead[ri], &r_ftail[ri], f_idx, weight)) {
                            out_fault[0] = 3; rc = -1; goto cleanup;
                        }
                        r_fcnt[ri]++;
                    } else {
                        int32_t p0 = link_place_off[l], p1 = link_place_off[l + 1], p;
                        int32_t base = r_uoff[ri];
                        for (p = p0; p < p1; p++) {
                            int32_t pu = place_units[p];
                            if (pu < 0 || pu >= cnt) continue;
                            if (u_cnt[base + pu] == 0) r_umap_n[ri]++;
                            if (!msim_fc_push_node(&nodes, &n_nodes, &cap_nodes,
                                                   &u_head[base + pu], &u_tail[base + pu],
                                                   f_idx, weight)) {
                                out_fault[0] = 3; rc = -1; goto cleanup;
                            }
                            u_cnt[base + pu]++;
                        }
                    }
                }
            }
        }
    }

    if (n_flows == 0) goto cleanup;

    /* Step 2: Fold private units into flow_limits and collect shared_units. */
    for (r = 0; r < n_res; r++) {
        int32_t fcnt = r_fcnt[r], umap_n = r_umap_n[r], cnt = r_cnt[r];
        int32_t base = r_uoff[r];
        double u_cap = r_ucap[r], sat_tol = r_sattol[r], bthresh = r_bthresh[r];

        if (umap_n == 0) {
            if (fcnt == 1) {
                msim_fc_node *nd = &nodes[r_fhead[r]];
                int32_t f_idx = nd->flow;
                double w = nd->w;
                if (n_priv >= cap_priv) {
                    int32_t ncap = cap_priv > 0 ? (cap_priv * 2) : 64;
                    msim_fc_priv *next = (msim_fc_priv *)realloc(privs, (size_t)ncap * sizeof(msim_fc_priv));
                    if (!next) { out_fault[0] = 3; rc = -1; goto cleanup; }
                    privs = next; cap_priv = ncap;
                }
                privs[n_priv].res = r; privs[n_priv].unit = 0; privs[n_priv].cap = u_cap;
                privs[n_priv].flow = f_idx; privs[n_priv].w = w; n_priv++;
                if (w > 0.0) {
                    double b = u_cap / w;
                    if (b < flow_limits[f_idx]) flow_limits[f_idx] = b;
                    while (flow_limits[f_idx] > 0.0 && flow_limits[f_idx] * w > u_cap)
                        flow_limits[f_idx] = nextafter(flow_limits[f_idx], 0.0);
                }
            } else if (fcnt > 1) {
                int32_t start = n_terms, nd;
                for (nd = r_fhead[r]; nd >= 0; nd = nodes[nd].next) {
                    if (!msim_fc_push_term(&terms, &n_terms, &cap_terms, nodes[nd].flow, nodes[nd].w)) {
                        out_fault[0] = 3; rc = -1; goto cleanup;
                    }
                }
                if (n_shared >= cap_shared) {
                    int32_t ncap = cap_shared > 0 ? (cap_shared * 2) : 64;
                    msim_fc_unit *next = (msim_fc_unit *)realloc(shared, (size_t)ncap * sizeof(msim_fc_unit));
                    if (!next) { out_fault[0] = 3; rc = -1; goto cleanup; }
                    shared = next; cap_shared = ncap;
                }
                shared[n_shared].res = r; shared[n_shared].unit = 0; shared[n_shared].cap = u_cap;
                shared[n_shared].sat_tol = sat_tol; shared[n_shared].bind_thresh = bthresh;
                shared[n_shared].start = start; shared[n_shared].count = n_terms - start;
                n_shared++;
            }
            continue;
        }
        if (fcnt == 0) {
            for (u = 0; u < cnt; u++) {
                int32_t uc = u_cnt[base + u];
                if (uc == 0) continue;
                if (uc == 1) {
                    msim_fc_node *nd = &nodes[u_head[base + u]];
                    int32_t f_idx = nd->flow;
                    double w = nd->w;
                    if (n_priv >= cap_priv) {
                        int32_t ncap = cap_priv > 0 ? (cap_priv * 2) : 64;
                        msim_fc_priv *next = (msim_fc_priv *)realloc(privs, (size_t)ncap * sizeof(msim_fc_priv));
                        if (!next) { out_fault[0] = 3; rc = -1; goto cleanup; }
                        privs = next; cap_priv = ncap;
                    }
                    privs[n_priv].res = r; privs[n_priv].unit = u; privs[n_priv].cap = u_cap;
                    privs[n_priv].flow = f_idx; privs[n_priv].w = w; n_priv++;
                    if (w > 0.0) {
                        double b = u_cap / w;
                        if (b < flow_limits[f_idx]) flow_limits[f_idx] = b;
                        while (flow_limits[f_idx] > 0.0 && flow_limits[f_idx] * w > u_cap)
                            flow_limits[f_idx] = nextafter(flow_limits[f_idx], 0.0);
                    }
                } else {
                    int32_t start = n_terms, nd;
                    for (nd = u_head[base + u]; nd >= 0; nd = nodes[nd].next) {
                        if (!msim_fc_push_term(&terms, &n_terms, &cap_terms, nodes[nd].flow, nodes[nd].w)) {
                            out_fault[0] = 3; rc = -1; goto cleanup;
                        }
                    }
                    if (n_shared >= cap_shared) {
                        int32_t ncap = cap_shared > 0 ? (cap_shared * 2) : 64;
                        msim_fc_unit *next = (msim_fc_unit *)realloc(shared, (size_t)ncap * sizeof(msim_fc_unit));
                        if (!next) { out_fault[0] = 3; rc = -1; goto cleanup; }
                        shared = next; cap_shared = ncap;
                    }
                    shared[n_shared].res = r; shared[n_shared].unit = u; shared[n_shared].cap = u_cap;
                    shared[n_shared].sat_tol = sat_tol; shared[n_shared].bind_thresh = bthresh;
                    shared[n_shared].start = start; shared[n_shared].count = n_terms - start;
                    n_shared++;
                }
            }
            continue;
        }
        /* Both full-placement flows and explicit partial-unit flows present. */
        if (umap_n < cnt) {
            int32_t u_rep = 0;
            for (u = 0; u < cnt; u++) {
                if (u_cnt[base + u] == 0) { u_rep = u; break; }
            }
            if (fcnt == 1) {
                msim_fc_node *nd = &nodes[r_fhead[r]];
                int32_t f_idx = nd->flow;
                double w = nd->w;
                if (n_priv >= cap_priv) {
                    int32_t ncap = cap_priv > 0 ? (cap_priv * 2) : 64;
                    msim_fc_priv *next = (msim_fc_priv *)realloc(privs, (size_t)ncap * sizeof(msim_fc_priv));
                    if (!next) { out_fault[0] = 3; rc = -1; goto cleanup; }
                    privs = next; cap_priv = ncap;
                }
                privs[n_priv].res = r; privs[n_priv].unit = u_rep; privs[n_priv].cap = u_cap;
                privs[n_priv].flow = f_idx; privs[n_priv].w = w; n_priv++;
                if (w > 0.0) {
                    double b = u_cap / w;
                    if (b < flow_limits[f_idx]) flow_limits[f_idx] = b;
                    while (flow_limits[f_idx] > 0.0 && flow_limits[f_idx] * w > u_cap)
                        flow_limits[f_idx] = nextafter(flow_limits[f_idx], 0.0);
                }
            } else {
                int32_t start = n_terms, nd;
                for (nd = r_fhead[r]; nd >= 0; nd = nodes[nd].next) {
                    if (!msim_fc_push_term(&terms, &n_terms, &cap_terms, nodes[nd].flow, nodes[nd].w)) {
                        out_fault[0] = 3; rc = -1; goto cleanup;
                    }
                }
                if (n_shared >= cap_shared) {
                    int32_t ncap = cap_shared > 0 ? (cap_shared * 2) : 64;
                    msim_fc_unit *next = (msim_fc_unit *)realloc(shared, (size_t)ncap * sizeof(msim_fc_unit));
                    if (!next) { out_fault[0] = 3; rc = -1; goto cleanup; }
                    shared = next; cap_shared = ncap;
                }
                shared[n_shared].res = r; shared[n_shared].unit = u_rep; shared[n_shared].cap = u_cap;
                shared[n_shared].sat_tol = sat_tol; shared[n_shared].bind_thresh = bthresh;
                shared[n_shared].start = start; shared[n_shared].count = n_terms - start;
                n_shared++;
            }
        }
        for (u = 0; u < cnt; u++) {
            int32_t start, nd, tcnt;
            if (u_cnt[base + u] == 0) continue;
            start = n_terms;
            for (nd = r_fhead[r]; nd >= 0; nd = nodes[nd].next) {
                if (!msim_fc_push_term(&terms, &n_terms, &cap_terms, nodes[nd].flow, nodes[nd].w)) {
                    out_fault[0] = 3; rc = -1; goto cleanup;
                }
            }
            for (nd = u_head[base + u]; nd >= 0; nd = nodes[nd].next) {
                if (!msim_fc_push_term(&terms, &n_terms, &cap_terms, nodes[nd].flow, nodes[nd].w)) {
                    out_fault[0] = 3; rc = -1; goto cleanup;
                }
            }
            tcnt = n_terms - start;
            msim_fc_sort_terms(terms + start, tcnt);
            if (n_shared >= cap_shared) {
                int32_t ncap = cap_shared > 0 ? (cap_shared * 2) : 64;
                msim_fc_unit *next = (msim_fc_unit *)realloc(shared, (size_t)ncap * sizeof(msim_fc_unit));
                if (!next) { out_fault[0] = 3; rc = -1; goto cleanup; }
                shared = next; cap_shared = ncap;
            }
            shared[n_shared].res = r; shared[n_shared].unit = u; shared[n_shared].cap = u_cap;
            shared[n_shared].sat_tol = sat_tol; shared[n_shared].bind_thresh = bthresh;
            shared[n_shared].start = start; shared[n_shared].count = tcnt;
            n_shared++;
        }
    }

    alphas = (double *)malloc((size_t)n_flows * sizeof(double));
    if (!alphas) { out_fault[0] = 3; rc = -1; goto cleanup; }

    fsum_cap = 16;
    for (i = 0; i < n_shared; i++) {
        if (shared[i].count > fsum_cap) fsum_cap = shared[i].count;
    }
    fsum_buf = (double *)malloc((size_t)fsum_cap * sizeof(double));
    if (!fsum_buf) { out_fault[0] = 3; rc = -1; goto cleanup; }

    /* Step 3: Binding filter, dedup, degree-1 shortcut & progressive filling. */
    if (n_shared == 0) {
        memcpy(alphas, flow_limits, (size_t)n_flows * sizeof(double));
    } else {
        int32_t hcap = 16;
        while (hcap < n_shared * 2) hcap *= 2;
        htab = (int32_t *)malloc((size_t)hcap * sizeof(int32_t));
        binding = (msim_fc_bind *)malloc((size_t)n_shared * sizeof(msim_fc_bind));
        if (!htab || !binding) { out_fault[0] = 3; rc = -1; goto cleanup; }
        for (i = 0; i < hcap; i++) htab[i] = -1;

        for (i = 0; i < n_shared; i++) {
            const msim_fc_unit *su = &shared[i];
            const msim_fc_term *ut = terms + su->start;
            double max_tot = msim_fc_unit_sum(ut, su->count, flow_limits, fsum_buf);
            if (max_tot > su->bind_thresh) {
                uint64_t h = msim_fc_hash_unit(su->cap, ut, su->count);
                uint32_t mask = (uint32_t)(hcap - 1);
                uint32_t pos = (uint32_t)h & mask;
                bool dup = false;
                while (htab[pos] >= 0) {
                    const msim_fc_bind *b = &binding[htab[pos]];
                    if (msim_fc_same_unit(su->cap, ut, su->count,
                                          b->cap, terms + b->start, b->count)) {
                        dup = true;
                        break;
                    }
                    pos = (pos + 1u) & mask;
                }
                if (!dup) {
                    htab[pos] = n_binding;
                    binding[n_binding].cap = su->cap;
                    binding[n_binding].sat_tol = su->sat_tol;
                    binding[n_binding].start = su->start;
                    binding[n_binding].count = su->count;
                    n_binding++;
                }
            }
        }
        if (n_binding == 0) {
            memcpy(alphas, flow_limits, (size_t)n_flows * sizeof(double));
        } else {
            deg = (int32_t *)calloc((size_t)n_flows, sizeof(int32_t));
            act_flag = (uint8_t *)calloc((size_t)n_flows, sizeof(uint8_t));
            rem_units = (msim_fc_bind *)malloc((size_t)n_binding * sizeof(msim_fc_bind));
            if (!deg || !act_flag || !rem_units) { out_fault[0] = 3; rc = -1; goto cleanup; }
            for (i = 0; i < n_flows; i++) alphas[i] = 0.0;
            for (i = 0; i < n_binding; i++) {
                int32_t k;
                const msim_fc_term *ut = terms + binding[i].start;
                for (k = 0; k < binding[i].count; k++) deg[ut[k].flow]++;
            }
            for (i = 0; i < n_flows; i++) {
                double lim = flow_limits[i];
                if (deg[i] == 0) alphas[i] = lim;
                else if (lim > 0.0) act_flag[i] = 1;
            }
            for (i = 0; i < n_binding; i++) {
                const msim_fc_bind *bu = &binding[i];
                const msim_fc_term *ut = terms + bu->start;
                int32_t n_t = bu->count, k;
                bool iso = true;
                for (k = 0; k < n_t; k++) {
                    int32_t f = ut[k].flow;
                    if (deg[f] != 1 || !act_flag[f]) { iso = false; break; }
                }
                if (iso) {
                    double slope, denom, b_val;
                    bool fits = true;
                    if (n_t == 1) slope = ut[0].w;
                    else if (n_t == 2) slope = ut[0].w + ut[1].w;
                    else {
                        for (k = 0; k < n_t; k++) fsum_buf[k] = ut[k].w;
                        slope = msim_fsum(fsum_buf, n_t);
                    }
                    denom = (n_t == 1) ? slope : nextafter(slope, INFINITY);
                    b_val = bu->cap / denom;
                    for (k = 0; k < n_t; k++) {
                        if (!(b_val <= flow_limits[ut[k].flow])) { fits = false; break; }
                    }
                    if (fits) {
                        double delta = b_val;
                        bool solved = false;
                        int32_t attempt;
                        for (attempt = 0; attempt < MSIM_FC_MAX_ATTEMPTS; attempt++) {
                            double tot;
                            if (delta <= 0.0) break;
                            tot = msim_fc_delta_sum(ut, n_t, delta, fsum_buf);
                            if (tot <= bu->cap) {
                                double res_v = bu->cap - tot;
                                double tol = msim_fc_sat_tol_exact(bu->cap, res_v, bu->sat_tol);
                                if (res_v <= tol) {
                                    for (k = 0; k < n_t; k++) {
                                        alphas[ut[k].flow] = delta;
                                        act_flag[ut[k].flow] = 0;
                                    }
                                    solved = true;
                                }
                                break;
                            }
                            delta = (attempt < MSIM_FC_NEXTAFTER_STEPS)
                                ? nextafter(delta, 0.0) : (delta * MSIM_FC_HALF_STEP);
                        }
                        if (solved) continue;
                    }
                }
                rem_units[n_rem++] = *bu;
            }

            if (n_rem > 0) {
                int32_t n_act = 0, n_open = n_rem, tot_comm = 0;
                act_idx = (int32_t *)malloc((size_t)n_flows * sizeof(int32_t));
                next_act = (int32_t *)malloc((size_t)n_flows * sizeof(int32_t));
                open_sh = (int32_t *)malloc((size_t)n_rem * sizeof(int32_t));
                still_open = (int32_t *)malloc((size_t)n_rem * sizeof(int32_t));
                next_open = (int32_t *)malloc((size_t)n_rem * sizeof(int32_t));
                comm_off = (int32_t *)malloc((size_t)(n_rem + 1) * sizeof(int32_t));
                totals = (double *)calloc((size_t)n_rem, sizeof(double));
                cand_tot = (double *)malloc((size_t)n_rem * sizeof(double));
                if (!act_idx || !next_act || !open_sh || !still_open || !next_open
                        || !comm_off || !totals || !cand_tot) {
                    out_fault[0] = 3; rc = -1; goto cleanup;
                }
                for (i = 0; i < n_flows; i++) if (act_flag[i]) act_idx[n_act++] = i;
                for (i = 0; i < n_rem; i++) {
                    open_sh[i] = i;
                    comm_off[i] = tot_comm;
                    tot_comm += rem_units[i].count;
                }
                comm_off[n_rem] = tot_comm;
                committed = (double *)calloc((size_t)(tot_comm > 0 ? tot_comm : 1), sizeof(double));
                if (!committed) { out_fault[0] = 3; rc = -1; goto cleanup; }

                while (n_act > 0 && n_open > 0) {
                    double bound_limit = INFINITY, headroom = INFINITY, delta;
                    int32_t n_still = 0, s_pos, attempt, n_cand = 0;
                    bool advanced = false, froze_any = false;
                    int32_t n_next_open = 0, n_next_act = 0;

                    for (s_pos = 0; s_pos < n_open; s_pos++) {
                        int32_t s_idx = open_sh[s_pos];
                        const msim_fc_bind *ru = &rem_units[s_idx];
                        const msim_fc_term *ut = terms + ru->start;
                        if (ru->count == 2) {
                            int32_t f0 = ut[0].flow, f1 = ut[1].flow;
                            double w0 = ut[0].w, w1 = ut[1].w;
                            uint8_t a0 = act_flag[f0], a1 = act_flag[f1];
                            double slope, denom;
                            if (!a0 && !a1) continue;
                            still_open[n_still++] = s_idx;
                            if (a0 && a1) {
                                slope = w0 + w1;
                                denom = slope > 0.0 ? nextafter(slope, INFINITY) : 0.0;
                            } else {
                                slope = a0 ? w0 : w1;
                                denom = slope;
                            }
                            if (denom > 0.0) {
                                double b = (ru->cap - totals[s_idx]) / denom;
                                if (b < bound_limit) bound_limit = b;
                            }
                        } else {
                            int32_t cnt_w = 0, k;
                            for (k = 0; k < ru->count; k++) {
                                if (act_flag[ut[k].flow]) fsum_buf[cnt_w++] = ut[k].w;
                            }
                            if (cnt_w > 0) {
                                double slope;
                                still_open[n_still++] = s_idx;
                                slope = (cnt_w == 1) ? fsum_buf[0] : msim_fsum(fsum_buf, cnt_w);
                                if (slope > 0.0) {
                                    double denom = (cnt_w == 1) ? slope : nextafter(slope, INFINITY);
                                    double b = (ru->cap - totals[s_idx]) / denom;
                                    if (b < bound_limit) bound_limit = b;
                                }
                            }
                        }
                    }
                    memcpy(open_sh, still_open, (size_t)n_still * sizeof(int32_t));
                    n_open = n_still;
                    if (n_open == 0) {
                        for (i = 0; i < n_act; i++) alphas[act_idx[i]] = flow_limits[act_idx[i]];
                        break;
                    }
                    for (i = 0; i < n_act; i++) {
                        int32_t fi = act_idx[i];
                        double hr = flow_limits[fi] - alphas[fi];
                        if (hr < headroom) headroom = hr;
                    }
                    delta = bound_limit < headroom ? bound_limit : headroom;
                    if (!isfinite(delta) || delta < 0.0) break;

                    for (attempt = 0; attempt < MSIM_FC_MAX_ATTEMPTS; attempt++) {
                        bool unsafe = false;
                        if (delta <= 0.0) { n_cand = 0; break; }
                        for (i = 0; i < n_act; i++) {
                            int32_t fi = act_idx[i];
                            if (alphas[fi] + delta > flow_limits[fi]) { unsafe = true; break; }
                        }
                        n_cand = 0;
                        if (!unsafe) {
                            for (s_pos = 0; s_pos < n_open; s_pos++) {
                                int32_t s_idx = open_sh[s_pos];
                                const msim_fc_bind *ru = &rem_units[s_idx];
                                const msim_fc_term *ut = terms + ru->start;
                                const double *cs = committed + comm_off[s_idx];
                                double tot;
                                if (ru->count == 2) {
                                    int32_t f0 = ut[0].flow, f1 = ut[1].flow;
                                    double v0 = act_flag[f0] ? ((alphas[f0] + delta) * ut[0].w) : cs[0];
                                    double v1 = act_flag[f1] ? ((alphas[f1] + delta) * ut[1].w) : cs[1];
                                    tot = v0 + v1;
                                } else {
                                    int32_t k;
                                    for (k = 0; k < ru->count; k++) {
                                        int32_t fk = ut[k].flow;
                                        fsum_buf[k] = act_flag[fk] ? ((alphas[fk] + delta) * ut[k].w) : cs[k];
                                    }
                                    tot = (ru->count == 1) ? fsum_buf[0] : msim_fsum(fsum_buf, ru->count);
                                }
                                if (tot > ru->cap) { unsafe = true; break; }
                                cand_tot[n_cand++] = tot;
                            }
                        }
                        if (!unsafe) break;
                        delta = (attempt < MSIM_FC_NEXTAFTER_STEPS)
                            ? nextafter(delta, 0.0) : (delta * MSIM_FC_HALF_STEP);
                        if (attempt + 1 == MSIM_FC_MAX_ATTEMPTS) {
                            delta = 0.0;
                            n_cand = 0;
                        }
                    }

                    if (delta > 0.0) {
                        for (i = 0; i < n_act; i++) {
                            int32_t fi = act_idx[i];
                            double lim_f = flow_limits[fi];
                            double raw_f = alphas[fi] + delta;
                            double val = raw_f > lim_f ? lim_f : raw_f;
                            if (val != alphas[fi]) advanced = true;
                            alphas[fi] = val;
                        }
                        for (s_pos = 0; s_pos < n_open; s_pos++) {
                            int32_t s_idx = open_sh[s_pos];
                            const msim_fc_bind *ru = &rem_units[s_idx];
                            const msim_fc_term *ut = terms + ru->start;
                            double *cs = committed + comm_off[s_idx];
                            int32_t k;
                            for (k = 0; k < ru->count; k++) {
                                int32_t fk = ut[k].flow;
                                if (act_flag[fk]) cs[k] = alphas[fk] * ut[k].w;
                            }
                            if (s_pos < n_cand) totals[s_idx] = cand_tot[s_pos];
                        }
                    }

                    for (s_pos = 0; s_pos < n_open; s_pos++) {
                        int32_t s_idx = open_sh[s_pos];
                        const msim_fc_bind *ru = &rem_units[s_idx];
                        double res_v = ru->cap - totals[s_idx];
                        double tol = msim_fc_sat_tol_exact(ru->cap, res_v, ru->sat_tol);
                        if (res_v <= tol) {
                            const msim_fc_term *ut = terms + ru->start;
                            int32_t k;
                            for (k = 0; k < ru->count; k++) {
                                if (act_flag[ut[k].flow]) {
                                    act_flag[ut[k].flow] = 0;
                                    froze_any = true;
                                }
                            }
                        } else {
                            next_open[n_next_open++] = s_idx;
                        }
                    }
                    memcpy(open_sh, next_open, (size_t)n_next_open * sizeof(int32_t));
                    n_open = n_next_open;

                    for (i = 0; i < n_act; i++) {
                        int32_t fi = act_idx[i];
                        double lim, al, max_la, cap_tol;
                        if (!act_flag[fi]) continue;
                        lim = flow_limits[fi];
                        al = alphas[fi];
                        max_la = fabs(lim) >= fabs(al) ? fabs(lim) : fabs(al);
                        cap_tol = MSIM_FC_CAP_ULPS * msim_ulp(max_la);
                        if (al >= lim - cap_tol || (!advanced && (lim - al) <= headroom + cap_tol)) {
                            act_flag[fi] = 0;
                            froze_any = true;
                        } else {
                            next_act[n_next_act++] = fi;
                        }
                    }
                    if (!froze_any) break;
                    memcpy(act_idx, next_act, (size_t)n_next_act * sizeof(int32_t));
                    n_act = n_next_act;
                }
            }
        }
    }

    /* Step 4: Independent final overallocation guard on private & shared units. */
    for (i = 0; i < n_priv; i++) {
        if (alphas[privs[i].flow] * privs[i].w > privs[i].cap) {
            out_fault[0] = 2;
            out_fault[1] = privs[i].res;
            out_fault[2] = privs[i].unit;
            rc = -2;
            goto cleanup;
        }
    }
    for (i = 0; i < n_shared; i++) {
        const msim_fc_unit *su = &shared[i];
        double tot_u = msim_fc_unit_sum(terms + su->start, su->count, alphas, fsum_buf);
        if (tot_u > su->cap) {
            out_fault[0] = 2;
            out_fault[1] = su->res;
            out_fault[2] = su->unit;
            rc = -2;
            goto cleanup;
        }
    }

    /* Step 5: Emit solved per-link rates. */
    for (i = 0; i < n_flows; i++) {
        double alpha = alphas[i];
        int32_t s0, s1, s;
        if (alpha <= 0.0) continue;
        s0 = flow_span_off[i];
        s1 = flow_span_off[i + 1];
        for (s = s0; s < s1; s++) {
            out_rate[flow_spans[s].link_idx] += alpha * flow_spans[s].demand;
        }
    }

cleanup:
    free(span_full); free(span_nu); free(span_l);
    free(fsum_buf);
    free(cand_tot); free(totals); free(committed); free(comm_off);
    free(next_open); free(still_open); free(open_sh);
    free(next_act); free(act_idx);
    free(rem_units); free(act_flag); free(deg);
    free(htab); free(binding);
    free(alphas);
    free(terms); free(shared); free(privs);
    free(flow_spans); free(flow_span_off); free(flow_limits);
    free(nodes);
    free(u_cnt); free(u_tail); free(u_head);
    free(r_umap_n); free(r_fcnt); free(r_ftail); free(r_fhead);
    free(r_uoff); free(r_cnt); free(r_bthresh); free(r_sattol); free(r_ucap);
    return rc;
}

#endif /* !msim_abi_version */
#endif /* MSIM_ENGINE_ACCEL_COUPLED_H */
