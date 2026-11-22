/* Prepared-row max-min solver, 2-row HBM bus terms, and observation buffer.
 *
 * [Role] Provide pure C11 structs and exported entrypoints for 2-row
 *   directional HBM bus constraints (``msim_hbm_terms``), 3-ledger access
 *   validation (``msim_validate_access``), normalized prepared-row weighted
 *   max-min allocation and integration (``msim_config_rows``,
 *   ``msim_prepare_terms``, ``msim_allocate_epoch``, ``msim_integrate_epoch``,
 *   ``msim_read_term_state``), ring-buffer handoff structs, and the bounded
 *   observation output buffer (``msim_set_obs_buffer``, ``msim_drain_obs``).
 * [Invariants] Pure ISO C11 with zero file-scope mutable state; all buffers
 *   live in the caller's ``msim_ctx`` or caller-owned output arrays.
 *   Binary64 operation order matches Python ``prepared_terms.py`` identically.
 * [Timeouts & Locks] Every progressive-filling loop is bounded by
 *   ``guard <= nprogs + 1``; no threads, locks, sleeps or I/O.
 * [Side-Effect Policy] Mutates only the supplied ``msim_ctx`` prepared-terms
 *   tables and caller-provided output arrays.
 */
#ifndef MSIM_ENGINE_ACCEL_TERMS_H
#define MSIM_ENGINE_ACCEL_TERMS_H

#include "_engine_accel_core.h"

enum {
    MSIM_HBM_DIR_READ = 0,
    MSIM_HBM_DIR_WRITE = 1,
    MSIM_ROW_POLICY_FAIR = 0,
    MSIM_ROW_POLICY_FIFO = 1,
    MSIM_OBS_SERVICE = 1,
    MSIM_OBS_WAIT = 2,
    MSIM_OBS_ALLOC = 3,
    MSIM_OBS_PERIOD = 4
};

typedef struct {
    uint32_t offset;
    uint32_t count;
} msim_slice;

typedef struct {
    uint32_t die;
    uint32_t row_r;
    uint32_t row_w;
    uint32_t ctr_r;
    uint32_t ctr_w;
    double bw_r_bytes_per_ms;
    double bw_w_bytes_per_ms;
    double alpha_rw;
} msim_hbm_bus;

typedef struct {
    uint64_t logical_bytes;
    uint64_t aligned_bytes;
    uint64_t hit_bytes;
    uint64_t miss_bytes;
    uint64_t sector_bytes;
    double dram_service_bytes;
    uint32_t copy_pipe_row;
    uint32_t l2_row;
    uint16_t die;
    uint8_t direction;
    uint8_t flags;
} msim_access;

typedef struct {
    uint32_t row;
    uint32_t flow;
    double ms_per_z;
} msim_term;

typedef struct {
    uint32_t counter;
    uint32_t flow;
    double work_per_z;
} msim_delivery;

typedef struct {
    uint64_t sequence;
    uint32_t task;
    uint32_t placement_id;
    uint32_t member_map;
    int32_t priority;
    double z_left;
    double weight;
    double max_dz_ms;
} msim_progress;

typedef struct {
    uint32_t row_id;
    uint32_t scope;
    uint32_t unit;
    uint8_t policy;
    uint8_t pad[7];
    double capacity;
} msim_capacity_row;

#define MSIM_TERMS_ABI_VERSION 3u
#define MSIM_TERMS_ABI_PROBE_CELLS 16u

typedef struct {
    uint64_t occurrence;
    uint64_t generation;
    uint64_t allocation_id;
    uint32_t die;
    uint32_t owner;
    uint32_t pool_id;
    uint32_t slot_id;
    uint32_t producer_node;
    uint32_t publish_node;
    uint32_t last_use_node;
    uint32_t consumer_count;
} msim_slot_gen;

typedef struct {
    uint32_t src_node;
    uint32_t dst_node;
    uint16_t kind;
    uint16_t flags;
    uint32_t pad;
} msim_dep;

static uint64_t msim_ms_to_ns_even(double t_ms)
{
    double scaled = t_ms * 1000000.0;
    double fl, frac, rounded;
    if (!isfinite(scaled) || scaled <= 0.0) return 0u;
    if (scaled >= 18446744073709551616.0) return UINT64_MAX;
    fl = floor(scaled);
    frac = scaled - fl;
    if (frac > 0.5) rounded = fl + 1.0;
    else if (frac < 0.5) rounded = fl;
    else rounded = (fmod(fl, 2.0) != 0.0) ? (fl + 1.0) : fl;
    if (rounded >= 18446744073709551616.0) return UINT64_MAX;
    return (uint64_t)rounded;
}

static bool msim_emit_obs_service(
    msim_ctx *c, double t0_ms, double t1_ms, uint64_t task_seq,
    uint32_t pipe_id, uint32_t owner_id, double used)
{
    msim_obs *rec;
    if (c->obs_buf == NULL) return true;
    if (c->obs_count >= c->obs_cap) {
        c->obs_overflow++;
        return false;
    }
    rec = &c->obs_buf[c->obs_count++];
    rec->t_start_ns = msim_ms_to_ns_even(t0_ms);
    rec->t_end_ns = msim_ms_to_ns_even(t1_ms);
    rec->task_id = task_seq;
    rec->pipe_id = pipe_id;
    rec->owner_id = owner_id;
    rec->kind = MSIM_OBS_SERVICE;
    rec->reason_code = 0;
    rec->work_done = used;
    return true;
}

/* When embedded by the sanitizer harness under renamed symbols, omit exported
 * prepared-terms entrypoints so the second TU emits neither duplicate symbols
 * nor unused-function warnings. */
#ifndef msim_abi_version

MSIM_EXPORT uint32_t msim_terms_abi_version(void)
{
    return MSIM_TERMS_ABI_VERSION;
}

MSIM_EXPORT int32_t msim_terms_abi_probe(uint64_t *out, size_t nout)
{
    if (out == NULL || nout < (size_t)MSIM_TERMS_ABI_PROBE_CELLS)
        return MSIM_E_RANGE;
    out[0] = (uint64_t)MSIM_TERMS_ABI_VERSION;
    out[1] = (uint64_t)sizeof(msim_hbm_bus);
    out[2] = (uint64_t)offsetof(msim_hbm_bus, alpha_rw);
    out[3] = (uint64_t)sizeof(msim_access);
    out[4] = (uint64_t)offsetof(msim_access, dram_service_bytes);
    out[5] = (uint64_t)sizeof(msim_term);
    out[6] = (uint64_t)sizeof(msim_delivery);
    out[7] = (uint64_t)sizeof(msim_progress);
    out[8] = (uint64_t)offsetof(msim_progress, max_dz_ms);
    out[9] = (uint64_t)sizeof(msim_capacity_row);
    out[10] = (uint64_t)offsetof(msim_capacity_row, capacity);
    out[11] = (uint64_t)sizeof(msim_obs);
    out[12] = (uint64_t)offsetof(msim_obs, work_done);
    out[13] = (uint64_t)sizeof(msim_slot_gen);
    out[14] = (uint64_t)offsetof(msim_slot_gen, consumer_count);
    out[15] = (uint64_t)sizeof(msim_dep);
    return 0;
}

MSIM_EXPORT int32_t msim_validate_ring(
    const msim_slot_gen *slots, uint32_t nslots,
    const msim_dep *deps, uint32_t ndeps, uint32_t nnodes)
{
    uint32_t i, j;
    if ((nslots > 0 && slots == NULL) || (ndeps > 0 && deps == NULL))
        return MSIM_E_RANGE;
    for (i = 0; i < nslots; i++) {
        if (slots[i].producer_node >= nnodes || slots[i].publish_node >= nnodes
            || slots[i].last_use_node >= nnodes)
            return MSIM_E_RANGE;
        for (j = i + 1; j < nslots; j++) {
            if (slots[i].occurrence == slots[j].occurrence
                && slots[i].allocation_id == slots[j].allocation_id
                && slots[i].die == slots[j].die
                && slots[i].owner == slots[j].owner
                && slots[i].pool_id == slots[j].pool_id
                && slots[i].slot_id == slots[j].slot_id
                && slots[i].generation == slots[j].generation)
                return MSIM_E_VALUE;
        }
    }
    for (i = 0; i < ndeps; i++) {
        if (deps[i].src_node >= nnodes || deps[i].dst_node >= nnodes)
            return MSIM_E_RANGE;
        if (deps[i].src_node == deps[i].dst_node || deps[i].kind > 5u)
            return MSIM_E_VALUE;
    }
    return 0;
}

/* Emit 2-row directional HBM bus normalized coefficients (ms per unit z). */
MSIM_EXPORT int32_t msim_hbm_terms(
    const msim_hbm_bus *bus, uint8_t dir, double dram_bytes,
    double *out_r, double *out_w)
{
    if (bus == NULL || out_r == NULL || out_w == NULL) return MSIM_E_RANGE;
    if (!isfinite(bus->bw_r_bytes_per_ms) || bus->bw_r_bytes_per_ms <= 0.0)
        return MSIM_E_VALUE;
    if (!isfinite(bus->bw_w_bytes_per_ms) || bus->bw_w_bytes_per_ms <= 0.0)
        return MSIM_E_VALUE;
    if (!isfinite(bus->alpha_rw) || bus->alpha_rw < 0.0 || bus->alpha_rw > 1.0)
        return MSIM_E_VALUE;
    if (!isfinite(dram_bytes) || dram_bytes < 0.0) return MSIM_E_VALUE;
    if (dir == MSIM_HBM_DIR_READ) {
        double base = dram_bytes / bus->bw_r_bytes_per_ms;
        *out_r = base;
        *out_w = bus->alpha_rw * base;
        return 0;
    }
    if (dir == MSIM_HBM_DIR_WRITE) {
        double base = dram_bytes / bus->bw_w_bytes_per_ms;
        *out_r = bus->alpha_rw * base;
        *out_w = base;
        return 0;
    }
    return MSIM_E_VALUE;
}

/* Validate 3-ledger byte conservation on a bound access record. */
MSIM_EXPORT int32_t msim_validate_access(const msim_access *acc)
{
    if (acc == NULL) return MSIM_E_RANGE;
    if (acc->direction > MSIM_HBM_DIR_WRITE) return MSIM_E_VALUE;
    if (acc->hit_bytes > acc->aligned_bytes
        || acc->aligned_bytes - acc->hit_bytes != acc->miss_bytes)
        return MSIM_E_VALUE;
    if (!isfinite(acc->dram_service_bytes) || acc->dram_service_bytes < 0.0)
        return MSIM_E_VALUE;
    if (acc->miss_bytes == 0 && (acc->sector_bytes != 0 || acc->dram_service_bytes != 0.0))
        return MSIM_E_VALUE;
    return 0;
}

/* Attach or detach caller-owned observation buffer on ``msim_ctx``. */
MSIM_EXPORT int32_t msim_set_obs_buffer(void *ctx, msim_obs *buf, uint32_t cap)
{
    msim_ctx *c = (msim_ctx *)ctx;
    if (!ctx_ok(c)) return MSIM_E_STATE;
    if (buf == NULL && cap > 0) return MSIM_E_RANGE;
    c->obs_buf = buf;
    c->obs_cap = cap;
    c->obs_count = 0;
    c->obs_overflow = 0;
    return 0;
}

/* Return filled observation count and reset write cursor & overflow counter. */
MSIM_EXPORT uint32_t msim_drain_obs(void *ctx, uint32_t *out_overflow)
{
    msim_ctx *c = (msim_ctx *)ctx;
    uint32_t n;
    if (!ctx_ok(c)) {
        if (out_overflow != NULL) *out_overflow = 0;
        return 0;
    }
    n = c->obs_count;
    if (out_overflow != NULL) *out_overflow = c->obs_overflow;
    c->obs_count = 0;
    c->obs_overflow = 0;
    return n;
}

/* Configure normalized capacity rows for the prepared-terms epoch solver. */
MSIM_EXPORT int32_t msim_config_rows(
    void *ctx, const msim_capacity_row *rows, uint32_t nrows)
{
    msim_ctx *c = (msim_ctx *)ctx;
    uint32_t i;
    if (!ctx_ok(c)) return MSIM_E_STATE;
    if (nrows > (uint32_t)MSIM_MAX_R) return MSIM_E_RANGE;
    if (nrows > 0 && rows == NULL) return MSIM_E_RANGE;
    for (i = 0; i < nrows; i++) {
        if (rows[i].policy > MSIM_ROW_POLICY_FIFO) return MSIM_E_VALUE;
        if (!isfinite(rows[i].capacity) || rows[i].capacity < 0.0)
            return MSIM_E_VALUE;
    }
    if (nrows > 0 && !msim_grow((void **)&c->pt_rows, &c->pt_row_cap,
                                (int32_t)nrows, sizeof(msim_capacity_row)))
        return MSIM_E_RANGE;
    if (nrows > 0) memcpy(c->pt_rows, rows, (size_t)nrows * sizeof(msim_capacity_row));
    c->pt_nrows = (int32_t)nrows;
    c->pt_nprogs = 0;
    c->pt_nterms = 0;
    c->pt_ndels = 0;
    c->pt_nctrs = 0;
    return 0;
}

static bool prog_order_less(const msim_progress *P, uint32_t a, uint32_t b)
{
    const msim_progress *x = &P[a], *y = &P[b];
    if (x->priority != y->priority) return x->priority < y->priority;
    if (x->sequence != y->sequence) return x->sequence < y->sequence;
    if (x->placement_id != y->placement_id) return x->placement_id < y->placement_id;
    if (x->member_map != y->member_map) return x->member_map < y->member_map;
    return a < b;
}

/* Prepare flows, coalesced (flow, row) terms, and delivery rows inside ``ctx``. */
MSIM_EXPORT int32_t msim_prepare_terms(
    void *ctx,
    const msim_progress *progs, uint32_t nprogs,
    const msim_term *terms, uint32_t nterms,
    const msim_delivery *dels, uint32_t ndels,
    uint32_t ncounters)
{
    msim_ctx *c = (msim_ctx *)ctx;
    uint32_t i, j;
    if (!ctx_ok(c)) return MSIM_E_STATE;
    if (nprogs > (uint32_t)MSIM_MAX_F || nterms > (uint32_t)MSIM_MAX_COEF
        || ndels > (uint32_t)MSIM_MAX_COEF || ncounters > (uint32_t)MSIM_MAX_R)
        return MSIM_E_RANGE;
    if ((nprogs > 0 && progs == NULL) || (nterms > 0 && terms == NULL)
        || (ndels > 0 && dels == NULL))
        return MSIM_E_RANGE;
    for (i = 0; i < nprogs; i++) {
        if (!isfinite(progs[i].z_left) || progs[i].z_left < 0.0
            || !isfinite(progs[i].weight) || progs[i].weight < 0.0
            || isnan(progs[i].max_dz_ms) || progs[i].max_dz_ms < 0.0)
            return MSIM_E_VALUE;
    }
    for (i = 0; i < nterms; i++) {
        if (terms[i].flow >= nprogs || terms[i].row >= (uint32_t)c->pt_nrows)
            return MSIM_E_RANGE;
        if (!isfinite(terms[i].ms_per_z) || terms[i].ms_per_z < 0.0)
            return MSIM_E_VALUE;
    }
    for (i = 0; i < ndels; i++) {
        if (dels[i].flow >= nprogs || dels[i].counter >= ncounters)
            return MSIM_E_RANGE;
        if (!isfinite(dels[i].work_per_z) || dels[i].work_per_z < 0.0)
            return MSIM_E_VALUE;
    }
    if (nprogs > 0 && !msim_grow((void **)&c->pt_progs, &c->pt_prog_cap,
                                 (int32_t)nprogs, sizeof(msim_progress)))
        return MSIM_E_RANGE;
    if (nterms > 0 && !msim_grow((void **)&c->pt_terms, &c->pt_term_cap,
                                 (int32_t)nterms, sizeof(msim_term)))
        return MSIM_E_RANGE;
    if (ndels > 0 && !msim_grow((void **)&c->pt_dels, &c->pt_del_cap,
                                (int32_t)ndels, sizeof(msim_delivery)))
        return MSIM_E_RANGE;
    if (ncounters > 0 && !msim_grow((void **)&c->pt_ctrs, &c->pt_ctr_cap,
                                    (int32_t)ncounters, sizeof(double)))
        return MSIM_E_RANGE;
    if (nprogs > 0) memcpy(c->pt_progs, progs, (size_t)nprogs * sizeof(msim_progress));
    if (ndels > 0) memcpy(c->pt_dels, dels, (size_t)ndels * sizeof(msim_delivery));
    if (ncounters > 0) memset(c->pt_ctrs, 0, (size_t)ncounters * sizeof(double));
    /* Copy positive terms, sort by (flow, row), and coalesce duplicate pairs. */
    {
        msim_term *out = (msim_term *)c->pt_terms;
        uint32_t kept = 0, w = 0;
        for (i = 0; i < nterms; i++) {
            if (terms[i].ms_per_z <= 0.0) continue;
            j = kept;
            while (j > 0 && (out[j - 1].flow > terms[i].flow
                             || (out[j - 1].flow == terms[i].flow
                                 && out[j - 1].row > terms[i].row))) {
                out[j] = out[j - 1];
                j--;
            }
            out[j] = terms[i];
            kept++;
        }
        for (i = 0; i < kept; i++) {
            if (w > 0 && out[w - 1].flow == out[i].flow && out[w - 1].row == out[i].row) {
                out[w - 1].ms_per_z += out[i].ms_per_z;
            } else {
                out[w++] = out[i];
            }
        }
        c->pt_nterms = (int32_t)w;
    }
    c->pt_nprogs = (int32_t)nprogs;
    c->pt_ndels = (int32_t)ndels;
    c->pt_nctrs = (int32_t)ncounters;
    return 0;
}

/* Weighted max-min epoch allocation matching Python ``allocate_prepared_epoch``. */
MSIM_EXPORT int32_t msim_allocate_epoch(void *ctx, double *out_dz, uint32_t nprogs)
{
    msim_ctx *c = (msim_ctx *)ctx;
    const msim_capacity_row *rows;
    const msim_progress *progs;
    const msim_term *terms;
    int32_t nr, np, nt, i, j, p_start;
    uint32_t *order;
    int32_t *t_start, *t_end, *fifo_win, *free_list, *next_free;
    double *rem_cap, *slope, *bounds;
    bool *sat;
    if (!ctx_ok(c)) return MSIM_E_STATE;
    nr = c->pt_nrows;
    np = c->pt_nprogs;
    nt = c->pt_nterms;
    if ((int32_t)nprogs != np) return MSIM_E_RANGE;
    if (np > 0 && out_dz == NULL) return MSIM_E_RANGE;
    for (i = 0; i < np; i++) out_dz[i] = 0.0;
    if (nr == 0 || np == 0) return 0;

    rows = (const msim_capacity_row *)c->pt_rows;
    progs = (const msim_progress *)c->pt_progs;
    terms = (const msim_term *)c->pt_terms;
    for (i = 0; i < nt; i++) {
        if (terms[i].flow >= (uint32_t)np || terms[i].row >= (uint32_t)nr)
            return MSIM_E_RANGE;
    }

    order = (uint32_t *)malloc((size_t)np * sizeof(uint32_t));
    t_start = (int32_t *)malloc((size_t)np * sizeof(int32_t));
    t_end = (int32_t *)malloc((size_t)np * sizeof(int32_t));
    free_list = (int32_t *)malloc((size_t)np * sizeof(int32_t));
    next_free = (int32_t *)malloc((size_t)np * sizeof(int32_t));
    fifo_win = (int32_t *)malloc((size_t)nr * sizeof(int32_t));
    rem_cap = (double *)malloc((size_t)nr * sizeof(double));
    slope = (double *)malloc((size_t)nr * sizeof(double));
    bounds = (double *)malloc((size_t)nr * sizeof(double));
    sat = (bool *)malloc((size_t)nr * sizeof(bool));
    if (!order || !t_start || !t_end || !free_list || !next_free
        || !fifo_win || !rem_cap || !slope || !bounds || !sat) {
        free(order); free(t_start); free(t_end); free(free_list); free(next_free);
        free(fifo_win); free(rem_cap); free(slope); free(bounds); free(sat);
        return MSIM_E_RANGE;
    }

    for (i = 0; i < np; i++) { t_start[i] = 0; t_end[i] = 0; }
    for (i = 0; i < nt; i++) {
        int32_t f = (int32_t)terms[i].flow;
        if (t_end[f] == 0) t_start[f] = i;
        t_end[f] = i + 1;
    }
    for (i = 0; i < np; i++) {
        uint32_t cur = (uint32_t)i;
        j = i;
        while (j > 0 && prog_order_less(progs, cur, order[j - 1])) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = cur;
    }
    for (j = 0; j < nr; j++) {
        fifo_win[j] = -1;
        rem_cap[j] = rows[j].capacity;
    }
    for (i = 0; i < np; i++) {
        int32_t f = (int32_t)order[i];
        if (progs[f].z_left <= 0.0 || progs[f].weight <= 0.0 || progs[f].max_dz_ms <= 0.0)
            continue;
        for (j = t_start[f]; j < t_end[f]; j++) {
            uint32_t r = terms[j].row;
            if (terms[j].ms_per_z > 0.0 && rows[r].policy == MSIM_ROW_POLICY_FIFO
                && fifo_win[r] < 0)
                fifo_win[r] = f;
        }
    }

    p_start = 0;
    while (p_start < np) {
        int32_t prio = progs[order[p_start]].priority;
        int32_t p_end = p_start + 1, nf = 0, guard = 0;
        while (p_end < np && progs[order[p_end]].priority == prio) p_end++;
        for (i = p_start; i < p_end; i++) {
            int32_t f = (int32_t)order[i];
            bool blocked = false;
            if (progs[f].z_left <= 0.0 || progs[f].weight <= 0.0 || progs[f].max_dz_ms <= 0.0)
                continue;
            for (j = t_start[f]; j < t_end[f]; j++) {
                uint32_t r = terms[j].row;
                if (terms[j].ms_per_z > 0.0) {
                    if (rows[r].policy == MSIM_ROW_POLICY_FIFO && fifo_win[r] != f) {
                        blocked = true; break;
                    }
                    if (rem_cap[r] <= 0.0) { blocked = true; break; }
                }
            }
            if (!blocked) free_list[nf++] = f;
        }
        while (nf > 0 && guard++ <= np + 1) {
            double delta = INFINITY;
            int32_t nnf = 0;
            for (j = 0; j < nr; j++) { slope[j] = 0.0; bounds[j] = INFINITY; sat[j] = false; }
            for (i = 0; i < nf; i++) {
                int32_t f = free_list[i];
                double wf = progs[f].weight;
                for (j = t_start[f]; j < t_end[f]; j++)
                    slope[terms[j].row] += terms[j].ms_per_z * wf;
            }
            for (j = 0; j < nr; j++) {
                if (slope[j] > 0.0) {
                    double b = rem_cap[j] / slope[j];
                    bounds[j] = b;
                    if (b < delta) delta = b;
                }
            }
            for (i = 0; i < nf; i++) {
                int32_t f = free_list[i];
                double rv = (progs[f].max_dz_ms - out_dz[f]) / progs[f].weight;
                if (rv < delta) delta = rv;
            }
            if (!isfinite(delta) || delta < 0.0) break;
            for (i = 0; i < nf; i++) {
                int32_t f = free_list[i];
                out_dz[f] += delta * progs[f].weight;
            }
            for (j = 0; j < nr; j++) {
                if (slope[j] > 0.0) {
                    double nv = rem_cap[j] - delta * slope[j];
                    rem_cap[j] = nv > 0.0 ? nv : 0.0;
                    if (bounds[j] <= delta || rem_cap[j] <= 0.0) sat[j] = true;
                }
            }
            for (i = 0; i < nf; i++) {
                int32_t f = free_list[i];
                bool hit = false;
                if (out_dz[f] >= progs[f].max_dz_ms) continue;
                for (j = t_start[f]; j < t_end[f]; j++) {
                    if (sat[terms[j].row]) { hit = true; break; }
                }
                if (!hit) next_free[nnf++] = f;
            }
            for (i = 0; i < nnf; i++) free_list[i] = next_free[i];
            nf = nnf;
        }
        p_start = p_end;
    }

    free(order); free(t_start); free(t_end); free(free_list); free(next_free);
    free(fifo_win); free(rem_cap); free(slope); free(bounds); free(sat);
    return 0;
}

/* Integrate one epoch ``dt_ms`` on prepared progress & counter deliveries. */
MSIM_EXPORT int32_t msim_integrate_epoch(
    void *ctx, const double *dz_per_ms, uint32_t nprogs,
    double dt_ms, double *out_dz_steps)
{
    msim_ctx *c = (msim_ctx *)ctx;
    msim_progress *progs;
    const msim_delivery *dels;
    int32_t np, nd, i;
    double *steps_buf;
    if (!ctx_ok(c)) return MSIM_E_STATE;
    np = c->pt_nprogs;
    nd = c->pt_ndels;
    if ((int32_t)nprogs != np) return MSIM_E_RANGE;
    if (!isfinite(dt_ms) || dt_ms < 0.0) return MSIM_E_VALUE;
    if (np > 0 && dz_per_ms == NULL) return MSIM_E_RANGE;
    if (np == 0) return 0;

    progs = (msim_progress *)c->pt_progs;
    dels = (const msim_delivery *)c->pt_dels;
    steps_buf = out_dz_steps != NULL
        ? out_dz_steps
        : (double *)malloc((size_t)np * sizeof(double));
    if (steps_buf == NULL) return MSIM_E_RANGE;
    for (i = 0; i < np; i++) {
        double rate = dz_per_ms[i];
        double zl = progs[i].z_left;
        double step, used, residual;
        if (zl <= 0.0 || rate <= 0.0) { steps_buf[i] = 0.0; continue; }
        step = dt_ms * rate;
        if (step >= zl) {
            used = zl;
            progs[i].z_left = 0.0;
        } else {
            residual = zl - step;
            if (residual <= 4.0 * msim_ulp(zl)) {
                used = zl;
                progs[i].z_left = 0.0;
            } else {
                used = step;
                progs[i].z_left = residual;
            }
        }
        steps_buf[i] = used;
    }
    for (i = 0; i < nd; i++) {
        double s = steps_buf[dels[i].flow];
        if (s > 0.0) c->pt_ctrs[dels[i].counter] += s * dels[i].work_per_z;
    }
    if (out_dz_steps == NULL) free(steps_buf);
    return 0;
}

/* Read back current prepared progress residuals and delivered counters. */
MSIM_EXPORT int32_t msim_read_term_state(
    void *ctx, double *out_z_left, uint32_t nprogs,
    double *out_counters, uint32_t ncounters)
{
    msim_ctx *c = (msim_ctx *)ctx;
    const msim_progress *progs;
    int32_t i;
    if (!ctx_ok(c)) return MSIM_E_STATE;
    if ((int32_t)nprogs != c->pt_nprogs || (int32_t)ncounters != c->pt_nctrs)
        return MSIM_E_RANGE;
    progs = (const msim_progress *)c->pt_progs;
    if (out_z_left != NULL) {
        for (i = 0; i < c->pt_nprogs; i++) out_z_left[i] = progs[i].z_left;
    }
    if (out_counters != NULL && c->pt_nctrs > 0) {
        memcpy(out_counters, c->pt_ctrs, (size_t)c->pt_nctrs * sizeof(double));
    }
    return 0;
}

#endif /* !msim_abi_version */

#endif /* MSIM_ENGINE_ACCEL_TERMS_H */
