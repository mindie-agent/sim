"""Typed Simplified and GQA issue cursor for one core.

[Role]
  When ``issue_kind`` is explicit, one core keeps a single task cursor across
  its M ranges and S2 tiles.  Simplified issues QK(t), Vec1(t-1), PV(t-2),
  Vec2(t-3) from four ``RunInfo`` descriptors.  GQA issues QK(t), Vec1(t),
  PV(t-2), Vec2(t-2) from three descriptors.  Buffer names follow the operand
  policy (Q double or single, K/V double or single, P three), not the
  descriptor count.

[Invariants]
  ``issue_kind is None`` is not produced here and must not be invented by the
  caller.  Descriptor windows are not K/V slot counts.  No duration, recipe
  floor, or cache residual changes the window.  Owner ranges are an input;
  this module does not assign cores.

[Timeouts & Locks]
  Only the descriptor window is submitted ahead.  Finished steps drop their
  tile handles.  O(window) live stages per core.
"""
from __future__ import annotations

from mindie_sim.schema.common import ModelingError

_L1_POOL = 524288
_L0C_POOL = 262144
_BF16 = 2
_FP32 = 4
_WINDOWS = {"simplified": 4, "gqa": 3}
_LAGS = {"simplified": 3, "gqa": 2}
C2_GM_WORKSPACE_BYTES = 12_582_912


def source_issue_operand(reconstruct: bool | None) -> str | None:
    """GQA true loads the real S2 tile; Simplified false clips to the causal end.

    ``None`` leaves the original selector unknown and does not pick a branch.
    """
    if reconstruct is None:
        return None
    return "real" if reconstruct else "causal"


def source_issue_kind(reconstruct: bool | None, head_dim: int) -> str | None:
    """Map an explicit flag to a kernel.  Absent and unregistered stay unset.

    D256 ``reconstruct=true`` is not a registered noquant candidate, so it
    does not become GQA and does not become Simplified either.
    """
    if reconstruct is None:
        return None
    if reconstruct:
        if head_dim > 128:
            return None
        return "gqa"
    return "simplified"


def source_c2_path(head_dim: int, query_tile: int, key_tile: int,
                   reconstruct: bool | None) -> dict | None:
    """Registered noquant C2 placement.

    ``None`` does not choose reconstruct or a D256 template.  The only GM
    candidate is Simplified D256 128x128, and its reused workspace is not HBM.
    """
    if reconstruct is None or (head_dim > 128 and reconstruct):
        return None
    if (not reconstruct and head_dim > 128 and query_tile == 128
            and key_tile == 128):
        return {"path": "gm_workspace", "workspace_bytes": C2_GM_WORKSPACE_BYTES,
                "hbm_bytes": 0}
    return {"path": "ub", "workspace_bytes": 0, "hbm_bytes": 0}


# Legal (IsFd, EnableS1OutSplit) rows shared by the filtered configs.
# (true, true) is not in the BF16 same-dtype window.
_FD_S1_OUT = ((False, False), (True, False), (False, True))
_SIMPLIFIED_STRUCT = "FlashAttentionScoreSimplifiedTilingData"
_GQA_STRUCT = "FusedInferAttentionScoreTilingData"
_BF16_NAMES = {"bf16", "bfloat16"}


def _candidate(config: int, query_tile: int, key_tile: int, reconstruct: bool,
               struct: str, c2: str) -> dict:
    return {
        "config": config,
        "query_tile": query_tile,
        "key_tile": key_tile,
        "reconstruct": reconstruct,
        "struct": struct,
        "c2": c2,
        "workspace_bytes": C2_GM_WORKSPACE_BYTES if c2 == "gm_workspace" else 0,
        "hbm_bytes": 0,
        "fd_s1_out": _FD_S1_OUT,
    }


# Original ABI filter: BF16 in and out, TND, NoQuant, mask, no rope, no PA,
# no prefix, D=DV.  Built from the one r35 header audit.  `selected` stays
# null; head dimension does not choose a row.
_CANDIDATES = {
    128: (
        _candidate(1, 64, 256, False, _SIMPLIFIED_STRUCT, "ub"),
        _candidate(1, 64, 256, True, _GQA_STRUCT, "ub"),
        _candidate(3, 128, 128, False, _SIMPLIFIED_STRUCT, "ub"),
        _candidate(3, 128, 128, True, _GQA_STRUCT, "ub"),
    ),
    256: (
        _candidate(6, 128, 128, False, _SIMPLIFIED_STRUCT, "gm_workspace"),
        _candidate(18, 64, 256, False, _SIMPLIFIED_STRUCT, "ub"),
    ),
}


def _unidentified(reason: str) -> dict:
    return {"selected": None, "class": None, "reconstruct": (),
            "candidates": (), "reason": reason}


def _canon_dtype(value) -> str:
    if not isinstance(value, str):
        return ""
    key = value.strip().lower()
    return "bf16" if key in _BF16_NAMES else key


def source_compatible_alternatives(head_dim: int, query_lengths, *,
                                   query_dtype: str = "bf16",
                                   kv_dtype: str = "bf16",
                                   out_dtype: str = "bf16",
                                   layout: str = "TND",
                                   quant: str = "NoQuant",
                                   mask: bool = True,
                                   rope: bool = False,
                                   kv_paged: bool = False,
                                   kv_prefix: bool = False) -> dict:
    """Registered prefill family for the original operator fields.

    The return value lists every config that survives that filter.  It does
    not pick Simplified versus GQA, GM versus UB, or a tile.  q_len 1 is
    IncreFlashAttention and yields an empty set.
    """
    if isinstance(head_dim, bool) or not isinstance(head_dim, int):
        raise ModelingError("attention_prefill_head_dim_must_be_int")
    lengths = tuple(query_lengths)
    for length in lengths:
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ModelingError(
                "attention_prefill_query_len_must_be_nonnegative_int")
    layout_name = layout.strip().upper() if isinstance(layout, str) else ""
    quant_name = quant.strip().lower() if isinstance(quant, str) else ""
    original = (
        _canon_dtype(query_dtype) == "bf16"
        and _canon_dtype(kv_dtype) == "bf16"
        and _canon_dtype(out_dtype) == "bf16"
        and layout_name == "TND"
        and quant_name in {"noquant", "noquantmode"}
        and mask is True and rope is False
        and kv_paged is False and kv_prefix is False)
    if not original:
        return _unidentified("outside_original_abi")
    if not lengths:
        return _unidentified("no_query")
    if not any(length > 1 for length in lengths):
        reason = ("q_len_1_incre_flash_attention"
                  if all(length == 1 for length in lengths)
                  else "no_prefill_query")
        return _unidentified(reason)
    candidates = _CANDIDATES.get(head_dim)
    if candidates is None:
        return _unidentified("head_dim_unregistered")
    reconstruct = tuple(sorted({row["reconstruct"] for row in candidates}))
    shared = reconstruct == (False,)
    return {
        "selected": None,
        "class": "simplified" if shared else "mixed",
        "reconstruct": reconstruct,
        "candidates": candidates,
        "reason": "topology_open" if shared else "reconstruct_mixed",
    }


def l1_operand_capacity(s1: int, s2: int, d: int, dv: int | None = None) -> dict:
    """BF16 L1 reservation from QL1 / KVL1 / the 3-buffer P policy.

    Counts are the static buffers, not in-flight descriptors.  The four
    noquant templates below all fit in the 524288 B pool.  This does not
    choose a pipeline depth.
    """
    if min(s1, s2, d) < 1:
        raise ModelingError("attention_prefill_l1_shape_must_be_positive")
    dv = d if dv is None else dv
    q_count = 1 if d > 256 else 2
    kv_count = 1 if s2 == 256 and d > 128 else 2
    q, k, v = s1 * d * _BF16, d * s2 * _BF16, dv * s2 * _BF16
    p = s1 * s2 * _BF16
    total = q_count * q + kv_count * (k + v) + 3 * p
    return {
        "q_bytes": q, "q_count": q_count,
        "k_bytes": k, "k_count": kv_count,
        "v_bytes": v, "v_count": kv_count,
        "p_bytes": p, "p_count": 3,
        "total": total, "l1_pool": _L1_POOL, "fits": total <= _L1_POOL,
    }


def l0c_policy(s1: int, s2: int, dv: int) -> dict:
    """4x64KiB when both products fit in L0C/4, otherwise DB 2x128KiB."""
    quarter = _L0C_POOL // 4
    four = s1 * s2 * _FP32 <= quarter and s1 * dv * _FP32 <= quarter
    count = 4 if four else 2
    piece = quarter if four else _L0C_POOL // 2
    return {"count": count, "piece": piece, "pool": _L0C_POOL,
            "total": count * piece}


def owner_segments(spec, ranges, task_at) -> list:
    """Cube segments for one core, in owner-range order.  No new core search."""
    segments = []
    for ti, key_start, key_end in ranges:
        count = key_end - key_start
        if count <= 0:
            continue
        base = task_at(spec, ti)
        segments.append({
            "rows": base.rows,
            "horizon": base.horizon,
            "key_start": key_start,
            "key_tiles": count,
            "partial": key_start > 0 or key_end < base.key_tiles,
        })
    return segments


def bmm_call(s1: int, s2: int, d: int, dv: int) -> dict:
    """BF16, not fp8, not DN, d<=256.  A label only; it does not scale time.

    ``s1`` is accepted so the template identity stays with the cube block.
    The calls themselves key off ``d``, ``dv`` and ``s2``.
    """
    del s1
    if d > 256:
        raise ModelingError("attention_prefill_bmm_label_is_bf16_d_le_256")
    if d > 128:
        if s2 == 256:
            qk = "IterateBmm1NdL0Split MatmulN<64,128,256> singleK=d"
        else:
            qk = "IterateBmm1NdL0Split MatmulK<128,128,128> singleK=d"
    else:
        qk = "IterateBmm1Nd MatmulBase<128,128,d>"
    if dv > 128:
        pv = "MatmulN singleN=dV baseN=128"
    elif s2 == 128:
        pv = "MatmulFull<128,dV,128>"
    else:
        pv = "MatmulBase<128,dV,128>"
    return {"qk": qk, "pv": pv}


class IssueCursor:
    """Bounded cross-M issue cursor.  One instance is one core."""

    def __init__(self, context, spec, segments, core: int):
        kind = spec.issue_kind
        if kind not in _WINDOWS:
            raise ModelingError(f"attention_prefill_issue_kind_unset:{kind}")
        self.context = context
        self.spec = spec
        self.core = core
        self.kind = kind
        self.window = _WINDOWS[kind]
        self.lag = _LAGS[kind]
        policy = l1_operand_capacity(max(1, spec.query_tile), spec.key_tile,
                                     spec.head_dim)
        self.k_count = policy["k_count"]
        self.v_count = policy["v_count"]
        self.l0c = l0c_policy(max(1, spec.query_tile), spec.key_tile,
                              spec.head_dim)
        self.tiles = []
        self.segments = []
        for segment in segments:
            count = int(segment["key_tiles"])
            if count < 0:
                raise ModelingError("attention_prefill_negative_key_tiles")
            first = len(self.tiles)
            for local in range(count):
                self.tiles.append((len(self.segments), local))
            self.segments.append({
                "rows": int(segment["rows"]),
                "horizon": int(segment["horizon"]),
                "key_start": int(segment["key_start"]),
                "partial": bool(segment["partial"]),
                "first": first,
                "last": len(self.tiles) - 1,
            })
        self.n = len(self.tiles)

    def start(self, engine, name, ready, *, owner=""):
        from mindie_sim.prediction.attention_prefill import (
            _allocation, _operand_role, _placement, _read_role, _write_role,
        )

        self.engine = engine
        self.name = name
        self.owner = owner
        self.ready = ready
        self._allocation = _allocation
        self._placement = _placement
        self._read = _read_role(self.context)
        self._write = _write_role(self.context)
        self._operand = _operand_role(self.context)
        self.events = {}
        self.next_step = 0
        self.inflight = 0
        self.finished = 0
        self.total_steps = self.n + self.lag if self.n else 0
        self.last_cube = ready
        self.last_vector = ready
        self.last_mte = ready
        self.q_hold = {}
        self.q_slot_of = {}
        self.q_free = [ready, ready]
        self.k_free = [ready] * self.k_count
        self.v_free = [ready] * self.v_count
        complete = engine.future(name + ":cursor_complete")
        self.complete = complete
        if not self.total_steps:
            engine.resolve(complete)
        else:
            engine.submit(name + ":begin", after=(ready,),
                          on_complete=self._begin, owner=owner)
        return engine.submit(name + ":complete", after=(complete,), owner=owner)

    def _begin(self, _event):
        self._issue()

    def _retire(self, _event):
        self.inflight -= 1
        self.finished += 1
        if self.finished == self.total_steps:
            self.events.clear()
            self.engine.resolve(self.complete)
            return
        self._issue()

    def _issue(self):
        while self.next_step < self.total_steps and self.inflight < self.window:
            step = self.next_step
            events = self._submit_step(step)
            self.next_step += 1
            self.inflight += 1
            self.engine.submit(
                f"{self.name}:step{step}:retire", after=tuple(events),
                on_complete=self._retire, owner=self.owner)
            stale = step - (self.window + self.lag)
            if stale >= 0:
                self.events.pop(stale, None)

    def _submit_step(self, step: int):
        posted = []
        if self.kind == "simplified":
            posted += self._qk(step, step)
            posted += self._vec1(step, step - 1)
            posted += self._pv(step, step - 2)
            posted += self._vec2(step, step - 3)
        else:
            posted += self._qk(step, step)
            posted += self._vec1(step, step)
            posted += self._pv(step, step - 2)
            posted += self._vec2(step, step - 2)
        if not posted:
            posted.append(self.engine.submit(
                f"{self.name}:step{step}:empty", after=(self.ready,),
                owner=self.owner))
        return posted

    def _tile(self, index: int):
        if index < 0 or index >= self.n:
            return None
        seg_i, local = self.tiles[index]
        seg = self.segments[seg_i]
        start = (seg["key_start"] + local) * self.spec.key_tile
        valid = min(self.spec.key_tile, seg["horizon"] - start)
        if valid < 0:
            valid = 0
        return seg_i, seg, local, valid

    def _qk(self, step: int, tile: int):
        found = self._tile(tile)
        if found is None:
            return []
        seg_i, seg, local, valid = found
        rows, dim = seg["rows"], self.spec.head_dim
        q_bytes = rows * dim * self.spec.query_bytes
        kv = valid * dim * self.spec.kv_bytes
        rect = rows * valid
        posted = []
        if local == 0:
            q_slot = seg_i % 2
            q_name = f"q{q_slot}"
            acquire = self._bufs(
                ("l1", q_name, q_bytes),
                ("l0a", q_name, min(q_bytes, 32 * 1024)))
            q_event = self._post(
                f"m{seg_i}:q_load", step, tile, "q_load",
                {"hbm": q_bytes, self._read: q_bytes},
                after=(self.q_free[q_slot],), acquire=acquire,
                roles=(self._read, self._operand))
            self.q_hold[seg_i] = acquire
            self.q_slot_of[seg_i] = q_slot
            posted.append(q_event)
        else:
            q_event = None
        k_slot = tile % self.k_count
        k_acquire = self._bufs(
            ("l1", f"k{k_slot}", kv),
            ("l0b", f"k{k_slot}", min(kv, 32 * 1024)))
        k_after = (self.k_free[k_slot], self.last_mte, self.last_cube)
        k_event = self._post(
            f"t{tile}:k_load", step, tile, "k_load",
            {"hbm": kv, self._read: kv,
             **({"mte1": kv} if self._operand else {})},
            after=k_after, acquire=k_acquire,
            roles=(self._read, self._operand),
            extra={"k_slot": k_slot, "k_count": self.k_count})
        self.last_mte = k_event
        posted.append(k_event)
        qk_after = [k_event, self.last_cube]
        if q_event is not None:
            qk_after.append(q_event)
        free = list(k_acquire)
        if tile == seg["last"]:
            free.extend(self.q_hold.get(seg_i, ()))
        l0c_slot = tile % self.l0c["count"]
        score_buf = self._bufs(("l0c", f"s{l0c_slot}", rows * valid * _FP32))
        # IterateBmm1 fixpipe releases L0C before the call returns.
        free.extend(score_buf)
        qk = self._post(
            f"t{tile}:qk", step, tile, "qk",
            {"mte1": q_bytes} if self._operand else {},
            matrix={self.spec.query_dtype: 2 * dim * rect},
            after=tuple(qk_after), acquire=tuple(score_buf), free=tuple(free),
            roles=("cube", self._operand))
        self.last_cube = qk
        self.k_free[k_slot] = qk
        if tile == seg["last"]:
            self.q_free[self.q_slot_of[seg_i]] = qk
        self.events.setdefault(tile, {})["qk"] = qk
        posted.append(qk)
        return posted

    def _vec1(self, step: int, tile: int):
        found = self._tile(tile)
        if found is None or "qk" not in self.events.get(tile, {}):
            return []
        _seg_i, seg, _local, valid = found
        rows = seg["rows"]
        rect = rows * valid
        p_slot = tile % 3
        p_bytes = rows * valid * self.spec.query_bytes
        p_buf = self._bufs(("l1", f"p{p_slot}", p_bytes))
        outer = min(rows, self.spec.softmax_outer)
        prob = self._bufs(("ub", f"p{p_slot}", outer * valid * _FP32))
        vec = self._post(
            f"t{tile}:vec1", step, tile, "vec1",
            {"vector": 7 * rect + rows},
            after=(self.events[tile]["qk"], self.last_vector),
            acquire=p_buf + prob, roles=("vector",))
        self.last_vector = vec
        self.events[tile]["vec1"] = vec
        self.events[tile]["p"] = p_buf
        self.events[tile]["prob"] = prob
        return [vec]

    def _pv(self, step: int, tile: int):
        found = self._tile(tile)
        if found is None or "vec1" not in self.events.get(tile, {}):
            return []
        _seg_i, seg, _local, valid = found
        rows, dim = seg["rows"], self.spec.head_dim
        kv = valid * dim * self.spec.kv_bytes
        rect = rows * valid
        v_slot = tile % self.v_count
        v_acquire = self._bufs(
            ("l1", f"v{v_slot}", kv),
            ("l0b", f"v{v_slot}", min(kv, 32 * 1024)))
        v_load = self._post(
            f"t{tile}:v_load", step, tile, "v_load",
            {"hbm": kv, self._read: kv,
             **({"mte1": kv} if self._operand else {})},
            after=(self.events[tile]["vec1"], self.v_free[v_slot],
                   self.last_mte, self.last_cube),
            acquire=v_acquire, roles=(self._read, self._operand),
            extra={"v_slot": v_slot, "v_count": self.v_count})
        self.last_mte = v_load
        p_mte = rect * self.spec.query_bytes
        p_op = self._post(
            f"t{tile}:p_operand", step, tile, "p_operand",
            {"mte1": p_mte} if self._operand else {},
            after=(self.events[tile]["vec1"],),
            roles=(self._operand,))
        l0c_slot = tile % self.l0c["count"]
        pv_buf = self._bufs(("l0c", f"c{l0c_slot}", rows * dim * _FP32))
        pv = self._post(
            f"t{tile}:pv", step, tile, "pv", {},
            matrix={self.spec.query_dtype: 2 * dim * rect},
            after=(v_load, p_op, self.last_cube),
            acquire=pv_buf,
            free=tuple(v_acquire) + tuple(pv_buf)
            + tuple(self.events[tile].get("p", ())),
            roles=("cube",))
        self.last_cube = pv
        self.v_free[v_slot] = pv
        self.events[tile]["pv"] = pv
        self.events[tile]["pv_buf"] = pv_buf
        return [v_load, p_op, pv]

    def _vec2(self, step: int, tile: int):
        found = self._tile(tile)
        if found is None or "pv" not in self.events.get(tile, {}):
            return []
        seg_i, seg, _local, _valid = found
        rows, dim = seg["rows"], self.spec.head_dim
        vec = self._post(
            f"t{tile}:vec2", step, tile, "vec2",
            {"vector": rows * (3 * dim + 9)},
            after=(self.events[tile]["pv"], self.last_vector),
            free=tuple(self.events[tile].get("prob", ())),
            roles=("vector",))
        self.last_vector = vec
        self.events[tile]["vec2"] = vec
        posted = [vec]
        if tile == seg["last"]:
            posted.append(self._store(step, tile, seg))
        return posted

    def _store(self, step: int, tile: int, seg: dict):
        from mindie_sim.prediction.attention_prefill import _FP32_BYTES

        rows, dim = seg["rows"], self.spec.head_dim
        if seg["partial"]:
            part = rows * dim * _FP32_BYTES
            lse = rows * 2 * _FP32_BYTES
            resources = {"hbm": part + lse, "mte3": part + lse,
                         "vector": rows * (dim + 2)}
        else:
            out = rows * dim * self.spec.query_bytes
            resources = {"hbm": out, "mte3": out, "vector": rows * (dim + 1)}
        return self._post(
            f"t{tile}:store", step, tile, "store", resources,
            after=(self.events[tile]["vec2"],),
            roles=(self._write, "vector"))

    def _bufs(self, *items):
        out = []
        for pool, slot, size in items:
            # One engine allocation name is one live buffer.  L1 and L0/UB
            # copies of the same operand need distinct names.
            alloc = self._allocation(self.context, self.core, pool,
                                     f"{self.name}:{pool}:{slot}", size)
            if alloc is not None:
                out.append(alloc)
        return tuple(out)

    def _post(self, suffix, step, tile, stage, resources, *, matrix=None,
              after=(), acquire=(), free=(), roles=(), extra=None):
        from mindie_sim.prediction.mechanism import Work

        present = tuple(role for role in roles if role)
        placement = self._placement(self.context, self.core, *present)
        work = self.context.bind(Work(resources=resources, matrix=matrix or {}))
        meta = {"step": step, "tile": tile, "stage": stage,
                "kind": self.kind, "window": self.window}
        if extra:
            meta.update(extra)
        event = self.engine.submit(
            f"{self.name}:{suffix}", work=work, after=tuple(after),
            acquire=tuple(acquire), free=tuple(free), placement=placement,
            metadata=meta, owner=self.owner)
        return event
