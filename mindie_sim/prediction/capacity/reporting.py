"""Stage and rank capacity plans and allocation reports.

[Role] Project fixed memory, cache and live holders into capacity reports.
[Invariants] Keep the committed allocation view and physical rank limits distinct.
  Preserve the existing reporting fields and units.
[Timeouts & Locks] Bounded by ranks, stages and live allocation entries. No locks or I/O.
[Side-Effect Policy] Reads the current memory ledger to construct plans and reports.
"""
from __future__ import annotations

from mindie_sim.catalog.cache_pages import Allocation, AllocationPlan


class _MemoryReportingMixin:
    def _stage_dies(self, replica: int, stage: int):
        p = self.case.runtime.parallel
        if not 0 <= replica < p.dp or not 0 <= stage < p.pp:
            raise ValueError("memory_rank_plan_index_out_of_range")
        width = p.tp * p.pcp
        first = (replica * p.pp + stage) * width
        return self.case.machine.dies[first:first + width]

    def _rank_coord(self, replica: int, stage: int, local_rank: int):
        from mindie_sim.schema.common import ModelingError
        from mindie_sim.schema.parallel import ReplicaLocalRank

        placement = getattr(self, "_cached_placement", None)
        if placement is None:
            try:
                placement = self.case.machine.validate_placement(self.case.runtime.parallel)
            except ModelingError:
                placement = False
            self._cached_placement = placement
        if placement:
            try:
                return placement.coord_for(
                    int(replica), int(stage), ReplicaLocalRank(int(local_rank)),
                )
            except ModelingError:
                return None
        return None

    def _physical_cache_rank(self, replica: int, stage: int, local_rank: int) -> int:
        p = self.case.runtime.parallel
        coord = self._rank_coord(replica, stage, local_rank)
        eff_rep = coord.replica if coord is not None else int(replica)
        eff_local = int(coord.local_rank) if coord is not None else int(local_rank)
        if self.rank_token_owners is not None:
            return eff_local
        dp_width = max(1, int(getattr(p, "dcp_dp_width", 1) or 1))
        spans_dp = bool(getattr(p, "cp_spans_dp", False)) and dp_width > 1
        if spans_dp:
            return ((eff_rep % dp_width) * int(p.pp)) * self.width + eff_local
        if (
            getattr(p, "cp_plan", "") == "dsa_cp"
            and int(getattr(p, "cp", 1) or 1) > 1
            and int(getattr(p, "pcp", 1) or 1) == 1
            and coord is not None
        ):
            return int(coord.dcp_lane)
        return eff_local

    def _rank_allocations(
        self, replica: int, stage: int, *, tokens: int | None, include_cache: bool,
        rank: int | None = None,
    ) -> tuple[Allocation, ...]:
        if rank is None:
            best_rk = max(
                range(self.width),
                key=lambda rk: (
                    self.selected_rank[replica][stage][rk]
                    + self.shared_rank[replica][stage][rk]
                    + self.granted_rank[replica][stage][rk]
                    + self.reader_rank[replica][stage][rk]
                ),
            )
            selected = self.selected_rank[replica][stage][best_rk]
            shared = self.shared_rank[replica][stage][best_rk]
            granted = self.granted_rank[replica][stage][best_rk]
            readers = self.reader_rank[replica][stage][best_rk]
            weight_bytes = self.weights[stage]
            workspace_bytes = self.workspace[stage]
        else:
            rk_idx = int(rank)
            selected = self.selected_rank[replica][stage][rk_idx]
            shared = self.shared_rank[replica][stage][rk_idx]
            granted = self.granted_rank[replica][stage][rk_idx]
            readers = self.reader_rank[replica][stage][rk_idx]
            weight_bytes = self.rank_weights[replica][stage][rk_idx]
            workspace_bytes = self.rank_workspace[replica][stage][rk_idx]
        prep_persist = float(
            self._prepared_persistent_by_stage[stage]
            if stage < len(getattr(self, "_prepared_persistent_by_stage", ()))
            else 0.0
        )
        c8_static = float(
            self.c8_static_channel_scale_by_stage[stage]
            if stage < len(getattr(self, "c8_static_channel_scale_by_stage", ()))
            else 0.0
        )
        allocations = [
            Allocation("weights", weight_bytes, "weights",
                       note="persistent resident weight bytes"),
        ]
        if prep_persist > 0.0:
            allocations.append(Allocation(
                "prepared_weights", prep_persist, "prepared_weights",
                note="registered persistent prepared weight copies",
            ))
        if c8_static > 0.0:
            allocations.append(Allocation(
                "c8_static_channel_scale", c8_static, "cache",
                note="fixed per-die GQA C8 static channel scales",
            ))
        allocations.extend([
            Allocation("activation_workspace", workspace_bytes, "workspace",
                       note="simultaneous implementation workspace"),
            Allocation("temporary_dequant", self.temporary_dequant[stage], "temporary",
                       note="declared transient representation"),
            Allocation("selected_plan", selected,
                       "workspace", note="selected implementation allocation"),
            Allocation("shared_pages", shared,
                       "shared_page", note="physical prefix/page owner and refs"),
            Allocation("grants", granted, "grant",
                       note="short-lived batch grants"),
            Allocation("grant_readers", readers, "grant_reader",
                       note="last-use readers of released grants"),
            Allocation("reserve.workspace", self.policy.workspace_bytes, "reserve"),
            Allocation("reserve.graph_pool", self.policy.graph_pool_bytes, "reserve"),
            Allocation("reserve.communication", self.policy.communication_buffer_bytes,
                       "reserve"),
            Allocation("reserve.runtime", self.policy.runtime_bytes, "reserve"),
        ])
        if include_cache:
            span = self.case.runtime.max_model_len if tokens is None else tokens
            if rank is None:
                _gross, resident, staging = self._kv_residency(span)
                resident_value = resident[stage]
                staging_value = staging[stage]
            else:
                phys_rk = self._physical_cache_rank(replica, stage, int(rank))
                gross = self._rank_cache_bytes(span, padded=True, rank=phys_rk)
                layer_max = self._rank_max_layer_cache_bytes(span, phys_rk)
                resident_value = (
                    float(gross[stage]) if self.kvc_hbm_ratio == 1.0
                    else float(gross[stage]) * self.kvc_hbm_ratio)
                staging_value = (
                    0.0 if self.kvc_hbm_ratio == 1.0
                    else float(layer_max[stage]) * (1.0 - self.kvc_hbm_ratio))
            if self.kvc_hbm_ratio == 1.0:
                allocations.append(Allocation(
                    "cache_allocation", resident_value, "cache",
                    note="physical pages at declared token span"))
            else:
                allocations.append(Allocation(
                    "kv_hbm_resident", resident_value, "cache",
                    note="declared KVC HBM residency ratio"))
                allocations.append(Allocation(
                    "kv_hbm_staging", staging_value, "staging",
                    note="required staging for the offloaded KV tier"))
        return tuple(allocations)

    def rank_capacity_plan(
        self, replica: int, stage: int, *, tokens: int | None = None,
        include_cache: bool = True, rank: int | None = None,
    ) -> AllocationPlan:
        """Explicit allocation for one replica/stage, optionally one Rank.

        Without ``rank`` this remains the conservative stage view (minimum
        mounted capacity in the TP/CP group).  ``rank_capacity_plans`` exposes
        every physical Rank's own capacity and its own T05 owner rows so
        heterogeneous dies and uneven token partitions are not represented by
        a single minimum or a stage average.
        """
        dies = self._stage_dies(replica, stage)
        if rank is not None:
            if not 0 <= int(rank) < len(dies):
                raise ValueError("memory_rank_index_out_of_range")
            mounted = dies[int(rank)].hbm_bytes * self.case.runtime.gpu_memory_utilization
            basis = f"physical_rank_{int(rank)}_allocations"
        else:
            mounted = min(
                die.hbm_bytes * self.case.runtime.gpu_memory_utilization
                for die in dies)
            basis = "per_rank_stage_physical_allocations"
        return AllocationPlan(
            self._rank_allocations(replica, stage, tokens=tokens,
                                   include_cache=include_cache, rank=rank),
            capacity_bytes=mounted, basis=basis)

    def rank_capacity_plans(
        self, replica: int, stage: int, *, tokens: int | None = None,
        include_cache: bool = True,
    ) -> tuple[AllocationPlan, ...]:
        """One capacity plan per physical Rank/die in this replica/stage."""
        return tuple(
            self.rank_capacity_plan(
                replica, stage, tokens=tokens, include_cache=include_cache, rank=index)
            for index in range(len(self._stage_dies(replica, stage)))
        )

    def allocation_report(self, tokens: int | None = None) -> dict:
        """Public explainable allocation view for one token span."""
        case, p = self.case, self.case.runtime.parallel
        tokens = case.runtime.max_model_len if tokens is None else int(tokens)
        if tokens < 0:
            raise ValueError("memory_allocation_tokens_requires_nonnegative")
        cache = self.cache_allocation(tokens)
        gross_kv, resident_kv, staging_kv = self._kv_residency(tokens)
        hbm_live, _host_live = self._state_usage()
        width = p.tp * p.pcp
        mounted = [
            [
                min(d.hbm_bytes * case.runtime.gpu_memory_utilization
                    for d in self._stage_dies(replica, stage))
                for stage in range(p.pp)
            ]
            for replica in range(p.dp)
        ]
        rank_divergent = self.rank_token_owners is not None or int(getattr(p, "cp", 1) or 1) > 1
        persistent: list[list[float]] = []
        headroom: list[list[float]] = []
        for replica in range(p.dp):
            rep_pers: list[float] = []
            rep_head: list[float] = []
            for stage in range(p.pp):
                stage_dies = self._stage_dies(replica, stage)
                prep_persist = float(
                    self._prepared_persistent_by_stage[stage]
                    if stage < len(getattr(self, "_prepared_persistent_by_stage", ()))
                    else 0.0
                )
                c8_static = float(
                    self.c8_static_channel_scale_by_stage[stage]
                    if stage < len(getattr(self, "c8_static_channel_scale_by_stage", ()))
                    else 0.0
                )
                fixed_common = (
                    prep_persist + c8_static + self.policy.reserved
                    + self.temporary_dequant[stage]
                )
                die_residents: list[float] = []
                die_headrooms: list[float] = []
                for rk in range(self.width):
                    non_req = max(
                        0.0,
                        float(hbm_live[replica][stage][rk])
                        - float(self.live_rank[replica][stage][rk])
                        - float(self.staging_rank[replica][stage][rk]),
                    )
                    if rank_divergent:
                        phys_rk = self._physical_cache_rank(replica, stage, rk)
                        rk_gross = self._rank_cache_bytes(tokens, padded=True, rank=phys_rk)
                        rk_lmax = self._rank_max_layer_cache_bytes(tokens, phys_rk)
                        rk_res = (
                            float(rk_gross[stage]) if self.kvc_hbm_ratio == 1.0
                            else float(rk_gross[stage]) * self.kvc_hbm_ratio
                        )
                        rk_stg = (
                            0.0 if self.kvc_hbm_ratio == 1.0
                            else float(rk_lmax[stage]) * (1.0 - self.kvc_hbm_ratio)
                        )
                    else:
                        rk_res = float(resident_kv[stage])
                        rk_stg = float(staging_kv[stage])
                    fixed_rk = (
                        self.rank_weights[replica][stage][rk]
                        + self.rank_workspace[replica][stage][rk]
                        + fixed_common
                    )
                    used_rk = fixed_rk + rk_res + rk_stg + non_req
                    cap_rk = stage_dies[rk].hbm_bytes * case.runtime.gpu_memory_utilization
                    die_residents.append(used_rk)
                    die_headrooms.append(cap_rk - used_rk)
                rep_pers.append(max(die_residents, default=0.0))
                rep_head.append(min(die_headrooms, default=0.0))
            persistent.append(rep_pers)
            headroom.append(rep_head)
        plans = [
            [self.rank_capacity_plan(replica, stage, tokens=tokens).to_dict()
             for stage in range(p.pp)]
            for replica in range(p.dp)
        ]
        physical_plans = [
            [[plan.to_dict() for plan in self.rank_capacity_plans(
                replica, stage, tokens=tokens)]
             for stage in range(p.pp)]
            for replica in range(p.dp)
        ]
        return {
            "basis": "declared_physical_allocations_not_access_traffic",
            "sequence_tokens": tokens,
            "weights": {
                "total_bytes_per_stage_rank": list(self.weights),
                "payload_bytes_per_stage_rank": list(self.weights_payload),
                "quant_metadata_bytes_per_stage_rank": list(self.weights_quant_metadata),
                "gross_bytes_per_stage_rank": list(self.gross_weights),
                "offload_saved_bytes_per_stage_rank": list(self.expert_hbm_saved),
            },
            "reserve": {
                "workspace_bytes": self.policy.workspace_bytes,
                "graph_pool_bytes": self.policy.graph_pool_bytes,
                "communication_buffer_bytes": self.policy.communication_buffer_bytes,
                "runtime_bytes": self.policy.runtime_bytes,
                "total_bytes_per_rank": self.policy.reserved,
            },
            "workspace": {
                "base_bytes_per_stage_rank": list(self.workspace_base),
                "async_bytes_per_stage_rank": list(self.async_workspace),
                "expert_staging_bytes_per_stage_rank": list(self.workspace_expert_staging),
                "static_offload_bytes_per_stage_rank": list(self.workspace_static_offload),
                "concurrent_copies": self.concurrent_workspace_copies,
                "total_bytes_per_stage_rank": list(self.workspace),
            },
            "temporary_dequant": {
                "bytes_per_stage_rank": list(self.temporary_dequant),
                "basis": "explicitly declared transient representation",
            },
            "cache": cache,
            "grants_bytes_per_replica_stage": self.granted,
            "peak_combined_bytes_per_replica_stage": self.peak_combined,
            "budget_version": self.budget_version,
            "grant_bindings": [
                {"replica": replica, "name": name, **self._binding_report(binding)}
                for (replica, name), binding in sorted(self.grant_bindings.items())
            ],
            "onchip_effects": [
                {"replica": replica, "stage": stage, "name": name, **binding}
                for (replica, stage, name), binding in sorted(self.onchip_effects.items())
            ],
            "selected_plan_bytes_per_replica_stage": self.selected_reserved,
            "shared_page_bytes_per_replica_stage": self.shared_reserved,
            "kv_offload": {
                "kvc_hbm_ratio": self.kvc_hbm_ratio,
                "gross_kv_bytes_per_stage_rank": gross_kv,
                "hbm_resident_kv_bytes_per_stage_rank": resident_kv,
                "staging_bytes_per_stage_rank": staging_kv,
                "basis": "same_residency_and_staging_as_resize",
            },
            "mounted_capacity_bytes_per_replica_stage": mounted,
            "peak_resident_bytes_per_replica_stage": persistent,
            "headroom_bytes_per_replica_stage": headroom,
            "rank_plans": plans,
            "physical_rank_plans": physical_plans,
            "physical_rank_live_bytes_per_replica_stage": self.live_rank,
            "physical_rank_staging_bytes_per_replica_stage": self.staging_rank,
            "physical_rank_selected_bytes_per_replica_stage": self.selected_rank,
            "physical_rank_shared_bytes_per_replica_stage": self.shared_rank,
            "physical_rank_grant_bytes_per_replica_stage": self.granted_rank,
            "physical_rank_reader_bytes_per_replica_stage": self.reader_rank,
            "physical_rank_headroom_bytes_per_replica_stage": [
                [
                    [cap - float(used) for cap, used in zip(
                        self.rank_capacity[replica][stage],
                        hbm_live[replica][stage])]
                    for stage in range(p.pp)
                ]
                for replica in range(p.dp)
            ],
            "physical_rank_cache_allocation": (
                [list(self.rank_cache_allocation(tokens))]
                if self.rank_token_owners is not None else []
            ),
            "rank_token_owners": list(self.rank_token_owners or ()),
            "fixed_plan": True,
            "fallback": "none",
        }

    def _max_model_len_report(self) -> dict:
        max_tokens = self.case.runtime.max_model_len
        cache_allocation = self.cache_allocation(max_tokens, include_components=False)
        gross_kv, resident_kv, staging_kv = self._kv_residency(max_tokens)
        section = {
            "cache_allocation_at_max_model_len": cache_allocation,
            "kv_offload_at_max_model_len": {
                "kvc_hbm_ratio": self.kvc_hbm_ratio,
                "gross_kv_bytes_per_stage_rank": gross_kv,
                "hbm_resident_kv_bytes_per_stage_rank": resident_kv,
                "staging_bytes_per_stage_rank": staging_kv,
                "basis": "same_residency_and_staging_as_resize",
            },
            "allocation_totals_at_max_model_len": {
                "weights_bytes_per_stage_rank": list(self.weights),
                "reserve_bytes_per_rank": self.policy.reserved,
                "workspace_bytes_per_stage_rank": list(self.workspace),
                "temporary_dequant_bytes_per_stage_rank": list(self.temporary_dequant),
                "cache_gross_bytes_per_stage_rank": gross_kv,
                "cache_hbm_resident_bytes_per_stage_rank": resident_kv,
                "cache_staging_bytes_per_stage_rank": staging_kv,
                "cache_unclassified_reserve_bytes_per_stage_rank":
                    cache_allocation["allocated"]["reserve_bytes_per_stage_rank"],
                "cache_allocated_bytes_per_stage_rank": resident_kv,
                "cache_valid_bytes_per_stage_rank":
                    cache_allocation["valid"]["total_bytes_per_stage_rank"],
                "resident_bytes_per_stage_rank": [
                    weight
                    + (float(self._prepared_persistent_by_stage[idx])
                       if idx < len(getattr(self, "_prepared_persistent_by_stage", ()))
                       else 0.0)
                    + (float(self.c8_static_channel_scale_by_stage[idx])
                       if idx < len(getattr(self, "c8_static_channel_scale_by_stage", ()))
                       else 0.0)
                    + self.policy.reserved + workspace + temporary + resident + staging
                    for idx, (weight, workspace, temporary, resident, staging) in enumerate(zip(
                        self.weights, self.workspace, self.temporary_dequant,
                        resident_kv, staging_kv))
                ],
            },
        }
        engram_host = list(getattr(self, "engram_host_dram_bytes", ()))
        if any(value > 0.0 for value in engram_host):
            section["engram_host_dram_bytes_per_stage_rank"] = engram_host
        if any(value > 0.0 for value in self.offload.values()):
            section["offload"] = {
                "expert_ddr_ratio": self.offload["expert_ddr_ratio"],
                "kvc_ddr_ratio": self.offload["kvc_ddr_ratio"],
                "kvc_ssd_ratio": self.offload["kvc_ssd_ratio"],
                "kvc_hbm_ratio": self.kvc_hbm_ratio,
                "gross_weights_bytes_per_stage_rank": self.gross_weights,
                "routed_expert_bytes_per_stage_rank": self.routed_expert_bytes,
                "expert_hbm_saved_bytes_per_stage_rank": self.expert_hbm_saved,
            }
        if self.sparse_offload:
            p = self.case.runtime.parallel
            block_size = int(self.sparse_offload["block_size"])
            options = self.sparse_offload.get("options") or {}
            keep_device_kv = bool(options.get("keep_device_kv_cache", False))
            if keep_device_kv or self.pages.groups:
                _gross_blk, res_blk, _stg_blk = self._kv_residency(block_size)
                stage_per_block_list = [float(v) for v in res_blk]
            else:
                idx_row_bytes = int(self.sparse_offload["index_row_bytes"])
                kv_row_bytes = int(self.sparse_offload.get("kv_row_bytes", 0))
                row_by_layer = self.sparse_offload.get("kv_row_bytes_by_layer") or {}
                offload_seq = tuple(int(lid) for lid in self.sparse_offload.get("offload_layers", ()))
                layerwise = bool(options.get("layerwise_kv_offload", False))
                num_shared = max(0, int(options.get("num_shared_layers", 0) or 0))
                if layerwise:
                    raw_stage_groups = self.sparse_offload.get("offload_layers_by_stage")
                    if raw_stage_groups:
                        stage_groups = tuple(
                            tuple(int(lid) for lid in stage_lids)
                            for stage_lids in raw_stage_groups
                        )
                    else:
                        grouped: list[list[int]] = [[] for _ in range(p.pp)]
                        for lid in offload_seq:
                            st = (
                                self.layer_stage[lid]
                                if 0 <= lid < len(self.layer_stage) else p.pp - 1
                            )
                            grouped[st].append(lid)
                        stage_groups = tuple(tuple(stage_lids) for stage_lids in grouped)
                    zero_layers = {
                        lid
                        for stage_lids in stage_groups
                        for lid in sorted(
                            stage_lids,
                            key=lambda item: (
                                -int(row_by_layer.get(
                                    item, row_by_layer.get(str(item), kv_row_bytes),
                                )),
                                item,
                            ),
                        )[num_shared:]
                    }
                else:
                    zero_layers = set(offload_seq)
                stage_per_block_list = [0.0] * p.pp
                for lid in self.sparse_offload["full_indexer_layers"]:
                    st = (
                        self.layer_stage[int(lid)]
                        if 0 <= int(lid) < len(self.layer_stage) else p.pp - 1
                    )
                    stage_per_block_list[st] += float(block_size * idx_row_bytes)
                if kv_row_bytes > 0 or row_by_layer:
                    for plan in self.case.model.layer_plans:
                        if int(plan.index) in zero_layers:
                            continue
                        if plan.attention == "mla" and plan.cache_kind == "dense":
                            st = (
                                self.layer_stage[int(plan.index)]
                                if 0 <= int(plan.index) < len(self.layer_stage) else p.pp - 1
                            )
                            row_b = int(row_by_layer.get(int(plan.index), row_by_layer.get(str(plan.index), kv_row_bytes)))
                            stage_per_block_list[st] += float(block_size * row_b)
                stage_per_block_list = [v * self.kvc_hbm_ratio for v in stage_per_block_list]
            npu_limits = [
                int(max(0.0, min(self.capacity[rep][st] for rep in range(p.dp))) // per_blk)
                for st, per_blk in enumerate(stage_per_block_list)
                if per_blk > 0.0
            ]
            npu_limit_blocks = min(npu_limits) if npu_limits else 0
            dram_limit_blocks = int(self.sparse_offload["dram_limit_blocks"])
            workload_limit_blocks = int(self.sparse_offload["workload_limit_blocks"])
            limits = {
                "npu_hbm": npu_limit_blocks,
                "host_dram": dram_limit_blocks,
                "workload": workload_limit_blocks,
            }
            limiting_factor = min(limits, key=limits.__getitem__)
            section["sparse_kv_offload"] = {
                **self.sparse_offload,
                "npu_limit_blocks": npu_limit_blocks,
                "dram_limit_blocks": dram_limit_blocks,
                "workload_limit_blocks": workload_limit_blocks,
                "effective_blocks": limits[limiting_factor],
                "limiting_factor": limiting_factor,
                "host_dram_capacity_bytes_per_replica": self.host_dram_capacity,
                "peak_host_dram_bytes_per_replica": self.host_dram_peak,
                "live_host_dram_bytes_per_replica": self.host_dram_live,
            }
        return section

    def to_dict(self) -> dict:
        hbm_live, _host_live = self._state_usage()
        result = {
            "status": ("feasible" if all(value >= 0 for row in self.capacity for value in row)
                       else "oom"),
            "basis": "declared_geometry_and_live_allocations",
            "allocation_schema": "mindie-sim.memory-allocation/1",
            "weights_bytes_per_stage_rank": self.weights,
            "weights_payload_bytes_per_stage_rank": self.weights_payload,
            "weights_quant_metadata_bytes_per_stage_rank": self.weights_quant_metadata,
            "reserve_bytes_per_rank": self.policy.reserved,
            "reserve_breakdown": {
                "workspace_bytes": self.policy.workspace_bytes,
                "graph_pool_bytes": self.policy.graph_pool_bytes,
                "communication_buffer_bytes": self.policy.communication_buffer_bytes,
                "runtime_bytes": self.policy.runtime_bytes,
            },
            "activation_workspace_bytes_per_stage_rank": self.workspace,
            "workspace_breakdown": {
                "base_bytes_per_stage_rank": self.workspace_base,
                "async_bytes_per_stage_rank": self.async_workspace,
                "expert_staging_bytes_per_stage_rank": self.workspace_expert_staging,
                "static_offload_bytes_per_stage_rank": self.workspace_static_offload,
                "concurrent_copies": self.concurrent_workspace_copies,
            },
            "temporary_dequant_bytes_per_stage_rank": self.temporary_dequant,
            "async_scheduling": bool(self.case.runtime.async_scheduling),
            "async_workspace_overhead_bytes_per_stage_rank": self.async_workspace,
            "async_delayed_release_steps": self.async_delayed_release_steps,
            "async_in_flight_lookahead_steps": self.async_in_flight_lookahead_steps,
            "peak_delayed_release_cache_bytes_per_replica_stage": self.peak_delayed_release,
            "kv_capacity_bytes_per_replica_stage": self.capacity,
            "peak_cache_bytes_per_replica_stage": self.peak,
            "peak_kv_hbm_resident_bytes_per_replica_stage": self.peak_kv_resident,
            "peak_kv_staging_bytes_per_replica_stage": self.peak_kv_staging,
            "peak_combined_bytes_per_replica_stage": self.peak_combined,
            "budget_version": self.budget_version,
            "grant_bindings": [
                {"replica": replica, "name": name, **self._binding_report(binding)}
                for (replica, name), binding in sorted(self.grant_bindings.items())
            ],
            "onchip_effects": [
                {"replica": replica, "stage": stage, "name": name, **binding}
                for (replica, stage, name), binding in sorted(self.onchip_effects.items())
            ],
            "physical_rank_capacity_bytes_per_replica_stage": [
                [[die.hbm_bytes * self.case.runtime.gpu_memory_utilization
                  for die in self._stage_dies(replica, stage)]
                 for stage in range(self.case.runtime.parallel.pp)]
                for replica in range(self.case.runtime.parallel.dp)
            ],
            "physical_rank_live_bytes_per_replica_stage": self.live_rank,
            "physical_rank_staging_bytes_per_replica_stage": self.staging_rank,
            "physical_rank_selected_bytes_per_replica_stage": self.selected_rank,
            "physical_rank_shared_bytes_per_replica_stage": self.shared_rank,
            "physical_rank_grant_bytes_per_replica_stage": self.granted_rank,
            "physical_rank_reader_bytes_per_replica_stage": self.reader_rank,
            "physical_rank_headroom_bytes_per_replica_stage": [
                [
                    [cap - float(used) for cap, used in zip(
                        self.rank_capacity[replica][stage],
                        hbm_live[replica][stage])]
                    for stage in range(self.case.runtime.parallel.pp)
                ]
                for replica in range(self.case.runtime.parallel.dp)
            ],
            "rank_token_owners": list(self.rank_token_owners or ()),
            "live_cache_bytes_per_replica_stage": self.live,
            "live_grants_bytes_per_replica_stage": self.granted,
            "selected_plan_bytes_per_replica_stage": self.selected_reserved,
            "shared_page_bytes_per_replica_stage": self.shared_reserved,
            "shared_page_owners": [
                {"replica": key[0], "owner": key[1], **record}
                for key, record in sorted(
                    self.shared_pages.items(),
                    key=lambda item: (item[0][0], item[0][1], len(item[0]), item[0]),
                )
            ],
            "peak_resident_bytes_per_replica_stage": [
                [weight
                 + (float(self._prepared_persistent_by_stage[index])
                    if index < len(getattr(self, "_prepared_persistent_by_stage", ()))
                    else 0.0)
                 + (float(self.c8_static_channel_scale_by_stage[index])
                    if index < len(getattr(self, "c8_static_channel_scale_by_stage", ()))
                    else 0.0)
                 + self.policy.reserved + workspace + temporary
                 + self.peak_combined[replica][index]
                 for index, (weight, workspace, temporary) in enumerate(zip(
                     self.weights, self.workspace, self.temporary_dequant))]
                for replica in range(self.case.runtime.parallel.dp)
            ],
            "layer_partition": self.layers,
            "workspace_basis": "simultaneous_layer_tensors_plus_explicit_reserves",
            "fixed_plan": True,
            "fallback": "none",
        }
        result.update(self._max_model_len_report())
        return result
