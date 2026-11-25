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

    def _rank_allocations(
        self, replica: int, stage: int, *, tokens: int | None, include_cache: bool,
        rank: int | None = None,
    ) -> tuple[Allocation, ...]:
        if rank is None:
            selected = self.selected_reserved[replica][stage]
            shared = self.shared_reserved[replica][stage]
            granted = self.granted[replica][stage]
            readers = max(self.reader_rank[replica][stage])
        else:
            selected = self.selected_rank[replica][stage][int(rank)]
            shared = self.shared_rank[replica][stage][int(rank)]
            granted = self.granted_rank[replica][stage][int(rank)]
            readers = self.reader_rank[replica][stage][int(rank)]
        allocations = [
            Allocation("weights", self.weights[stage], "weights",
                       note="persistent resident weight bytes"),
            Allocation("activation_workspace", self.workspace[stage], "workspace",
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
        ]
        if include_cache:
            span = self.case.runtime.max_model_len if tokens is None else tokens
            if rank is None:
                _gross, resident, staging = self._kv_residency(span)
                resident_value = resident[stage]
                staging_value = staging[stage]
            else:
                gross = self._rank_cache_bytes(span, padded=True, rank=int(rank))
                layer_max = self._rank_max_layer_cache_bytes(span, int(rank))
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
        width = p.tp * p.pcp
        mounted = [
            [
                min(d.hbm_bytes * case.runtime.gpu_memory_utilization
                    for d in case.machine.dies[
                        (replica * p.pp + stage) * width:
                        (replica * p.pp + stage + 1) * width])
                for stage in range(p.pp)
            ]
            for replica in range(p.dp)
        ]
        persistent = [
            [
                self.weights[stage] + self.policy.reserved + self.workspace[stage]
                + self.temporary_dequant[stage] + resident_kv[stage]
                + staging_kv[stage] + self.selected_reserved[replica][stage]
                + self.shared_reserved[replica][stage] + self.granted[replica][stage]
                + max(self.reader_rank[replica][stage])
                for stage in range(p.pp)
            ]
            for replica in range(p.dp)
        ]
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
            "headroom_bytes_per_replica_stage": [
                [cap - used for cap, used in zip(cap_row, used_row)]
                for cap_row, used_row in zip(mounted, persistent)
            ],
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
                    [cap - used for cap, used in zip(
                        self.rank_capacity[replica][stage],
                        [a + b + c + d + e + f for a, b, c, d, e, f in zip(
                            self.live_rank[replica][stage],
                            self.staging_rank[replica][stage],
                            self.selected_rank[replica][stage],
                            self.shared_rank[replica][stage],
                            self.granted_rank[replica][stage],
                            self.reader_rank[replica][stage])])]
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
                    weight + self.policy.reserved + workspace + temporary + resident + staging
                    for weight, workspace, temporary, resident, staging in zip(
                        self.weights, self.workspace, self.temporary_dequant,
                        resident_kv, staging_kv)
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
            block_size = int(self.sparse_offload["block_size"])
            idx_row_bytes = int(self.sparse_offload["index_row_bytes"])
            full_idx_layers = len(self.sparse_offload["full_indexer_layers"])
            npu_per_block = max(1, block_size * idx_row_bytes * max(1, full_idx_layers))
            min_stage_cap = max(0.0, min(val for row in self.capacity for val in row))
            npu_limit_blocks = int(min_stage_cap // npu_per_block)
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
                    [cap - used for cap, used in zip(
                        self.rank_capacity[replica][stage],
                        [a + b + c + d + e + f for a, b, c, d, e, f in zip(
                            self.live_rank[replica][stage],
                            self.staging_rank[replica][stage],
                            self.selected_rank[replica][stage],
                            self.shared_rank[replica][stage],
                            self.granted_rank[replica][stage],
                            self.reader_rank[replica][stage])])]
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
                {"replica": rep, "owner": owner, **record}
                for (rep, owner), record in sorted(self.shared_pages.items())
            ],
            "peak_resident_bytes_per_replica_stage": [
                [weight + self.policy.reserved + workspace + temporary
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
