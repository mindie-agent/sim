"""AG014 full-package zero-model-name-branch governance verification.

[Role] Verify that ``scripts/check_arch_governance.py`` and
  ``scripts/_arch_governance_runtime.py`` accurately detect per-model-name
  branches across executable ``mindie_sim/**/*.py`` and
  ``mindie_sim/stacks/vllm_ascend/manifest.json``, while exempting structural
  predicates, fusion-tier constants, docstrings, ``SourceRef`` provenance, and
  narrow config-ingestion adapters.
[Invariants] Default CLI execution and ``--strict`` enforce AG014
  unconditionally as a hard failure (exit code 1, zero staged warnings);
  ``--strict-ag014`` remains accepted as a compatibility no-op flag.
[Timeouts & Locks] Pure Tier-0 in-memory/tmp_path AST and JSON scans.
[Side-Effect Policy] Writes synthetic fixtures exclusively under ``tmp_path``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.check_arch_governance as arch_gov

pytestmark = pytest.mark.tier0


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_ag014_catches_direct_aliased_set_dict_match_and_manifest_models(tmp_path: Path) -> None:
    proj = tmp_path / "repo"
    pred = proj / "mindie_sim" / "prediction"
    _write(
        pred / "bad_branches.py",
        'BAD_MODELS = frozenset({"qwen3_next"})\n'
        'MODEL_DISPATCH = {"glm-5.3-flash": 1}\n'
        'def decide(model, model_name: str, model_type: str, token: str) -> int:\n'
        '    if model_name == "kimi-k3":\n'
        '        return 1\n'
        '    if model.name == "hunyuan-video-dit":\n'
        '        return 5\n'
        '    if re.search("unregistered-model", model.name):\n'
        '        return 6\n'
        '    if re.search(pattern="kw-unregistered-model", string=model.name):\n'
        '        return 7\n'
        '    inline = {"qwen3-32b": 5}.get(model.name, 0)\n'
        '    alias = model.name.lower()\n'
        '    if alias.startswith("deepseek"):\n'
        '        return 2\n'
        '    if token in BAD_MODELS:\n'
        '        return 3\n'
        '    val = MODEL_DISPATCH.get(model_type, inline)\n'
        '    match model_type:\n'
        '        case "gemma4":\n'
        '            return 4\n'
        '        case _:\n'
        '            return val\n',
    )
    _write(
        proj / "mindie_sim" / "stacks" / "vllm_ascend" / "manifest.json",
        json.dumps(
            {
                "bindings": [
                    {
                        "mechanism_id": "bad_model_scoped_binding",
                        "state": "merged",
                        "models": ["glm-5.3-flash"],
                        "capability_predicates": [],
                    },
                    {
                        "mechanism_id": "bad_dict_models_binding",
                        "state": "merged",
                        "models": {"qwen3-32b": True},
                        "capability_predicates": [],
                    },
                    {
                        "mechanism_id": "clean_structural_binding",
                        "state": "merged",
                        "models": [],
                        "capability_predicates": ["qk_rope_head_dim == 0"],
                    },
                ]
            },
            indent=2,
        ),
    )

    findings = arch_gov.scan_ag014(proj)
    assert findings
    by_file: dict[str, list[arch_gov.Violation]] = {}
    for item in findings:
        assert item.rule == "AG014"
        by_file.setdefault(item.path.as_posix(), []).append(item)

    py_fps = {v.fingerprint for v in by_file["mindie_sim/prediction/bad_branches.py"]}
    assert any("model_name_compare" in fp and "kimi" in fp for fp in py_fps)
    assert any("model_name_compare" in fp and "hunyuan-video-dit" in fp for fp in py_fps)
    assert any("model_name_regex" in fp and "unregistered-model" in fp for fp in py_fps)
    assert any("model_name_regex" in fp and "kw-unregistered-model" in fp for fp in py_fps)
    assert any("model_name_table_lookup" in fp and "inline_dict:qwen3-32b" in fp for fp in py_fps)
    assert any("model_name_str_method" in fp and "deepseek" in fp for fp in py_fps)
    assert any("model_name_compare" in fp and "qwen3_next" in fp for fp in py_fps)
    assert any("model_name_table_lookup" in fp and "MODEL_DISPATCH".lower() in fp for fp in py_fps)
    assert any("model_name_match" in fp and "gemma4" in fp for fp in py_fps)

    manifest_items = by_file["mindie_sim/stacks/vllm_ascend/manifest.json"]
    assert len(manifest_items) == 2
    assert {m.fingerprint for m in manifest_items} == {
        "AG014:manifest_models:bad_model_scoped_binding",
        "AG014:manifest_models:bad_dict_models_binding",
    }


def test_ag014_allows_structural_predicates_docstrings_sourceref_and_config_exemptions(tmp_path: Path) -> None:
    proj = tmp_path / "repo"
    valid_rev = "a" * 40
    valid_sha = "b" * 64
    _write(
        proj / "mindie_sim" / "prediction" / "clean_structural.py",
        '"""Structural planner supporting Kimi-K3, DeepSeek-V4.1, and GLM-5.3 via traits."""\n'
        'from mindie_sim.schema.datapath import SourceRef\n'
        f'REF = SourceRef("vllm-ascend", "{valid_rev}", "vllm_ascend/models/qwen3_next.py", "{valid_sha}")\n'
        'NORM_ABI = "GemmaRmsNorm"\n'
        'def select_policy(plan, tier: str, kind: str = "") -> str:\n'
        '    """Mentioning Qwen3.8 or Gemma4 in a docstring is allowed."""\n'
        '    if plan.qk_rope_head_dim == 0 and plan.mla_variant == "latent_mqa":\n'
        '        return "nope_mqa"\n'
        '    if plan.has_sandwich_norm and plan.v_norm and plan.gdn_layout == "interleaved":\n'
        '        return "structural_norm"\n'
        '    if tier in ("K1", "K2", "K3", "K1_shipping", "K3_dsl_pipeline"):\n'
        '        return tier\n'
        '    if kind == "GemmaRmsNorm" or kind == NORM_ABI:\n'
        '        return "native_abi"\n'
        '    return "default"\n',
    )
    _write(
        proj / "mindie_sim" / "catalog" / "hf_config.py",
        'def parse_hf(model_type: str) -> str:\n'
        '    if model_type == "qwen3_5_text":\n'
        '        return "interleaved"\n'
        '    return "standard"\n',
    )
    _write(
        proj / "mindie_sim" / "stacks" / "vllm_ascend" / "manifest.json",
        json.dumps(
            {
                "bindings": [
                    {
                        "mechanism_id": "structural_mlapo",
                        "state": "merged",
                        "models": [],
                        "capability_predicates": ["qk_rope_head_dim > 0", "mla_variant == 'latent_mla'"],
                    }
                ]
            },
            indent=2,
        ),
    )

    assert arch_gov.scan_ag014(proj) == ()
    assert arch_gov.scan_repository(proj, include_ag014=True) == ()


def test_ag014_cli_default_on_strict_and_compat_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    proj = tmp_path / "repo"
    _write(
        proj / "mindie_sim" / "prediction" / "flagged.py",
        'def pick(model_name: str) -> int:\n'
        '    return 1 if model_name == "minimax-m3" else 0\n',
    )

    rc_default = arch_gov.main(["--root", str(proj)])
    out_default = capsys.readouterr().out
    assert rc_default == arch_gov.EXIT_VIOLATIONS
    assert "AG014" in out_default
    assert "staged warnings" not in out_default
    assert "arch governance violations: 1" in out_default

    rc_strict = arch_gov.main(["--root", str(proj), "--strict-ag014"])
    out_strict = capsys.readouterr().out
    assert rc_strict == arch_gov.EXIT_VIOLATIONS
    assert "AG014" in out_strict
    assert "staged warnings" not in out_strict
    assert "arch governance violations: 1" in out_strict

    _write(
        proj / "mindie_sim" / "prediction" / "flagged.py",
        'def pick(has_sandwich_norm: bool) -> int:\n'
        '    return 1 if has_sandwich_norm else 0\n',
    )
    assert arch_gov.main(["--root", str(proj)]) == arch_gov.EXIT_OK
    assert arch_gov.main(["--root", str(proj), "--strict-ag014"]) == arch_gov.EXIT_OK
