from __future__ import annotations

import types
from pathlib import Path

from chemlflow_dashboard import cli


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_tabpfn_chemeleon_doe_preflight_checks_optional_stack(monkeypatch) -> None:
    spec_path = REPO_ROOT / "doe" / "pgp_tabpfn_foundation_demo.yaml"
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _module: object())
    captured: dict[str, object] = {}

    def _fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return types.SimpleNamespace(returncode=0, stdout="probe ok", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)

    ok, result = cli._preflight_doe(spec_path)

    assert ok is True
    assert result["models"] == ["tabpfn"]
    assert result["feature_inputs"] == [
        "featurize.chemeleon_fp",
        "featurize.rdkit",
    ]
    assert set(result["modules"]) == {"chemprop", "tabpfn", "torch"}
    assert result["checkpoint"]["available"] is True
    probe_script = captured["command"][2]
    assert "ModelVersion.V2_6" in probe_script
    assert "from chemprop import data, featurizers, models, nn" in probe_script
