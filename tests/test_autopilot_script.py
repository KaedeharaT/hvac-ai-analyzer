from pathlib import Path


def test_autopilot_uses_repository_mutex_and_unique_per_stream_logs():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "run_codex_until_done.ps1").read_text(encoding="utf-8")
    assert "BuildingAIAutopilot_" in script
    assert "AUTOPILOT ALREADY RUNNING" in script
    assert "run_{0}_round_{1:D3}_attempt_{2:D2}" in script
    assert "$stdoutLog" in script and "$stderrLog" in script
    assert "run_001.log" not in script
    assert "BUILDINGAI AUTOPILOT COMPLETE" in script
