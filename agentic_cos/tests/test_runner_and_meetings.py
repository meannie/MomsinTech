import os
import subprocess
import sys
from datetime import date
from pathlib import Path

from acos.meetings import needs_prep

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runner" / "run_job.sh"


def run_job(local: Path, *args, claude="echo packet-text"):
    fake = local / "fake_claude.sh"
    fake.write_text(f"#!/bin/bash\ncat >/dev/null\n{claude}\n")
    fake.chmod(0o755)
    env = {**os.environ, "ACOS_LOCAL": str(local), "ACOS_PYTHON": sys.executable,
           "ACOS_CLAUDE": str(fake), "ACOS_DRY_SEND": "1", "PYTHONPATH": str(ROOT)}
    subprocess.run(["/bin/bash", str(RUN), *args], env=env, check=False)
    return (local / "logs" / f"{args[0]}.log").read_text()


def setup(tmp_path, mode="live"):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "digest.md").write_text("Summarize the material.")
    (tmp_path / "config.yaml").write_text(f"modules:\n  digest: {mode}\n")
    return tmp_path


def test_custom_job_composes_and_sends(tmp_path):
    log = run_job(setup(tmp_path), "digest")
    assert "would send digest: digest" in log and "packet-text" in log


def test_shadow_mode_marks_subject(tmp_path):
    assert "[shadow]" in run_job(setup(tmp_path, "shadow"), "digest")


def test_off_mode_does_nothing(tmp_path):
    log = run_job(setup(tmp_path, "off"), "digest")
    assert "off" in log and "would send" not in log


def test_only_on_other_day_skips(tmp_path):
    assert "only runs on 1999-01-01" in run_job(setup(tmp_path), "digest", "--only-on", "1999-01-01")
    assert "would send" in run_job(tmp_path, "digest", "--only-on", date.today().isoformat())


def test_once_per_day(tmp_path):
    local = setup(tmp_path)
    run_job(local, "digest", "--once", "day")
    assert "already ran for" in run_job(local, "digest", "--once", "day")


def test_lock_prevents_overlap(tmp_path):
    local = setup(tmp_path)
    (local / "state" / "locks" / "digest").mkdir(parents=True)
    assert "still running from before" in run_job(local, "digest")


def test_empty_claude_output_sends_nothing(tmp_path):
    log = run_job(setup(tmp_path), "digest", claude="true")
    assert "empty output" in log and "would send" not in log


def test_claude_gets_no_tools():
    src = RUN.read_text()
    assert '--tools ""' in src and "--allowedTools" not in src


MINE = {"example.com"}
RULES = {"important_keywords": ["board"], "skip_keywords": ["focus"]}


def ev(**kw):
    base = {"title": "Intro call", "all_day": False, "declined": False, "recurring": False,
            "attendees": [{"email": "pat@partner.example", "resource": False}]}
    base.update(kw)
    return base


def test_needs_prep():
    assert needs_prep(ev(), MINE, RULES)[0]
    assert not needs_prep(ev(attendees=[{"email": "kim@example.com"}]), MINE, RULES)[0]
    assert needs_prep(ev(title="Board prep", attendees=[]), MINE, RULES)[0]
    assert not needs_prep(ev(title="Focus time"), MINE, RULES)[0]
    assert not needs_prep(ev(declined=True), MINE, RULES)[0]
    assert not needs_prep(ev(all_day=True), MINE, RULES)[0]
    assert not needs_prep(ev(attendees=[{"email": "room@rooms.example", "resource": True}]), MINE, RULES)[0]
