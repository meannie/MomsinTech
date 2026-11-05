from pathlib import Path

from acos import config as cfgmod
from acos.schedule import cron_line, jobs, launchd
from acos.telegram_setup import find_chat


def cfg(modules, schedule=None):
    return cfgmod.from_dict({"user": {"address": "sam@example.com"}, "inboxes": [],
                             "modules": modules, "schedule": schedule or {}})


def test_off_modules_get_no_schedule():
    names = [j["job"] for j in jobs(cfg({"triage": "live", "approvals": False, "meeting_prep": "off"}))]
    assert names == ["triage"]


def test_defaults_and_overrides():
    js = {j["job"]: j for j in jobs(cfg({}, {"triage": "22:30", "custom": {"weekly_status": "08:00"}}))}
    assert js["approvals"]["every"] == 5 and js["triage"]["at"] == "22:30"
    assert js["meeting_prep_evening"]["once"] == "day" and js["weekly_status"]["at"] == "08:00"


def test_launchd_and_cron_output():
    root = Path("/opt/acos")
    name, body = launchd({"job": "triage", "at": "23:05"}, root)
    assert name == "local.acos.triage.plist"
    assert "<integer>23</integer>" in body and "<integer>5</integer>" in body and "/opt/acos/runner/run_job.sh" in body
    assert "<integer>300</integer>" in launchd({"job": "approvals", "every": 5}, root)[1]
    assert cron_line({"job": "triage", "at": "23:05"}, root) == "5 23 * * * /opt/acos/runner/run_job.sh triage"
    assert cron_line({"job": "x", "at": "07:00", "once": "day"}, root).endswith("run_job.sh x --once day")


def test_find_chat_picks_latest_private():
    ups = [{"message": {"chat": {"id": 1, "type": "private", "first_name": "A"}}},
           {"message": {"chat": {"id": -5, "type": "group"}}},
           {"message": {"chat": {"id": 2, "type": "private", "first_name": "B"}}}]
    assert find_chat(ups) == (2, "B")
    assert find_chat([]) is None
