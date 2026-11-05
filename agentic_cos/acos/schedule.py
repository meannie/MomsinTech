"""Write the scheduler files for every job that isn't off.

    python -m acos.schedule            # writes files to local/schedulers/ and prints how to turn them on
    python -m acos.schedule --remove   # prints how to turn them all off

It never installs anything itself. Turning the schedule on (launchctl, crontab or
systemctl) is a step the user approves.

Times come from config (all optional):
  schedule:
    triage: "23:00"
    meeting_prep_evening: "19:00"
    approvals_every_minutes: 5
    meeting_prep_soon_every_minutes: 15
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path

from . import config as cfgmod

DEFAULTS = {"triage": "23:00", "meeting_prep_evening": "19:00",
            "approvals_every_minutes": 5, "meeting_prep_soon_every_minutes": 15}


def jobs(cfg) -> list[dict]:
    """[{job, at: "HH:MM"} or {job, every: minutes}] for every module not switched off."""
    s = {**DEFAULTS, **(cfg.raw.get("schedule") or {})}
    mods = cfg.raw.get("modules") or {}
    on = lambda m: mods.get(m, "shadow") not in ("off", False)  # noqa: E731
    out = []
    if on("approvals"):
        out.append({"job": "approvals", "every": int(s["approvals_every_minutes"])})
    if on("triage"):
        out.append({"job": "triage", "at": s["triage"]})
    if on("meeting_prep"):
        out.append({"job": "meeting_prep_evening", "at": s["meeting_prep_evening"], "once": "day"})
        out.append({"job": "meeting_prep_soon", "every": int(s["meeting_prep_soon_every_minutes"])})
    for name, when in (s.get("custom") or {}).items():
        if on(name):
            out.append({"job": name, "at": when, "once": "day"})
    return out


def _args(j: dict) -> list[str]:
    return [j["job"]] + (["--once", j["once"]] if j.get("once") else [])


def launchd(j: dict, root: Path) -> tuple[str, str]:
    label = f"local.acos.{j['job']}"
    args = "".join(f"<string>{a}</string>" for a in ["/bin/bash", f"{root}/runner/run_job.sh", *_args(j)])
    if "every" in j:
        sched = f"<key>StartInterval</key><integer>{j['every'] * 60}</integer>"
    else:
        h, m = (int(x) for x in j["at"].split(":"))
        sched = (f"<key>StartCalendarInterval</key><dict><key>Hour</key><integer>{h}</integer>"
                 f"<key>Minute</key><integer>{m}</integer></dict>")
    plist = ('<?xml version="1.0" encoding="UTF-8"?>\n'
             '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
             f'<plist version="1.0"><dict><key>Label</key><string>{label}</string>'
             f'<key>ProgramArguments</key><array>{args}</array>{sched}'
             '<key>RunAtLoad</key><false/></dict></plist>\n')
    return f"{label}.plist", plist


def cron_line(j: dict, root: Path) -> str:
    cmd = f"{root}/runner/run_job.sh {' '.join(_args(j))}"
    if "every" in j:
        return f"*/{j['every']} * * * * {cmd}"
    h, m = (int(x) for x in j["at"].split(":"))
    return f"{m} {h} * * * {cmd}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--remove", action="store_true")
    p.add_argument("--kind", choices=["launchd", "cron"], default="launchd" if platform.system() == "Darwin" else "cron")
    a = p.parse_args()
    cfg = cfgmod.load()
    root = cfgmod.ROOT
    out = cfgmod.LOCAL / "schedulers"
    out.mkdir(parents=True, exist_ok=True)
    js = jobs(cfg)
    agents = Path.home() / "Library" / "LaunchAgents"

    if a.kind == "launchd":
        names = []
        for j in js:
            name, body = launchd(j, root)
            (out / name).write_text(body)
            names.append(name)
        if a.remove:
            print("To turn the schedule off, run:")
            for n in names:
                print(f"  launchctl unload {agents / n} && rm {agents / n}")
            return
        print(f"Wrote {len(names)} schedule files to {out}. To turn them on, run:")
        for n in names:
            print(f"  cp {out / n} {agents / n} && launchctl load {agents / n}")
    else:
        lines = [cron_line(j, root) for j in js]
        (out / "crontab.txt").write_text("\n".join(lines) + "\n")
        print("To turn the schedule on, run `crontab -e` and add:" if not a.remove
              else "To turn it off, run `crontab -e` and delete these lines:")
        print("\n".join("  " + l for l in lines))


if __name__ == "__main__":
    main()
