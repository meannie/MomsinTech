from datetime import datetime, timedelta, timezone

import pytest

from acos.config import DEFAULT_TRIAGE
from acos.triage.rules import Message, classify, keep_newest_insurance

ME = "sam@example.com"
NOW = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
RULES = {**DEFAULT_TRIAGE, "keep_visible_domains": ["kidsclub.example"],
         "always_needs_you_domains": ["school.example"], "vendor_domains": ["returns.example"]}


def msg(**kw):
    base = dict(id="1", from_name="Pat Lee", from_addr="pat@friend.example", to=[ME],
                subject="Dinner next week", received=NOW - timedelta(hours=1))
    base.update(kw)
    return Message(**base)


@pytest.mark.parametrize("m, label, archive", [
    (msg(), "Needs you", False),
    (msg(to=["group@friend.example"]), "FYI", False),
    (msg(from_addr="alerts@bank.example", subject="Security alert: new sign-in"), "Needs you", False),
    (msg(from_addr="no-reply@cal.example", subject="Invitation: Sync", is_invite=True), "Needs you", False),
    (msg(from_addr="office@school.example", is_list=True, subject="Picture day"), "Needs you", False),
    (msg(from_addr="news@kidsclub.example", is_list=True, subject="This week at the club"), "FYI", False),
    (msg(from_addr=ME), "(own message)", False),
    (msg(from_name="Pat (via Google Docs)", from_addr="comments-noreply@docs.example"), "Needs you", False),
    (msg(from_addr="hello@shop.example", subject="Re: your question"), "Needs you", False),
    (msg(from_addr="pat@friend.example", is_list=True, subject="PTA potluck"), "FYI", False),
    (msg(from_addr="billing@power.example", subject="Your bill is ready"), "Bills", False),
    (msg(from_addr="no-reply@shop.example", subject="Your order has shipped"), "Receipts", True),
    (msg(from_addr="news@brand.example", is_list=True, subject="Fall sale"), "Newsletters", True),
    (msg(from_addr="notifications@app.example", subject="Weekly summary"), "Notifications", True),
    (msg(from_addr="help@returns.example", subject="Return started"), "Notifications", True),
    (msg(from_addr="no-reply@app.example", subject="Your verification code"), "Notifications", False),
    (msg(from_addr="no-reply@app.example", subject="Your verification code",
         received=NOW - timedelta(hours=13)), "Notifications", True),
    (msg(from_addr="news@brand.example", is_list=True, subject="Fall sale", flagged=True), "Newsletters", False),
    (msg(from_addr="news@brand.example", is_list=True, subject="Fall sale", protected=True), "Newsletters", False),
    (msg(from_addr="pat@friend.example", commercial=True, subject="Promo"), "Newsletters", True),
])
def test_classify(m, label, archive):
    d = classify(m, ME, RULES, now=NOW)
    assert (d.label, d.archive) == (label, archive), d.reason


def test_archive_switch_respected():
    rules = {**RULES, "archive": {"receipts": False, "newsletters": True, "notifications": True}}
    d = classify(msg(from_addr="no-reply@shop.example", subject="Your receipt"), ME, rules, now=NOW)
    assert d.label == "Receipts" and d.archive is False


def test_people_never_archived_whatever_the_subject():
    for subject in ["Your order", "Fall sale", "Weekly summary", "receipt attached"]:
        assert classify(msg(subject=subject), ME, RULES, now=NOW).archive is False


def test_newest_insurance_notice_kept():
    rows = []
    for day, subj in [(1, "Your coverage confirmation 1001"), (2, "Your coverage confirmation 1002")]:
        m = msg(from_addr="no-reply@insurer.example", subject=subj, received=NOW - timedelta(days=day))
        d = classify(m, ME, RULES, now=NOW)
        rows.append({"from_addr": m.from_addr, "subject": subj, "received": m.received.isoformat(),
                     "label": d.label, "archive": d.archive, "reason": d.reason})
    keep_newest_insurance(rows)
    assert [r["archive"] for r in rows] == [False, True]
