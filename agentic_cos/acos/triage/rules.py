"""Triage rules: one label per message, and whether it may be archived.

Pure functions over a provider-neutral Message, so the same rules run for Gmail
and Microsoft 365 and can be tested without any account.

Labels (prefix from config, default "COS"):
  Needs you      a person, or an urgent alert, waiting on you        -> stays
  Bills          bill reminders                                      -> stays
  FYI            a person, but you're cc'd or it's a group           -> stays
  Receipts       orders, receipts, statements, insurance notices     -> archived
  Newsletters    mailing lists and promotions                        -> archived
  Notifications  automated notices                                   -> archived

Never archived: flagged/starred mail, mail with a protected label, calendar
invites, anything from a person, urgent alerts, bill reminders, login codes
under 12 hours old, and the newest insurance notice of each kind.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

URGENT = re.compile(
    r"\b(fraud|suspicious|security alert|unusual (sign-in|activity)|account (locked|suspended)|"
    r"past due|overdue|failed payment|payment (has )?(failed|declined)|action required|final notice|"
    r"verify your (identity|account)|urgent|open enrollment|added to your account|new sign-in|"
    r"new device|password (was )?(changed|reset)|unauthorized|violation reported|"
    r"invited you to comment|mentioned you|assigned (to )?you|signature request|please (e-?)?sign|"
    r"ready (for|to) sign|awaiting your signature)\b", re.I)
LOGIN_CODE = re.compile(r"\b(verification code|security code|one-time|sign-in code|login code|passcode|OTP)\b", re.I)
RECEIPT = re.compile(
    r"\b(receipt|your order|order (confirmation|#|number|shipped|delivered)|has shipped|out for delivery|"
    r"delivered|invoice|statement is (ready|available)|payment (received|confirmation|processed)|"
    r"you paid|refund|subscription (renewal|receipt)|billing)\b", re.I)
BILL = re.compile(r"\b(pay your bill|bill (is )?(ready|due|available)|payment (is )?due|amount due|"
                  r"minimum payment|due date|autopay (scheduled|reminder))\b", re.I)
INSURANCE = re.compile(r"\b(cobra|insurance|enrollment confirmation|benefits? (confirmation|statement|summary)|"
                       r"explanation of benefits|eob|policy (documents?|renewal|confirmation)|coverage)\b", re.I)
AUTOMATED_LOCAL = re.compile(
    r"^(no-?reply|do-?not-?reply|notifications?|notify|alerts?|updates?|mailer(-daemon)?|"
    r"automated|system|support|info|hello|team|news|newsletter|digest|marketing|billing|"
    r"receipts?|orders?|accounts?|security|calendar-notification)([+._-].*)?$", re.I)
BULK_DOMAIN = re.compile(r"^(e|em|email|mail|mailer|info\d*|news|marketing|notify|go|reply)\.")
DOC_SHARE = re.compile(r"\(?via google (docs|drive|sheets|slides)\)?|\(google (docs|drive|sheets|slides)\)", re.I)
NOREPLY = re.compile(r"^(no-?reply|do-?not-?reply|notifications?)", re.I)


@dataclass
class Message:
    """What the rules need from a message. Providers fill this from headers only, never bodies."""
    id: str
    from_name: str
    from_addr: str
    to: list[str]
    subject: str
    received: datetime                 # timezone-aware
    is_list: bool = False              # List-Unsubscribe / List-Id / Precedence: bulk
    auto_submitted: bool = False       # Auto-Submitted header other than "no"
    is_invite: bool = False            # text/calendar or a meeting request
    commercial: bool = False           # provider says promotions/social/updates/forums
    flagged: bool = False              # starred (Gmail) or flagged (Outlook)
    protected: bool = False            # carries a protected label/category
    extra: dict = field(default_factory=dict)


@dataclass
class Decision:
    label: str          # short name ("Needs you") or "(own message)"
    archive: bool
    reason: str


def _domain_match(domain: str, domains) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in domains)


def classify(m: Message, me: str, rules: dict, now: datetime | None = None) -> Decision:
    now = now or datetime.now(timezone.utc)
    addr = m.from_addr.lower()
    local, _, domain = addr.partition("@")
    subject = m.subject or ""
    to = [a.lower() for a in m.to]
    me = me.lower()

    bulk_domain = bool(BULK_DOMAIN.match(domain))
    vendor = _domain_match(domain, rules.get("vendor_domains", []))
    automated = bool(AUTOMATED_LOCAL.match(local)) or m.auto_submitted or bulk_domain or vendor
    keep = [r for r, on in (("flagged", m.flagged), ("protected label", m.protected)) if on]

    if URGENT.search(subject):
        return Decision("Needs you", False, "urgent alert wording")
    if m.is_invite:
        return Decision("Needs you", False, "calendar invite")
    if _domain_match(domain, rules.get("always_needs_you_domains", [])):
        return Decision("Needs you", False, f"always-needs-you sender {domain}")
    if _domain_match(domain, rules.get("keep_visible_domains", [])):
        return Decision("FYI", False, f"keep-visible sender {domain}")
    if addr == me:
        return Decision("(own message)", False, "your own message in a thread")
    if DOC_SHARE.search(m.from_name):
        return Decision("Needs you", False, "document comment or share")
    # A reply in a thread from a non-noreply address is a person, even from hello@ or info@.
    if subject.lower().startswith("re:") and not NOREPLY.match(local) and not bulk_domain and not m.commercial:
        return Decision("Needs you" if me in to else "FYI", False, "reply in a thread")
    person = not (m.is_list or automated or m.commercial)
    # A person posting through a group or list stays visible.
    if m.is_list and not (automated or m.commercial) and "via" not in m.from_name.lower() and "'" not in m.from_name:
        return Decision("FYI", False, "person posting to a group or list")
    if person:
        return Decision("Needs you", False, "person, addressed to you") if me in to \
            else Decision("FYI", False, "person, you're cc'd or it's a group")

    if BILL.search(subject):
        return Decision("Bills", False, "bill reminder (kept)")
    archive_cfg = rules.get("archive", {})
    if INSURANCE.search(subject):
        reason = "insurance/benefits notice"
        if keep:
            return Decision("Receipts", False, reason + "; kept: " + ", ".join(keep))
        return Decision("Receipts", bool(archive_cfg.get("receipts", True)), reason)
    if LOGIN_CODE.search(subject):
        label, reason, kind = "Notifications", "login or verification code", "notifications"
        if now - m.received < timedelta(hours=12):
            return Decision(label, False, reason + " (under 12h old, kept)")
    elif RECEIPT.search(subject):
        label, reason, kind = "Receipts", "receipt or order wording", "receipts"
    elif m.is_list or m.commercial:
        label, reason, kind = "Newsletters", "mailing list or promotion", "newsletters"
    else:
        label, reason, kind = "Notifications", "automated sender", "notifications"

    if keep:
        return Decision(label, False, reason + "; kept: " + ", ".join(keep))
    return Decision(label, bool(archive_cfg.get(kind, True)), reason)


def keep_newest_insurance(rows: list[dict]) -> None:
    """Keep the newest insurance/benefits notice per sender + subject shape in the inbox."""
    seen = set()
    for r in sorted(rows, key=lambda x: x["received"], reverse=True):
        if r["reason"].startswith("insurance/benefits"):
            key = (r["from_addr"].lower(), re.sub(r"\d+", "#", r["subject"].lower()))
            if key not in seen:
                seen.add(key)
                if r["archive"]:
                    r["archive"] = False
                    r["reason"] += " (newest of its kind, kept)"
