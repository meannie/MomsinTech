"""Mailbox access for triage: read headers, add a label, archive. Nothing else.

There is deliberately no trash, delete, send, forward or filter code in this
file. tests/test_providers.py fails if any of those calls appear.

Gmail:  labels are Gmail labels ("COS/Needs you"); archive = remove INBOX.
M365:   labels are Outlook categories ("COS: Needs you"); archive = move to
        the Archive folder. Both stay searchable.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from email.utils import getaddresses, parseaddr

from .rules import Message

GMAIL_HEADERS = ["From", "To", "Subject", "List-Unsubscribe", "List-Id", "Precedence",
                 "Auto-Submitted", "Content-Type"]
GMAIL_COMMERCIAL = {"CATEGORY_UPDATES", "CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_FORUMS"}


# ---------------------------------------------------------------- Gmail

def _run(req, tries: int = 6):
    """Execute a Google API request, backing off on rate limits."""
    try:
        from googleapiclient.errors import HttpError
    except ImportError:          # Microsoft-only installs don't need the Google client
        HttpError = type("HttpError", (Exception,), {})
    for i in range(tries):
        try:
            return req.execute()
        except HttpError as e:
            if e.resp.status in (403, 429, 500, 503) and i < tries - 1:
                time.sleep(min(60, 5 * 2 ** i))
                continue
            raise


def gmail_message(raw: dict, protected_ids: set[str]) -> Message:
    h = {x["name"].lower(): x["value"] for x in raw.get("payload", {}).get("headers", [])}
    labels = set(raw.get("labelIds", []))
    name, addr = parseaddr(h.get("from", ""))
    subject = h.get("subject", "")
    return Message(
        id=raw["id"],
        from_name=name,
        from_addr=addr,
        to=[a for _, a in getaddresses([h.get("to", "")])],
        subject=subject,
        received=datetime.fromtimestamp(int(raw["internalDate"]) / 1000, timezone.utc),
        is_list="list-unsubscribe" in h or "list-id" in h or h.get("precedence", "").lower() in ("bulk", "list"),
        auto_submitted=h.get("auto-submitted", "no").lower() != "no",
        is_invite="text/calendar" in h.get("content-type", "") or subject.startswith(("Invitation:", "Updated invitation:")),
        commercial=bool(labels & GMAIL_COMMERCIAL),
        flagged="STARRED" in labels,
        protected=bool(labels & protected_ids),
    )


class GmailProvider:
    def __init__(self, service, protected_labels: list[str]):
        self.svc = service
        self._labels = {l["name"]: l["id"] for l in _run(service.users().labels().list(userId="me")).get("labels", [])}
        self.protected_ids = {self._labels[n] for n in protected_labels if n in self._labels}

    def fetch(self, days: int) -> list[Message]:
        ids, token = [], None
        while True:
            r = _run(self.svc.users().messages().list(userId="me", q=f"in:inbox newer_than:{days}d",
                                                      maxResults=500, pageToken=token))
            ids += [m["id"] for m in r.get("messages", [])]
            token = r.get("nextPageToken")
            if not token:
                break
        out, pending, wait = [], ids, 2
        while pending:
            failed = []

            def cb(rid, resp, exc):
                (failed.append(rid) if exc is not None else out.append(resp))

            for i in range(0, len(pending), 25):
                batch = self.svc.new_batch_http_request(callback=cb)
                for mid in pending[i:i + 25]:
                    batch.add(self.svc.users().messages().get(
                        userId="me", id=mid, format="metadata", metadataHeaders=GMAIL_HEADERS), request_id=mid)
                _run(batch)
                time.sleep(0.5)
            if failed and wait > 64:
                raise RuntimeError(f"{len(failed)} messages still failing after retries")
            pending = failed
            if pending:
                time.sleep(wait)
                wait *= 2
        return [gmail_message(m, self.protected_ids) for m in out]

    def _label_id(self, name: str) -> str:
        if name not in self._labels:
            self._labels[name] = _run(self.svc.users().labels().create(
                userId="me", body={"name": name, "labelListVisibility": "labelShow",
                                   "messageListVisibility": "show"}))["id"]
        return self._labels[name]

    def apply(self, rows: list[dict], prefix: str) -> None:
        groups: dict[tuple[str, bool], list[str]] = {}
        for r in rows:
            if not r["label"].startswith("("):
                groups.setdefault((r["label"], r["archive"]), []).append(r["id"])
        for (label, archive), ids in groups.items():
            body = {"addLabelIds": [self._label_id(f"{prefix}/{label}")]}
            if archive:
                body["removeLabelIds"] = ["INBOX"]
            for i in range(0, len(ids), 1000):
                _run(self.svc.users().messages().batchModify(userId="me", body={**body, "ids": ids[i:i + 1000]}))


# ---------------------------------------------------------------- Microsoft 365 (Graph)

GRAPH = "https://graph.microsoft.com/v1.0"
M365_SELECT = "id,subject,from,toRecipients,receivedDateTime,flag,categories,meetingMessageType,internetMessageHeaders"


def m365_message(raw: dict, protected: list[str]) -> Message:
    h = {x["name"].lower(): x["value"] for x in raw.get("internetMessageHeaders", []) or []}
    sender = (raw.get("from") or {}).get("emailAddress", {})
    subject = raw.get("subject") or ""
    cats = set(raw.get("categories") or [])
    return Message(
        id=raw["id"],
        from_name=sender.get("name", ""),
        from_addr=sender.get("address", ""),
        to=[r["emailAddress"]["address"] for r in raw.get("toRecipients", []) if r.get("emailAddress")],
        subject=subject,
        received=datetime.fromisoformat(raw["receivedDateTime"].replace("Z", "+00:00")),
        is_list="list-unsubscribe" in h or "list-id" in h or h.get("precedence", "").lower() in ("bulk", "list"),
        auto_submitted=h.get("auto-submitted", "no").lower() != "no",
        is_invite=raw.get("meetingMessageType") not in (None, "none") or subject.startswith(("Invitation:", "Updated invitation:")),
        commercial=False,
        flagged=(raw.get("flag") or {}).get("flagStatus") == "flagged",
        protected=bool(cats & set(protected)),
        extra={"categories": sorted(cats)},
    )


class M365Provider:
    """`session` is a requests.Session with the Authorization header already set."""

    def __init__(self, session, protected_labels: list[str]):
        self.http = session
        self.protected = protected_labels

    def _req(self, method: str, url: str, **kw):
        for i in range(6):
            r = self.http.request(method, url, timeout=60, **kw)
            if r.status_code in (429, 500, 503) and i < 5:
                time.sleep(int(r.headers.get("Retry-After", 5 * 2 ** i)))
                continue
            r.raise_for_status()
            return r.json() if r.content else {}

    def fetch(self, days: int) -> list[Message]:
        since = datetime.now(timezone.utc).timestamp() - days * 86400
        iso = datetime.fromtimestamp(since, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = (f"{GRAPH}/me/mailFolders/inbox/messages?$top=50&$select={M365_SELECT}"
               f"&$filter=receivedDateTime ge {iso}")
        out = []
        while url:
            page = self._req("GET", url)
            out += [m365_message(m, self.protected) for m in page.get("value", [])]
            url = page.get("@odata.nextLink")
        return out

    def apply(self, rows: list[dict], prefix: str) -> None:
        for r in rows:
            if r["label"].startswith("("):
                continue
            cats = sorted(set(r.get("categories", [])) | {f"{prefix}: {r['label']}"})
            self._req("PATCH", f"{GRAPH}/me/messages/{r['id']}", json={"categories": cats})
            if r["archive"]:
                self._req("POST", f"{GRAPH}/me/messages/{r['id']}/move", json={"destinationId": "archive"})
