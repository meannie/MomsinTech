"""The mail code can label and archive, and nothing else."""

import ast
import re
from pathlib import Path

from acos.triage.providers import GmailProvider, M365Provider, gmail_message, m365_message

SRC = (Path(__file__).resolve().parent.parent / "acos" / "triage" / "providers.py").read_text()

def _code_without_docstrings(src: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)   # also drops comments


CODE = _code_without_docstrings(SRC)


def test_no_destructive_calls_in_mail_code():
    for bad in [r"\.trash\(", r"\.delete\(", r"\.send\(", r"sendMail", r"forward", r"\.settings\(",
                r"filters", r"permanentDelete", r'"DELETE"', r"'DELETE'", r"deleteditems"]:
        assert not re.search(bad, CODE), f"forbidden call in providers.py: {bad}"


class _Req:
    def __init__(self, result, log, what):
        self.result, self.log, self.what = result, log, what

    def execute(self):
        self.log.append(self.what)
        return self.result


class FakeGmail:
    def __init__(self):
        self.calls = []

    def users(self):
        return self

    def labels(self):
        return self

    def messages(self):
        return self

    def list(self, **kw):
        return _Req({"labels": [{"name": "Keep", "id": "L_keep"}]}, self.calls, ("labels.list", None))

    def create(self, userId, body):
        return _Req({"id": "L_" + body["name"]}, self.calls, ("labels.create", body["name"]))

    def batchModify(self, userId, body):
        return _Req({}, self.calls, ("batchModify", body))


def test_gmail_apply_only_labels_and_removes_inbox():
    g = FakeGmail()
    p = GmailProvider(g, ["Keep"])
    rows = [{"id": "a", "label": "Newsletters", "archive": True},
            {"id": "b", "label": "Needs you", "archive": False},
            {"id": "c", "label": "(own message)", "archive": False}]
    p.apply(rows, "COS")
    mods = [c[1] for c in g.calls if c[0] == "batchModify"]
    assert {tuple(m["ids"]) for m in mods} == {("a",), ("b",)}
    for m in mods:
        assert set(m.get("removeLabelIds", [])) <= {"INBOX"}
    assert any(m.get("removeLabelIds") == ["INBOX"] and m["ids"] == ["a"] for m in mods)
    assert p.protected_ids == {"L_keep"}


class FakeSession:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw.get("json")))

        class R:
            status_code, headers, content = 200, {}, b""

            def raise_for_status(self):
                pass
        return R()


def test_m365_apply_only_categorizes_and_moves_to_archive():
    s = FakeSession()
    M365Provider(s, []).apply([{"id": "x", "label": "Receipts", "archive": True, "categories": ["Blue"]},
                               {"id": "y", "label": "FYI", "archive": False, "categories": []}], "COS")
    assert s.calls == [
        ("PATCH", "https://graph.microsoft.com/v1.0/me/messages/x", {"categories": ["Blue", "COS: Receipts"]}),
        ("POST", "https://graph.microsoft.com/v1.0/me/messages/x/move", {"destinationId": "archive"}),
        ("PATCH", "https://graph.microsoft.com/v1.0/me/messages/y", {"categories": ["COS: FYI"]}),
    ]


def test_gmail_normalization():
    raw = {"id": "1", "internalDate": "1762257600000", "labelIds": ["INBOX", "STARRED", "CATEGORY_PROMOTIONS"],
           "payload": {"headers": [{"name": "From", "value": "Brand <news@brand.example>"},
                                   {"name": "To", "value": "sam@example.com"},
                                   {"name": "Subject", "value": "Sale"},
                                   {"name": "List-Unsubscribe", "value": "<mailto:x>"}]}}
    m = gmail_message(raw, set())
    assert (m.from_addr, m.is_list, m.flagged, m.commercial) == ("news@brand.example", True, True, True)


def test_m365_normalization():
    raw = {"id": "9", "subject": "Invitation: Board", "receivedDateTime": "2026-11-04T10:00:00Z",
           "from": {"emailAddress": {"name": "Pat", "address": "pat@org.example"}},
           "toRecipients": [{"emailAddress": {"address": "sam@work.example.org"}}],
           "flag": {"flagStatus": "flagged"}, "categories": ["Keep"], "meetingMessageType": "meetingRequest",
           "internetMessageHeaders": [{"name": "Auto-Submitted", "value": "auto-generated"}]}
    m = m365_message(raw, ["Keep"])
    assert (m.is_invite, m.flagged, m.protected, m.auto_submitted) == (True, True, True, True)
    assert m.to == ["sam@work.example.org"]
