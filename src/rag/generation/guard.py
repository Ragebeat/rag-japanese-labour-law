"""Output guard: links and e-mail domains outside the allow-list never leave
the system, streamed answers included."""

import re

# The injection test: a poisoned document made the model append the
# attacker's link 4-8 times out of 8 under every prompt defence, and one
# sentence of stronger wording beat delimiting and escaping, 8/8. Only a
# code check on the output held, 32 of 32. A prompt lowers the odds; an
# allow-list on what leaves the system removes them for this class.
#
# An ALLOW-list, not "is the link in the sources": a poisoned source
# contains it. Bare domains and e-mail addresses count too: an attacker
# does not need the https://.
ALLOWED_DOMAINS = ("e-gov.go.jp", "mhlw.go.jp")
DOMAIN_RE = re.compile(r"(?:https?://|[A-Za-z0-9._%+-]+@)?((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})"
                       r"(?:[/:?#][^\s「」『』、。（）()]*)?")
LINK_MASK = "［リンク削除］"


def _allowed(domain):
    d = domain.lower()
    return any(d == a or d.endswith("." + a) for a in ALLOWED_DOMAINS)


def guard_links(text):
    """-> (text with every non-allow-listed link masked, blocked domains)."""
    blocked = []

    def repl(m):
        if _allowed(m.group(1)):
            return m.group(0)
        blocked.append(m.group(1))
        return LINK_MASK
    return DOMAIN_RE.sub(repl, text), blocked


_URL_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                 "0123456789.-:/?#&=%_@+~")


class StreamGuard:
    """guard_links for streamed text. A link must not be half-sent before it
    can be checked, so a trailing run of URL-like characters is held back
    until something else follows it. Everything emitted is cut at a non-URL
    character, so no link ever straddles two emitted chunks."""

    def __init__(self):
        self.raw, self.sent, self.blocked = "", 0, []

    def push(self, delta):
        self.raw += delta
        cut = len(self.raw)
        while cut > self.sent and self.raw[cut - 1] in _URL_CHARS:
            cut -= 1
        return self._emit(cut)

    def flush(self):
        return self._emit(len(self.raw))

    def _emit(self, cut):
        chunk, blocked = guard_links(self.raw[self.sent:cut])
        self.sent = cut
        self.blocked += blocked
        return chunk
