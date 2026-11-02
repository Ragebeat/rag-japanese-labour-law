"""Long-term memory for the LangGraph agent: what may be stored about a user,
per-user isolation, and the system prompt built from it.

Memory is a persistence mechanism: a value stored once is pasted into the
system prompt of every later conversation, so every field is a closed set or
a checked quote of what the user said."""

import json
import re
import unicodedata
from datetime import date

from langgraph.store.memory import InMemoryStore

from rag.agent.loop import (
    LLM_TIMEOUT,
    MODEL,
    SYSTEM,
    URL,
)
from rag.config import ROOT

MEMORY_DIR = ROOT / "data" / "memory"   # one file per user; in .gitignore: personal data
STALE_DAYS = 90

# OWASP review. The namespace used to be ("users", "local",
# "profile") with one file: harmless while the graph is a CLI for one
# person, a cross-user leak the day it is put behind an API: user B would
# get user A's facts in the system prompt. Now every memory function
# REQUIRES a user_id, so wiring the graph to an API cannot forget isolation:
# without an id the code does not run.
#
# The id comes from authentication at the entry point: never from the
# request body, never from the model. It also becomes a file name, hence the
# strict pattern: "../profile" must not escape data/memory/.
USER_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def namespace(user_id):
    if not isinstance(user_id, str) or not USER_ID_RE.fullmatch(user_id):
        raise ValueError(f"invalid user_id: {user_id!r}")
    return ("users", user_id, "profile")


def memory_file(user_id):
    namespace(user_id)                     # validates before touching the disk
    return MEMORY_DIR / f"{user_id}.json"


def check_owner(values, user_id):
    """A thread belongs to whoever started it. The checkpointer keys history
    by thread_id alone; if a thread_id ever comes from a client, knowing
    someone's id would be enough to read their conversation (IDOR)."""
    owner = values.get("owner")
    if owner is not None and owner != user_id:
        raise PermissionError("thread belongs to another user")


# The allow-list. Adding a field here is a privacy decision, not a code change.
LABELS = {"employment_type": "雇用形態", "company_size": "会社の従業員数",
          "prefecture": "勤務地（都道府県）"}
EMPLOYMENT = {"正社員": ["正社員"], "パート": ["パート"],
              "アルバイト": ["アルバイト", "バイト"], "契約社員": ["契約社員"],
              "派遣": ["派遣"]}

# Prefecture: a closed list, not "a string the user typed". The first
# check accepted any value literally present in the user's text, so a long
# instruction passed off as a prefecture would pass, get confirmed, and land
# in the SYSTEM PROMPT of every later conversation: a stored injection.
# (It also had a bug: rstrip("都道府県") strips any of those characters,
# turning 京都府 into 京.) Memory is a persistence mechanism; what goes in
# must be one of 47 known values.
PREFECTURES = (
    "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
    "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
    "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県",
    "静岡県", "愛知県", "三重県", "滋賀県", "京都府", "大阪府", "兵庫県",
    "奈良県", "和歌山県", "鳥取県", "島根県", "岡山県", "広島県", "山口県",
    "徳島県", "香川県", "愛媛県", "高知県", "福岡県", "佐賀県", "長崎県",
    "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県",
)


def _short(p):
    # 北海道 is written whole; the rest drop the suffix (群馬, 京都, 東京).
    return p if p == "北海道" else p[:-1]


def canonical_prefecture(value, said):
    """The full name if value is a real prefecture AND the user named it;
    otherwise None. Stored canonically, so 群馬 and 群馬県 are one fact."""
    if not isinstance(value, str):
        return None
    v = unicodedata.normalize("NFKC", value).strip()
    for p in PREFECTURES:
        if v in (p, _short(p)) and _short(p) in said:
            return p
    return None


EXTRACT = """ユーザーの発言から、本人が明言した事実だけを抽出してください。
推測はしない。発言に書かれていないものは必ず null にする。
- employment_type: 正社員 / パート / アルバイト / 契約社員 / 派遣 のいずれか
- company_size: 勤務先の従業員数（整数）
- prefecture: 勤務地の都道府県"""

# The schema allows null for every field: without a slot for "not said" the
# model invents a value.
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "employment_type": {"type": ["string", "null"]},
        "company_size": {"type": ["integer", "null"]},
        "prefecture": {"type": ["string", "null"]},
    },
    "required": ["employment_type", "company_size", "prefecture"],
}


def load_store(user_id, store=None):
    """Loads one user's file into their namespace (of a shared store, if given)."""
    store = store or InMemoryStore()
    path = memory_file(user_id)
    if path.exists():
        for k, v in json.loads(path.read_text(encoding="utf-8")).items():
            store.put(namespace(user_id), k, v)
    return store


def profile(store, user_id):
    return {it.key: it.value for it in store.search(namespace(user_id))}


def save_store(store, user_id):
    path = memory_file(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile(store, user_id), ensure_ascii=False,
                               indent=2), encoding="utf-8")


def _valid(field, value):
    """The same closed sets on the way OUT of memory as on the way in: the
    file can be edited by hand or hold values from an older, looser check,
    and whatever passes here lands in the system prompt."""
    if field == "employment_type":
        return value in EMPLOYMENT
    if field == "company_size":
        return isinstance(value, int) and 0 < value < 10_000_000
    if field == "prefecture":
        return value in PREFECTURES
    return False


def system_prompt(store, user_id):
    """The system prompt with the long-term facts, rebuilt every turn.

    Staleness is decided by code, not left to the model: a fact older than
    STALE_DAYS is marked for re-confirmation.
    """
    prof = profile(store, user_id)
    if not prof:
        return SYSTEM
    today, lines = date.today(), []
    for k, v in prof.items():
        if not _valid(k, v.get("value")):
            continue
        age = (today - date.fromisoformat(v["stated_at"])).days
        stale = "（古い情報・使う前に本人に確認すること）" if age > STALE_DAYS else ""
        lines.append(f"- {LABELS[k]}: {v['value']}"
                     f"（{v['stated_at']} に本人が述べた）{stale}")
    if not lines:
        return SYSTEM
    return (SYSTEM + "\n\n【このユーザーについて本人が以前述べた事実】\n"
            + "\n".join(lines) + "\n回答に関係する場合のみ使うこと。")


def extract_facts(http, text):
    body = {"model": MODEL, "stream": False, "think": False,
            "format": EXTRACT_SCHEMA, "options": {"temperature": 0},
            "messages": [{"role": "system", "content": EXTRACT},
                         {"role": "user", "content": text}]}
    try:
        r = http.post(URL, json=body, timeout=LLM_TIMEOUT)
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])
    except Exception:                               # noqa: BLE001
        return {}


def verify_stated(facts, text):
    """Keeps only values that literally occur in what the user said.

    The model is asked not to infer; this makes it impossible. "三十人" in
    kanji numerals is rejected too: a missed fact is cheap, a stored guess
    is not.
    """
    t = unicodedata.normalize("NFKC", text)
    out = {}
    et = facts.get("employment_type")
    if et in EMPLOYMENT and any(w in t for w in EMPLOYMENT[et]):
        out["employment_type"] = et
    cs = facts.get("company_size")
    if isinstance(cs, int) and cs > 0 and str(cs) in t:
        out["company_size"] = cs
    pf = canonical_prefecture(facts.get("prefecture"), t)
    if pf:
        out["prefecture"] = pf
    return out
