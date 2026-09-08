import json
import pathlib
import time

import requests

BASE = "https://laws.e-gov.go.jp/api/2"
LIST_PATH = "/laws"  # <- check against the swagger spec
CATEGORY = "020"
LIMIT = 100
OUT = pathlib.Path("data/manifest.json")


def fetch_page(offset):
    r = requests.get(
        BASE + LIST_PATH,
        params={"category_cd": CATEGORY, "limit": LIMIT, "offset": offset},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def main():
    laws, offset = [], 0
    while True:
        page = fetch_page(offset)
        laws.extend(page["laws"])
        total = page["total_count"]
        print(f"offset={offset} got={len(page['laws'])} total={total}")
        nxt = page.get("next_offset")
        if nxt is None or len(laws) >= total:
            break
        offset = nxt
        time.sleep(1)

    records = []
    for item in laws:
        info, rev = item["law_info"], item["revision_info"]
        records.append(
            {
                "law_id": info["law_id"],
                "law_revision_id": rev["law_revision_id"],
                "law_type": info["law_type"],
                "law_title": rev["law_title"],
                "abbrev": rev["abbrev"],
                "category": rev["category"],
                "enforcement_date": rev["amendment_enforcement_date"],
                "repeal_status": rev["repeal_status"],
                "revision_status": rev["current_revision_status"],
            }
        )

    assert len(records) == total, f"собрано {len(records)}, ожидалось {total}"
    ids = [r["law_id"] for r in records]
    assert len(ids) == len(set(ids)), "дубли law_id"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"манифест: {len(records)} записей → {OUT}")


if __name__ == "__main__":
    main()
