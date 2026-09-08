import base64
import json
import pathlib
import time

import requests

BASE = "https://laws.e-gov.go.jp/api/2"
MANIFEST = pathlib.Path("data/manifest.json")
RAW = pathlib.Path("data/raw")
META = pathlib.Path("data/meta")
PAUSE = 1.0


def fetch_law(law_id):
    r = requests.get(
        f"{BASE}/law_data/{law_id}",
        params={"law_full_text_format": "xml"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    META.mkdir(parents=True, exist_ok=True)
    records = json.loads(MANIFEST.read_text(encoding="utf-8"))

    failed = []
    for i, rec in enumerate(records, 1):
        law_id = rec["law_id"]
        xml_path = RAW / f"{law_id}.xml"
        if xml_path.exists():
            continue

        try:
            data = fetch_law(law_id)
            ft = data["law_full_text"]
            if not isinstance(ft, str):
                raise ValueError(f"law_full_text не строка, а {type(ft).__name__}")
            xml = base64.b64decode(ft).decode("utf-8")
            xml_path.write_text(xml, encoding="utf-8")
            meta = {k: v for k, v in data.items() if k != "law_full_text"}
            (META / f"{law_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{i}/{len(records)}] {law_id} {len(xml)} симв. {rec['law_title'][:30]}")
        except Exception as e:
            print(f"[{i}/{len(records)}] FAIL {law_id}: {e}")
            failed.append({"law_id": law_id, "error": str(e)})

        time.sleep(PAUSE)

    if failed:
        pathlib.Path("data/failed.json").write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nне скачано: {len(failed)} → data/failed.json")
    else:
        print(f"\nготово, файлов: {len(list(RAW.glob('*.xml')))}")
    if len(list(RAW.glob("*.xml"))) != len(records):
        print("ВНИМАНИЕ: файлов меньше, чем записей в манифесте")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
