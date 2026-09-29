"""Re-score saved answers with the current check_quotes — no regeneration.

Generation took ~40 minutes; the first version of the quote metric was
wrong (it demanded 「…」[N] back to back and missed 71 of 102 quotes). The
answers were fine — only the scoring was. Keeping raw outputs in the report
is what makes a metric fixable in seconds instead of another 40-minute run:
generation and evaluation are separate steps.

The context is rebuilt from the report's article list. ⚠️ An approximation:
in `whole` mode an article longer than ARTICLE_CAP reached the model as ONE
paragraph, while here it is rebuilt whole. So for long articles the check is
slightly lenient — a quote from a paragraph the model never saw would pass.

  uv run python src/rescore_quotes.py            # the quote-prompt report
  uv run python src/rescore_quotes.py --base     # the baseline report (old prompt)
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from generate import (DEFAULT_MODE, VERDICT_RE, _by_article,  # noqa: E402
                      check_quotes, quote_summary, report_path)


def rebuild(context, arts):
    chunks = []
    for ref in context:
        law_id, art = ref.split(" ", 1)
        seen, bodies = set(), []
        for p in arts.get((law_id, art), []):
            body = p["text"].split("\n", 1)[-1]
            if body not in seen:
                seen.add(body)
                bodies.append(body)
        chunks.append({"text": f"{ref}\n" + "\n".join(bodies)})
    return chunks


def main():
    base = "--base" in sys.argv
    path = report_path(DEFAULT_MODE, "" if base else "_quotes")
    rows = json.loads(path.read_text(encoding="utf-8"))
    arts = _by_article()
    for r in rows:
        body = VERDICT_RE.sub("", r["answer"]).strip()
        r.update(check_quotes(body, rebuild(r["context"], arts)))
    print(f"пересчёт: {path.name} ({'базовый промпт' if base else 'промпт с цитатами'})")
    quote_summary(rows)


if __name__ == "__main__":
    main()
