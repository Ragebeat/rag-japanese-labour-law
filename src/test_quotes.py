"""Deterministic checks for the verbatim-quote check in generate.py.

A checker that is itself wrong makes every number it reports worthless — and
the first run of context_arts() in phase 4 produced 6 false hallucinations out
of 6 for exactly that reason. So the checker is tested on hand-made answers
before it is trusted on model output.

  uv run python src/test_quotes.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from generate import check_quotes  # noqa: E402

# Two tiny "articles", shaped like chunks: a header line, then the body.
A24 = ("労働基準法 第二十四条\n"
       "賃金は、通貨で、直接労働者に、その全額を支払わなければならない。"
       "ただし、法令若しくは労働協約に別段の定めがある場合においては、"
       "通貨以外のもので支払い、また、法令に別段の定めがある場合又は"
       "書面による協定がある場合においては、賃金の一部を控除して支払うことができる。")
A10 = ("育児介護休業法 第十条\n"
       "事業主は、労働者が育児休業申出をしたことを理由として、"
       "当該労働者に対して解雇その他不利益な取扱いをしてはならない。")
A6 = ("育児介護休業法 第六条\n"
      "事業主は、労働者からの育児休業申出があったときは、"
      "当該育児休業申出を拒むことができない。")
CH = [{"text": A24}, {"text": A10}, {"text": A6}]


def check(name, got, want):
    ok = got == want
    print(f"{'✓' if ok else '✗'} {name}: {got}" + ("" if ok else f"  (ждали {want})"))
    return ok


def run():
    r = []

    q = check_quotes("「賃金は、通貨で、直接労働者に、その全額を支払わなければならない」[1]", CH)
    r.append(check("дословная цитата проходит", (q["quotes"], q["quote_bad"]), (1, [])))

    # Full-width digits and stray spaces must not count as distortion.
    q = check_quotes("「賃金は、通貨で、 直接労働者に」[1]", CH)
    r.append(check("пробелы не ломают сверку", q["quote_bad"], []))

    q = check_quotes("「賃金は、通貨で、…その全額を支払わなければならない」[1]", CH)
    r.append(check("многоточие: фрагменты по порядку", q["quote_bad"], []))

    # Case 4: two exceptions spliced and presented as a quote.
    q = check_quotes("「法令若しくは労働協約に別段の定めがある場合においては、"
                     "賃金の一部を控除して支払うことができる」[1]", CH)
    r.append(check("случай 4: склейка → не подстрока, нигде",
                   [b["found_in"] for b in q["quote_bad"]], [[]]))

    # Case 5: text of article [2] quoted under number [3].
    q = check_quotes("「解雇その他不利益な取扱いをしてはならない」[3]", CH)
    r.append(check("случай 5: чужой номер → найдено в [2]",
                   [b["found_in"] for b in q["quote_bad"]], [[2]]))

    q = check_quotes("入社後6か月で10日付与される。「賃金の一部を控除して支払うことができる」[1]", CH)
    r.append(check("число вне цитаты ловится", q["num_unquoted"], ["6か月", "10日"]))

    q = check_quotes("[1] 労働基準法 第二十四条によると「賃金は、通貨で」[1]", CH)
    r.append(check("номер статьи — не «число вне цитаты»", q["num_unquoted"], []))

    q = check_quotes("「労働者」の場合", CH)
    r.append(check("короткое 「」 — термин, не цитата", (q["quotes"], q["quote_loose"]), (0, [])))

    # The first real answer with the quote prompt: the undisputed sentence
    # quoted, the exceptions paraphrased — the splice repeated outside quotes.
    # Verbatim-ness says 1/1; coverage must say 1/3.
    q = check_quotes(
        "賃金は、原則として「通貨で、直接労働者に、その全額を支払わなければならない」[1] "
        "労働基準法 第二十四条。ただし、法令や労働協約などに基づき、賃金の一部を控除できる"
        "場合がある[1] 労働基準法 第二十四条。具体的には、法令に別段の定めがある場合"
        "や、書面による協定がある場合に限り控除が可能である[1] 労働基準法 第二十四条。",
        CH)
    r.append(check("склейка вне кавычек: дословно 1/1, покрытие 1/3",
                   (q["quotes"] - len(q["quote_bad"]), q["claims_quoted"], q["claims"]),
                   (1, 1, 3)))
    return r


if __name__ == "__main__":
    res = run()
    bad = res.count(False)
    print(f"\n{len(res) - bad}/{len(res)}")
    sys.exit(1 if bad else 0)
