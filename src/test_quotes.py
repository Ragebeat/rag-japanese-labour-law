"""Deterministic checks for the verbatim-quote check in generate.py.

A checker that is itself wrong makes every number it reports worthless — and
the first run of context_arts() produced 6 false hallucinations out
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
    print(f"{'✓' if ok else '✗'} {name}: {got}" + ("" if ok else f"  (expected {want})"))
    return ok


def run():
    r = []

    q = check_quotes("「賃金は、通貨で、直接労働者に、その全額を支払わなければならない」[1]", CH)
    r.append(check("a verbatim quote passes", (q["quotes"], q["quote_bad"]), (1, [])))

    # Full-width digits and stray spaces must not count as distortion.
    q = check_quotes("「賃金は、通貨で、 直接労働者に」[1]", CH)
    r.append(check("spaces do not break matching", q["quote_bad"], []))

    q = check_quotes("「賃金は、通貨で、…その全額を支払わなければならない」[1]", CH)
    r.append(check("ellipsis: fragments in order", q["quote_bad"], []))

    # Case 4: two exceptions spliced and presented as a quote.
    q = check_quotes("「法令若しくは労働協約に別段の定めがある場合においては、"
                     "賃金の一部を控除して支払うことができる」[1]", CH)
    r.append(check("splice → not a substring, nowhere",
                   [b["found_in"] for b in q["quote_bad"]], [[]]))

    # Case 5: text of article [2] quoted under number [3].
    q = check_quotes("「解雇その他不利益な取扱いをしてはならない」[3]", CH)
    r.append(check("wrong number → found in [2]",
                   [b["found_in"] for b in q["quote_bad"]], [[2]]))

    q = check_quotes("入社後6か月で10日付与される。「賃金の一部を控除して支払うことができる」[1]", CH)
    r.append(check("a number outside quotes is caught", q["num_unquoted"], ["6か月", "10日"]))

    q = check_quotes("[1] 労働基準法 第二十四条によると「賃金は、通貨で」[1]", CH)
    r.append(check("an article number is not a number outside quotes", q["num_unquoted"], []))

    q = check_quotes("「労働者」の場合", CH)
    r.append(check("a short 「」 is a term, not a quote", (q["quotes"], q["quote_loose"]), (0, [])))

    # The first real answer with the quote prompt: the undisputed sentence
    # quoted, the exceptions paraphrased — the splice repeated outside quotes.
    # Verbatim-ness says 1/1; coverage must say 1/3.
    q = check_quotes(
        "賃金は、原則として「通貨で、直接労働者に、その全額を支払わなければならない」[1] "
        "労働基準法 第二十四条。ただし、法令や労働協約などに基づき、賃金の一部を控除できる"
        "場合がある[1] 労働基準法 第二十四条。具体的には、法令に別段の定めがある場合"
        "や、書面による協定がある場合に限り控除が可能である[1] 労働基準法 第二十四条。",
        CH)
    r.append(check("splice outside quotes: verbatim 1/1, coverage 1/3",
                   (q["quotes"] - len(q["quote_bad"]), q["claims_quoted"], q["claims"]),
                   (1, 1, 3)))

    # A real case from earlier runs: 時季変更権 put under 第3項 — it is in 第5項. A shortened
    # stand-in for 第39条 with paragraph labels, as whole_article() builds it.
    a39 = {"text": "労働基準法 第三十九条\n"
                   "（第3項）次に掲げる労働者の有給休暇の日数については、比例して付与する。\n"
                   "（第5項）使用者は、前各項の規定による有給休暇を労働者の請求する時季に"
                   "与えなければならない。",
           "paragraphs": {
               3: "次に掲げる労働者の有給休暇の日数については、比例して付与する。",
               5: "使用者は、前各項の規定による有給休暇を労働者の請求する時季に"
                  "与えなければならない。"}}
    quote39 = "「前各項の規定による有給休暇を労働者の請求する時季に与えなければならない」[1] "

    q = check_quotes(quote39 + "労働基準法 第三十九条第3項", [a39])
    r.append(check("cites 第3項, the quote is in 第5項",
                   [(b["cited"], b["found"]) for b in q["para_bad"]], [([3], [5])]))

    q = check_quotes(quote39 + "労働基準法 第三十九条第五項", [a39])
    r.append(check("correct 第五項 (kanji) — no error",
                   (q["para_claims"], q["para_bad"]), (1, [])))

    # 第N項 inside a quote is the statute's own wording, not the answer's source.
    q = check_quotes("「第三項の規定による」[1] 労働基準法 第三十九条", [a39])
    r.append(check("第N項 inside a quote is not a citation", q["para_claims"], 0))
    return r


if __name__ == "__main__":
    res = run()
    bad = res.count(False)
    print(f"\n{len(res) - bad}/{len(res)}")
    sys.exit(1 if bad else 0)
