"""Deterministic checks for the verbatim-quote check in generation/checks.py.

A checker that is itself wrong makes every number it reports worthless, and
the first run of context_arts() produced 6 false hallucinations out
of 6 for exactly that reason. So the checker is tested on hand-made answers
before it is trusted on model output.

  uv run pytest tests/test_quotes.py
"""

from rag.generation.checks import check_quotes

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

# A shortened stand-in for 第39条 with paragraph labels, as whole_article() builds it.
A39 = {"text": "労働基準法 第三十九条\n"
               "（第3項）次に掲げる労働者の有給休暇の日数については、比例して付与する。\n"
               "（第5項）使用者は、前各項の規定による有給休暇を労働者の請求する時季に"
               "与えなければならない。",
       "paragraphs": {
           3: "次に掲げる労働者の有給休暇の日数については、比例して付与する。",
           5: "使用者は、前各項の規定による有給休暇を労働者の請求する時季に"
              "与えなければならない。"}}
QUOTE39 = "「前各項の規定による有給休暇を労働者の請求する時季に与えなければならない」[1] "


def test_verbatim_quote_passes():
    q = check_quotes("「賃金は、通貨で、直接労働者に、その全額を支払わなければならない」[1]", CH)
    assert (q["quotes"], q["quote_bad"]) == (1, [])


def test_spaces_do_not_break_matching():
    assert check_quotes("「賃金は、通貨で、 直接労働者に」[1]", CH)["quote_bad"] == []


def test_ellipsis_fragments_in_order():
    q = check_quotes("「賃金は、通貨で、…その全額を支払わなければならない」[1]", CH)
    assert q["quote_bad"] == []


def test_spliced_exceptions_are_found_nowhere():
    q = check_quotes("「法令若しくは労働協約に別段の定めがある場合においては、"
                     "賃金の一部を控除して支払うことができる」[1]", CH)
    assert [b["found_in"] for b in q["quote_bad"]] == [[]]


def test_wrong_number_points_to_the_real_source():
    """Text of article [2] quoted under number [3]."""
    q = check_quotes("「解雇その他不利益な取扱いをしてはならない」[3]", CH)
    assert [b["found_in"] for b in q["quote_bad"]] == [[2]]


def test_number_outside_quotes_is_caught():
    q = check_quotes("入社後6か月で10日付与される。「賃金の一部を控除して支払うことができる」[1]", CH)
    assert q["num_unquoted"] == ["6か月", "10日"]


def test_article_number_is_not_a_loose_number():
    assert check_quotes("[1] 労働基準法 第二十四条によると「賃金は、通貨で」[1]", CH)["num_unquoted"] == []


def test_short_brackets_are_a_term_not_a_quote():
    q = check_quotes("「労働者」の場合", CH)
    assert (q["quotes"], q["quote_loose"]) == (0, [])


def test_splice_outside_quotes_lowers_coverage():
    """The first real answer with the quote prompt: the undisputed sentence
    quoted, the exceptions paraphrased. Verbatim-ness says 1/1; coverage must say 1/3."""
    q = check_quotes(
        "賃金は、原則として「通貨で、直接労働者に、その全額を支払わなければならない」[1] "
        "労働基準法 第二十四条。ただし、法令や労働協約などに基づき、賃金の一部を控除できる"
        "場合がある[1] 労働基準法 第二十四条。具体的には、法令に別段の定めがある場合"
        "や、書面による協定がある場合に限り控除が可能である[1] 労働基準法 第二十四条。",
        CH)
    assert (q["quotes"] - len(q["quote_bad"]), q["claims_quoted"], q["claims"]) == (1, 1, 3)


def test_cited_paragraph_must_hold_the_quote():
    """A real case from earlier runs: 時季変更権 put under 第3項; it is in 第5項."""
    q = check_quotes(QUOTE39 + "労働基準法 第三十九条第3項", [A39])
    assert [(b["cited"], b["found"]) for b in q["para_bad"]] == [([3], [5])]


def test_correct_paragraph_in_kanji_passes():
    q = check_quotes(QUOTE39 + "労働基準法 第三十九条第五項", [A39])
    assert (q["para_claims"], q["para_bad"]) == (1, [])


def test_paragraph_inside_a_quote_is_not_a_citation():
    """第N項 inside a quote is the statute's own wording, not the answer's source."""
    assert check_quotes("「第三項の規定による」[1] 労働基準法 第三十九条", [A39])["para_claims"] == 0
