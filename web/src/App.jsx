import { useEffect, useRef, useState } from "react";
import { streamAsk } from "./api";
import { T } from "./i18n";

const EXAMPLES = [
  "有給って入社してすぐもらえるの？",
  "残業したら何割増しになる？",
  "パワハラ対策は会社の義務？",
  "バイトでも有給ある？",
];

// The verdict arrives as the first line of the answer. It is shown as a badge
// and removed from the text — otherwise it reads as part of the answer.
function splitVerdict(text) {
  const m = text.match(/^判定[:：]\s*(回答可能|回答不可)\s*/);
  if (!m) return { verdict: null, body: text };
  return { verdict: m[1], body: text.slice(m[0].length) };
}

// [N] markers are clickable: they tie a statement to its source.
function Answer({ text, onPick, active }) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <p className="answer">
      {parts.map((p, i) => {
        const m = p.match(/^\[(\d+)\]$/);
        if (!m) return <span key={i}>{p}</span>;
        const n = Number(m[1]);
        return (
          <button
            key={i}
            className={"marker" + (active === n ? " on" : "")}
            onClick={() => onPick(active === n ? null : n)}
          >
            {n}
          </button>
        );
      })}
    </p>
  );
}

function Source({ s, used, active, onPick, t }) {
  const [open, setOpen] = useState(false);
  return (
    <div
      className={
        "source" + (used ? " used" : " unused") + (active ? " active" : "")
      }
      onMouseEnter={() => onPick(s.n)}
      onMouseLeave={() => onPick(null)}
    >
      <div className="src-head">
        <span className="num">{s.n}</span>
        <span className="score">{s.score.toFixed(3)}</span>
        <span className="law">{s.law}</span>
        <b>{s.article}</b>
        <span className="caption">{s.caption}</span>
      </div>
      <div className="src-meta">
        {used ? t.used : t.unused}
        {s.rank_before ? ` · ${t.beforeRerank}: ${s.rank_before}` : ""}
      </div>
      <button className="toggle" onClick={() => setOpen(!open)}>
        {open ? t.hide : t.show}
      </button>
      {open && <pre className="src-text">{s.text}</pre>}
    </div>
  );
}

export default function App() {
  const [lang, setLang] = useState(
    () => localStorage.getItem("lang") || "ja"
  );
  const t = T[lang];

  const [q, setQ] = useState(EXAMPLES[0]);
  const [raw, setRaw] = useState("");
  const [sources, setSources] = useState([]);
  const [meta, setMeta] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [active, setActive] = useState(null);
  const [ms, setMs] = useState(null);
  // How many requests are ahead in the server queue; null — not queued.
  const [ahead, setAhead] = useState(null);
  const abort = useRef(null);

  useEffect(() => localStorage.setItem("lang", lang), [lang]);
  useEffect(() => () => abort.current?.abort(), []);

  async function run(question) {
    if (!question.trim() || busy) return;
    abort.current?.abort();
    abort.current = new AbortController();
    setBusy(true);
    setErr(null);
    setRaw("");
    setSources([]);
    setMeta(null);
    setActive(null);
    setAhead(null);
    const t0 = performance.now();
    try {
      await streamAsk(
        question,
        {
          queued: (d) => setAhead(d.ahead),
          sources: (d) => {
            setAhead(null);
            setSources(d.sources);
          },
          delta: (d) => setRaw((prev) => prev + d.text),
          done: (d) => {
            setMeta(d);
            setMs(Math.round(performance.now() - t0));
          },
          // The server sends a code, not the exception text: the text is translated here.
          error: (d) => {
            setAhead(null);
            setErr(t.errors[d.code] ?? d.message ?? d.code);
          },
        },
        abort.current.signal
      );
    } catch (e) {
      if (e.name !== "AbortError") setErr(String(e));
    } finally {
      setAhead(null);
      setBusy(false);
    }
  }

  const { verdict, body } = splitVerdict(raw);
  const used = new Set(meta?.used ?? []);

  return (
    <div className="wrap">
      <header>
        <div className="top">
          <h1>日本の労働法令 RAG</h1>
          <div className="lang">
            {["ja", "ru"].map((l) => (
              <button
                key={l}
                className={lang === l ? "on" : ""}
                onClick={() => setLang(l)}
              >
                {l === "ja" ? "日本語" : "RU"}
              </button>
            ))}
          </div>
        </div>
        <p className="sub">{t.sub}</p>
      </header>

      <div className="ask">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run(q)}
          placeholder={t.placeholder}
        />
        <button onClick={() => run(q)} disabled={busy}>
          {busy ? "…" : t.ask}
        </button>
      </div>

      <div className="examples">
        {EXAMPLES.map((e) => (
          <button key={e} onClick={() => { setQ(e); run(e); }}>
            {e}
          </button>
        ))}
      </div>

      {err && <div className="err">{err}</div>}

      <div className="cols">
        <section className="left">
          {verdict && (
            <div className={"verdict " + (verdict === "回答可能" ? "ok" : "no")}>
              {verdict === "回答可能" ? t.verdictOk : t.verdictNo}
            </div>
          )}
          {body && <Answer text={body} onPick={setActive} active={active} />}
          {busy && !body && (
            <div className="hint">
              {ahead ? `${t.queued}: ${ahead}` : t.working}
            </div>
          )}

          {meta && (
            <div className="panel">
              <div>
                {t.submitted}: <b>{meta.sources.length}</b> · {t.usedCount}{" "}
                <b>{meta.used.length}</b> · {t.unusedCount}{" "}
                <b>{meta.unused.length}</b>
              </div>
              <div>
                {t.cited}: {meta.cited_articles.join(" · ") || "—"}
              </div>
              {meta.transitive.length > 0 && (
                <div>
                  {t.transitive}: {meta.transitive.join(" · ")}
                </div>
              )}
              {meta.fabricated.length > 0 && (
                <div className="bad">
                  {t.fabricated}: {meta.fabricated.join(" · ")}
                </div>
              )}
              {ms && (
                <div className="dim">
                  {(ms / 1000).toFixed(1)} {t.sec}
                </div>
              )}
            </div>
          )}
        </section>

        <aside className="right">
          {sources.length > 0 && (
            <h2>
              {t.found} <span className="dim">({sources.length})</span>
            </h2>
          )}
          {sources.map((s) => (
            <Source
              key={s.n}
              s={s}
              used={meta ? used.has(s.n) : null}
              active={active === s.n}
              onPick={setActive}
              t={t}
            />
          ))}
        </aside>
      </div>
    </div>
  );
}
