import { useEffect, useMemo, useState } from "react";
import type { Episode, Manifest } from "./types";
import { useDecisions } from "./useDecisions";
import { Detail } from "./Detail";

type Sort = "score-desc" | "score-asc" | "episode";
type Filter = "all" | "undecided" | "flagged" | "clean";

export default function App() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [sort, setSort] = useState<Sort>("score-desc");
  const [filter, setFilter] = useState<Filter>("all");
  const { decisions, set, reset } = useDecisions();

  useEffect(() => {
    fetch("review.json")
      .then((r) => {
        if (!r.ok) throw new Error(`review.json: ${r.status}`);
        return r.json();
      })
      // Deliberately do not seed the selection: `current` falls back to the first row of
      // the active sort, so the page opens on the worst-scoring episode rather than
      // whichever one happens to be first in the file.
      .then((m: Manifest) => setManifest(m))
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const shown = useMemo(() => {
    if (!manifest) return [];
    let list = manifest.episodes.slice();
    if (filter === "undecided") list = list.filter((e) => !decisions[e.episode]);
    if (filter === "flagged") list = list.filter((e) => e.reason !== "clean");
    if (filter === "clean") list = list.filter((e) => e.reason === "clean");
    list.sort((a, b) =>
      sort === "episode" ? a.episode - b.episode
        : sort === "score-asc" ? a.score - b.score
        : b.score - a.score,
    );
    return list;
  }, [manifest, sort, filter, decisions]);

  const current: Episode | undefined =
    shown.find((e) => e.episode === selected) ?? shown[0];

  // Keyboard review: verdict on K/D, move with J/K-style arrows. A reviewer going through
  // 600 episodes should never need the mouse.
  useEffect(() => {
    const step = (d: number) => {
      if (!current) return;
      const i = shown.findIndex((e) => e.episode === current.episode);
      const next = shown[Math.min(Math.max(i + d, 0), shown.length - 1)];
      if (next) setSelected(next.episode);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.metaKey || e.ctrlKey) return;
      const k = e.key.toLowerCase();
      if (k === "k") { if (current) set(current.episode, decisions[current.episode] === "keep" ? null : "keep"); }
      else if (k === "d") { if (current) set(current.episode, decisions[current.episode] === "discard" ? null : "discard"); }
      else if (k === "arrowdown" || k === "j") { e.preventDefault(); step(1); }
      else if (k === "arrowup") { e.preventDefault(); step(-1); }
      else return;
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [current, shown, decisions, set]);

  if (error) {
    return (
      <div className="fatal">
        <h1>Could not load the review data</h1>
        <p><code>{error}</code></p>
        <p className="muted">
          Generate it with <code>tools/export_review.py</code>, then serve this directory.
        </p>
      </div>
    );
  }
  if (!manifest || !current) return <div className="loading">Loading review data…</div>;

  const kept = Object.values(decisions).filter((v) => v === "keep").length;
  const discarded = Object.values(decisions).filter((v) => v === "discard").length;
  const done = kept + discarded;

  const exportDecisions = () => {
    const keep = manifest.episodes
      .filter((e) => decisions[e.episode] === "keep")
      .map((e) => e.episode);
    const discard = manifest.episodes
      .filter((e) => decisions[e.episode] === "discard")
      .map((e) => e.episode);
    const blob = new Blob(
      [JSON.stringify({ dataset: manifest.dataset, keep, discard }, null, 2)],
      { type: "application/json" },
    );
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "review_decisions.json";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>Demonstration review</h1>
          <p className="sub">
            {manifest.dataset} · {manifest.n_exported} of {manifest.n_total} episodes exported
            for review
          </p>
        </div>
        <div className="top-right">
          <span className="counter">
            <b>{done}</b>/{shown.length} reviewed · {kept} keep · {discarded} discard
          </span>
          <button className="btn ghost" onClick={exportDecisions} disabled={done === 0}>
            Export decisions
          </button>
          <button className="btn ghost" onClick={reset} disabled={done === 0}>Reset</button>
        </div>
      </header>

      <div className="controls">
        <label>
          Sort
          <select value={sort} onChange={(e) => setSort(e.target.value as Sort)}>
            <option value="score-desc">worst score first</option>
            <option value="score-asc">best score first</option>
            <option value="episode">episode order</option>
          </select>
        </label>
        <label>
          Show
          <select value={filter} onChange={(e) => setFilter(e.target.value as Filter)}>
            <option value="all">all</option>
            <option value="undecided">undecided</option>
            <option value="flagged">flagged only</option>
            <option value="clean">clean only</option>
          </select>
        </label>
        <span className="hint">
          <kbd>K</kbd> keep · <kbd>D</kbd> discard · <kbd>↑</kbd><kbd>↓</kbd> move
        </span>
      </div>

      <div className="body">
        <ul className="list">
          {shown.map((e) => {
            const v = decisions[e.episode];
            return (
              <li key={e.episode}>
                <button
                  className={`row${e.episode === current.episode ? " sel" : ""}${v ? ` ${v}` : ""}`}
                  onClick={() => setSelected(e.episode)}
                >
                  <span className="row-ep">#{e.episode}</span>
                  <span className="row-score">{e.score.toFixed(2)}</span>
                  <span className={`row-reason reason-${e.reason}`}>
                    {e.reason === "clean" ? "clean" : e.reason.replace(/_/g, " ")}
                  </span>
                  <span className="row-verdict">
                    {v === "keep" ? "✓" : v === "discard" ? "✕" : ""}
                  </span>
                </button>
              </li>
            );
          })}
          {shown.length === 0 && <li className="empty">nothing matches this filter</li>}
        </ul>

        <Detail
          ep={current}
          manifest={manifest}
          verdict={decisions[current.episode]}
          onVerdict={(v) => set(current.episode, v)}
        />
      </div>
    </div>
  );
}
