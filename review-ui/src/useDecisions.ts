import { useCallback, useEffect, useState } from "react";
import type { Decisions, Verdict } from "./types";

const KEY = "so101-review-decisions";

/** Reviewer verdicts, kept in localStorage so a half-finished pass survives a reload.
 *  Every access is guarded: storage throws in private windows and during thumbnailing,
 *  and the page has to render correctly when it comes back empty. */
export function useDecisions() {
  const [decisions, setDecisions] = useState<Decisions>(() => {
    try {
      const raw = localStorage.getItem(KEY);
      return raw ? (JSON.parse(raw) as Decisions) : {};
    } catch {
      return {};
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(decisions));
    } catch {
      /* storage unavailable: decisions stay in memory for this session */
    }
  }, [decisions]);

  const set = useCallback((episode: number, verdict: Verdict | null) => {
    setDecisions((d) => {
      const next = { ...d };
      if (verdict === null) delete next[episode];
      else next[episode] = verdict;
      return next;
    });
  }, []);

  const reset = useCallback(() => setDecisions({}), []);
  return { decisions, set, reset };
}
