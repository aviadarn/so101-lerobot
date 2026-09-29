import type { Episode, Manifest, Signal } from "./types";
import { SIGNAL_HELP } from "./types";

/** What the score is made of, so a reviewer can disagree with it on evidence.
 *  Bars are the weighted contribution; the raw measurement is shown alongside because
 *  "2.4 sigma above the fleet" means nothing without "took 243 frames". */
export function SignalBars({ ep, manifest }: { ep: Episode; manifest: Manifest }) {
  const entries = manifest.signals
    .map((s) => ({ signal: s, contrib: ep.contrib[s] ?? 0, z: ep.z[s] ?? 0, raw: ep.raw[s] }))
    .sort((a, b) => b.contrib - a.contrib);
  const max = Math.max(...entries.map((e) => e.contrib), 1);

  return (
    <div className="signals">
      {entries.map((e) => {
        const dominant = e.signal === ep.reason;
        return (
          <div className={`signal${dominant ? " dominant" : ""}`} key={e.signal}
               title={SIGNAL_HELP[e.signal as Signal]}>
            <span className="signal-name">{e.signal.replace(/_/g, " ")}</span>
            <span className="signal-track">
              <span className="signal-fill" style={{ width: `${(e.contrib / max) * 100}%` }} />
            </span>
            <span className="signal-num">
              {e.contrib > 0.005 ? `+${e.contrib.toFixed(2)}` : "—"}
            </span>
            <span className="signal-raw">
              {e.raw === undefined ? "" : formatRaw(e.signal, e.raw)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function formatRaw(signal: string, v: number): string {
  if (signal === "duration") return `${v.toFixed(0)} frames`;
  if (signal === "grip_reversals") return `${v.toFixed(0)} reversals`;
  if (signal === "pause_frac" || signal === "grasp_phase") return `${(v * 100).toFixed(0)}%`;
  return v.toFixed(2);
}
