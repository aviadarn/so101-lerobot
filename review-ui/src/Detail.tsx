import { useEffect, useRef, useState } from "react";
import type { Episode, Manifest, Verdict } from "./types";
import { SIGNAL_HELP } from "./types";
import { TrajectoryChart } from "./TrajectoryChart";
import { SignalBars } from "./SignalBars";

const JOINT_COLOURS = ["#3b7dd8", "#2f9e6e", "#c9822f", "#9159c9", "#3aa8b8", "#c94f4f"];

export function Detail({
  ep,
  manifest,
  verdict,
  onVerdict,
}: {
  ep: Episode;
  manifest: Manifest;
  verdict: Verdict | undefined;
  onVerdict: (v: Verdict | null) => void;
}) {
  const overhead = useRef<HTMLVideoElement>(null);
  const wrist = useRef<HTMLVideoElement>(null);
  const [progress, setProgress] = useState(0);
  const [visible, setVisible] = useState<Set<number>>(
    () => new Set(manifest.joints.map((_, i) => i)),
  );

  // Both cameras are the same episode, so scrubbing one must move the other.
  useEffect(() => {
    setProgress(0);
    const a = overhead.current;
    const b = wrist.current;
    [a, b].forEach((v) => {
      if (v) {
        v.currentTime = 0;
        v.play().catch(() => {
          /* autoplay blocked: the reviewer can press play */
        });
      }
    });
  }, [ep.episode]);

  const onTime = () => {
    const v = overhead.current;
    if (!v || !v.duration || Number.isNaN(v.duration)) return;
    setProgress(v.currentTime / v.duration);
    const w = wrist.current;
    if (w && Math.abs(w.currentTime - v.currentTime) > 0.12) w.currentTime = v.currentTime;
  };

  const toggleJoint = (i: number) =>
    setVisible((s) => {
      const n = new Set(s);
      if (n.has(i)) n.delete(i);
      else n.add(i);
      return n;
    });

  return (
    <section className="detail">
      <header className="detail-head">
        <div>
          <h2>Episode {ep.episode}</h2>
          <p className="sub">
            {ep.frames} frames · score <b>{ep.score.toFixed(2)}</b>{" "}
            <span className="muted">(fleet median {manifest.score_median.toFixed(2)})</span>
          </p>
        </div>
        <div className="verdict-buttons">
          <button className={`btn keep${verdict === "keep" ? " on" : ""}`}
                  onClick={() => onVerdict(verdict === "keep" ? null : "keep")}>
            Keep <kbd>K</kbd>
          </button>
          <button className={`btn discard${verdict === "discard" ? " on" : ""}`}
                  onClick={() => onVerdict(verdict === "discard" ? null : "discard")}>
            Discard <kbd>D</kbd>
          </button>
        </div>
      </header>

      <p className={`reason reason-${ep.reason}`}>
        <b>{ep.reason === "clean" ? "No flag" : ep.reason.replace(/_/g, " ")}</b>
        {" — "}
        {SIGNAL_HELP[ep.reason]}
      </p>

      <div className="videos">
        {(["overhead", "wrist"] as const).map((cam) => {
          const src = ep.clips[cam];
          return (
            <figure key={cam}>
              {src ? (
                <video ref={cam === "overhead" ? overhead : wrist} src={src}
                       onTimeUpdate={cam === "overhead" ? onTime : undefined}
                       loop muted playsInline controls={cam === "overhead"} />
              ) : (
                <div className="no-clip">no {cam} clip exported</div>
              )}
              <figcaption>{cam}</figcaption>
            </figure>
          );
        })}
      </div>

      <TrajectoryChart ep={ep} progress={progress} visible={visible} />
      <div className="legend">
        {manifest.joints.map((j, i) => (
          <button key={j} className={`chip${visible.has(i) ? " on" : ""}`}
                  style={{ borderColor: JOINT_COLOURS[i] }} onClick={() => toggleJoint(i)}>
            <span className="dot" style={{ background: JOINT_COLOURS[i] }} />
            {j.replace(/_/g, " ")}
          </button>
        ))}
      </div>

      <h3 className="signals-head">Why this score</h3>
      <SignalBars ep={ep} manifest={manifest} />
    </section>
  );
}
