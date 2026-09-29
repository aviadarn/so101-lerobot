import { useMemo } from "react";
import type { Episode } from "./types";

const COLOURS = ["#3b7dd8", "#2f9e6e", "#c9822f", "#9159c9", "#3aa8b8", "#c94f4f"];
const W = 560;
const H = 190;
const PAD = { l: 34, r: 8, t: 10, b: 18 };

/** Six joint channels, normalised per channel so shape is comparable across joints, with a
 *  playhead driven by the video's current time and a marker at the grasp. Hand-drawn SVG
 *  rather than a charting dependency: it is 40 lines and the alternative is 300 kB. */
export function TrajectoryChart({
  ep,
  progress,
  visible,
}: {
  ep: Episode;
  progress: number;
  visible: Set<number>;
}) {
  const paths = useMemo(() => {
    const iw = W - PAD.l - PAD.r;
    const ih = H - PAD.t - PAD.b;
    return ep.traj.map((series) => {
      const lo = Math.min(...series);
      const hi = Math.max(...series);
      const span = hi - lo || 1;
      return series
        .map((v, i) => {
          const x = PAD.l + (i / Math.max(series.length - 1, 1)) * iw;
          const y = PAD.t + ih - ((v - lo) / span) * ih;
          return `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
        })
        .join(" ");
    });
  }, [ep]);

  const playX = PAD.l + progress * (W - PAD.l - PAD.r);
  const graspX =
    ep.grasp_phase === null ? null : PAD.l + ep.grasp_phase * (W - PAD.l - PAD.r);

  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img"
         aria-label="joint trajectories over the episode">
      <rect x={PAD.l} y={PAD.t} width={W - PAD.l - PAD.r} height={H - PAD.t - PAD.b}
            className="chart-bg" />
      {graspX !== null && (
        <g>
          <line x1={graspX} x2={graspX} y1={PAD.t} y2={H - PAD.b} className="grasp-line" />
          <text x={graspX + 3} y={PAD.t + 10} className="chart-label">grasp</text>
        </g>
      )}
      {paths.map((d, i) =>
        visible.has(i) ? (
          <path key={i} d={d} fill="none" stroke={COLOURS[i] ?? "#888"} strokeWidth="1.6" />
        ) : null,
      )}
      <line x1={playX} x2={playX} y1={PAD.t} y2={H - PAD.b} className="play-line" />
      <text x={2} y={PAD.t + 8} className="chart-label">max</text>
      <text x={2} y={H - PAD.b} className="chart-label">min</text>
    </svg>
  );
}
