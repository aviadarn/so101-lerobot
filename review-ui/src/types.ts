export type Signal =
  | "duration" | "path_ratio" | "jerk" | "pause_frac"
  | "grip_reversals" | "grasp_phase" | "median_dev";

export interface Episode {
  episode: number;
  score: number;
  /** Worst-contributing signal, or "clean" when nothing exceeded the threshold. */
  reason: Signal | "clean";
  frames: number;
  grasp_frame: number;
  grasp_phase: number | null;
  raw: Record<string, number>;
  z: Record<string, number>;
  contrib: Record<string, number>;
  /** One array of values per joint, already time-normalised by the exporter. */
  traj: number[][];
  clips: Partial<Record<"overhead" | "wrist", string>>;
}

export interface Manifest {
  dataset: string;
  n_total: number;
  n_exported: number;
  joints: string[];
  weights: Record<string, number>;
  signals: Signal[];
  score_median: number;
  episodes: Episode[];
}

export type Verdict = "keep" | "discard";
export type Decisions = Record<number, Verdict>;

export const SIGNAL_HELP: Record<Signal | "clean", string> = {
  duration: "Took longer than the fleet median — hesitation or retries.",
  path_ratio: "Wandered: path length well above the direct home→grasp→release route.",
  jerk: "Jerky motion rather than a smooth reach.",
  pause_frac: "Stalled mid-episode with the arm near stationary.",
  grip_reversals: "The gripper changed direction more than a clean single close.",
  grasp_phase: "Grasped late in the episode — the approach took most of it.",
  median_dev: "Trajectory sits far from what the rest of the fleet did.",
  clean: "No signal exceeded the flagging threshold.",
};
