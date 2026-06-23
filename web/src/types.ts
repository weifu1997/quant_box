export type ReadinessStatus = "ready" | "blocked" | "candidate_only" | "missing" | "error";
export type GateStatus = "pass" | "fail" | "warn" | "missing" | "hold";

export interface Readiness {
  status: ReadinessStatus;
  label: string;
  summary: string;
  is_executable: boolean | null;
}

export interface LatestRun {
  generated_at?: string | null;
  status?: string | null;
  strategy_mode?: string | null;
  signal_date?: string | null;
  intended_trade_date?: string | null;
  requested_date?: string | null;
  target_date?: string | null;
  latest_trade_date?: string | null;
  latest_stage?: {
    name?: string;
    state?: string;
    updated_at?: string;
    message?: string;
  } | null;
}

export interface Gate {
  id: string;
  label: string;
  status: GateStatus;
  summary: string;
  issues: string[];
  details: Record<string, unknown>;
}

export interface Orders {
  path: string;
  exists: boolean;
  valid: boolean;
  columns: string[];
  rows: Record<string, string>[];
  total_rows: number;
  preview_limit: number;
  action_counts: Record<string, number>;
  actionable_count: number;
  error?: string | null;
}

export interface Artifact {
  id: string;
  label: string;
  kind: string;
  path: string;
  exists: boolean;
  downloadable: boolean;
}

export interface ReportSection {
  mode: string;
  daily_markdown: Artifact;
  summary: Array<{ label: string; value: string | number | null }>;
}

export interface DashboardSnapshot {
  version: number;
  output_dir: string;
  readiness: Readiness;
  latest_run: LatestRun;
  gates: Gate[];
  block_reasons: string[];
  quality_warnings: string[];
  signal_summary: Record<string, number>;
  orders: Orders;
  artifacts: Artifact[];
  report: ReportSection;
  errors: string[];
}

export type DashboardJobAction = "repair_point_in_time" | "run_auto_signal";
export type DashboardRunMode = "candidate" | "normal";
export type DashboardJobStatus = "running" | "succeeded" | "failed" | "stale";

export interface DashboardJob {
  version: number;
  id: string;
  action: DashboardJobAction;
  mode?: DashboardRunMode | null;
  label: string;
  status: DashboardJobStatus | string;
  message: string;
  command: string[];
  started_at: string;
  completed_at?: string | null;
  return_code?: number | null;
  log_path: string;
  log_tail: string[];
  pid?: number;
}

export interface DashboardJobsResponse {
  jobs: DashboardJob[];
  active_job?: DashboardJob | null;
}
