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
  blocker_actions: BlockerAction[];
  quality_warnings: string[];
  freshness_notes: string[];
  signal_summary: Record<string, number>;
  orders: Orders;
  artifacts: Artifact[];
  report: ReportSection;
  errors: string[];
}

export type DashboardJobAction = "repair_point_in_time" | "run_auto_signal";
export type DashboardRunMode = "candidate" | "normal";
export type DashboardJobStatus = "running" | "stopping" | "succeeded" | "failed" | "stale" | "cancelled";

export interface BlockerAction {
  id: string;
  source: "block_reason" | "freshness_note" | string;
  reason: string;
  issue: string;
  title: string;
  detail: string;
  severity: "danger" | "warning" | "info" | "hold" | string;
  action?: {
    label: string;
    action: DashboardJobAction;
    mode?: DashboardRunMode | null;
  } | null;
}

export interface JobProgressStep {
  id: string;
  label: string;
  status: "pending" | "running" | "complete" | "failed" | "skipped" | string;
  message?: string | null;
  updated_at?: string | null;
}

export interface JobProgress {
  summary: string;
  percent: number;
  active_step?: string | null;
  steps: JobProgressStep[];
}

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
  progress?: JobProgress;
  pid?: number;
}

export interface DashboardJobsResponse {
  jobs: DashboardJob[];
  active_job?: DashboardJob | null;
}

export type PrecheckStatus = "pass" | "warn" | "fail" | "missing";

export interface DashboardPrecheckItem {
  id: string;
  label: string;
  status: PrecheckStatus | string;
  summary: string;
  issues: string[];
  details: Record<string, unknown>;
  action?: {
    label: string;
    action: DashboardJobAction;
    mode?: DashboardRunMode | null;
  } | null;
}

export interface DashboardPrecheck {
  version: number;
  generated_at: string;
  status: "pass" | "warn" | "fail" | string;
  summary: string;
  can_run_normal: boolean;
  target_date_resolution: Record<string, unknown>;
  items: DashboardPrecheckItem[];
}
