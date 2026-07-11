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

export type StockQuoteStatus = "live" | "fallback";

export interface StockDetail {
  version: number;
  instrument: string;
  name: string;
  status: StockQuoteStatus;
  is_live: boolean;
  source: "tushare_rt_k" | "local_daily";
  price: number;
  change?: number | null;
  change_pct?: number | null;
  pre_close?: number | null;
  open?: number | null;
  high?: number | null;
  low?: number | null;
  volume?: number | null;
  amount?: number | null;
  market_date?: string | null;
  retrieved_at: string;
  message: string;
}

export type DashboardJobAction =
  | "repair_point_in_time"
  | "run_auto_signal"
  | "check_tushare_config"
  | "update_market_data"
  | "update_point_in_time_all"
  | "update_fundamentals"
  | "build_historical_universe"
  | "convert_data"
  | "calculate_factors"
  | "factor_diagnostics"
  | "optimize_parameters"
  | "run_backtest"
  | "quant_diagnostics"
  | "optimization_review"
  | "evidence_optimizer"
  | "fundamental_screen"
  | "generate_candidate_signal"
  | "risk_refine"
  | "regime_blend_probe"
  | "rebalance_drift_probe"
  | "annual_router_grid";
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
  report_artifact?: Pick<Artifact, "id" | "label" | "kind" | "exists" | "downloadable"> | null;
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
  parameters?: Record<string, unknown> | null;
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

export interface DashboardWorkflow {
  action: string;
  label: string;
  category: "data" | "pipeline" | "research" | "signal" | string;
  description: string;
  duration: string;
  parameters: WorkflowParameter[];
}

export interface WorkflowParameter {
  name: string;
  label: string;
  type: "boolean" | "integer" | "number" | "date" | "text" | string;
  default?: unknown;
  optional?: boolean;
  min?: number;
  max?: number;
  help?: string;
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

export type FillStatus = "PENDING" | "FILLED" | "PARTIAL" | "CANCELLED" | "SKIPPED";

export interface ExecutionFillRow {
  row_id: number;
  signal_date?: string | null;
  intended_trade_date?: string | null;
  instrument: string;
  side: string;
  planned_order_shares: number | string;
  fill_status: FillStatus | string;
  actual_trade_date?: string | null;
  executed_shares?: number | string | null;
  executed_price?: number | string | null;
  commission_cost?: number | string | null;
  tax_cost?: number | string | null;
  transfer_fee_cost?: number | string | null;
  slippage_note?: string | null;
  broker_order_id?: string | null;
  fill_note?: string | null;
  [key: string]: unknown;
}

export interface ExecutionHoldingRow {
  instrument: string;
  shares: number;
}

export interface ExecutionSummary {
  fill_rows: number;
  applied_fill_rows: number;
  executed_shares: number;
  fill_status_counts: Record<string, number>;
}

export interface ExecutionWorkspaceData {
  version: number;
  status: "missing" | "error" | "ready" | "needs_input" | string;
  message: string;
  source_id?: string | null;
  source_path?: string | null;
  signal_date?: string | null;
  intended_trade_date?: string | null;
  rows: ExecutionFillRow[];
  holdings: ExecutionHoldingRow[];
  editable_fields: string[];
  issues?: string[];
  pending_count?: number;
}

export interface ExecutionPreview {
  valid: boolean;
  issues: string[];
  source_id: string;
  current_holdings: ExecutionHoldingRow[];
  updated_holdings: ExecutionHoldingRow[];
  summary: ExecutionSummary;
}

export interface ExecutionApplyResult {
  status: "applied" | string;
  message: string;
  source_id: string;
  holdings_path: string;
  audit_path: string;
  holdings: ExecutionHoldingRow[];
  summary: ExecutionSummary;
}

export interface AccountFormData {
  total_asset: number | string;
  cash: number | string;
  max_position_pct?: number | string | null;
  lot_size: number | string;
  star_market_lot_size: number | string;
}

export interface AccountHoldingRow {
  instrument: string;
  shares: number | string;
}

export interface AccountWorkspaceData {
  version: number;
  status: "ready" | "needs_input" | string;
  message: string;
  account: AccountFormData;
  holdings: AccountHoldingRow[];
  issues: string[];
  account_file: string;
  holdings_file: string;
  account_file_exists: boolean;
  holdings_file_exists: boolean;
}

export interface AccountPreview {
  valid: boolean;
  issues: string[];
  account: AccountFormData;
  holdings: AccountHoldingRow[];
  position_count: number;
  holding_shares: number;
}

export interface AccountApplyResult {
  status: "applied" | string;
  message: string;
  account_file: string;
  holdings_file: string;
  backup_dir?: string | null;
  account: AccountFormData;
  holdings: AccountHoldingRow[];
}
