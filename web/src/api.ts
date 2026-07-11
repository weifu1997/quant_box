import type {
  AccountApplyResult,
  AccountFormData,
  AccountHoldingRow,
  AccountPreview,
  AccountWorkspaceData,
  DashboardJob,
  DashboardJobAction,
  DashboardJobsResponse,
  DashboardPrecheck,
  DashboardRunMode,
  DashboardSnapshot,
  DashboardWorkflow,
  ExecutionApplyResult,
  ExecutionFillRow,
  ExecutionPreview,
  ExecutionWorkspaceData,
  StockDetail
} from "./types";

export async function fetchAccountWorkspace(signal?: AbortSignal): Promise<AccountWorkspaceData> {
  const response = await fetch("/api/dashboard/account", { signal });
  if (!response.ok) throw new Error(await responseError(response, "账户工作区读取失败"));
  return (await response.json()) as AccountWorkspaceData;
}

export async function previewAccountUpdate(account: AccountFormData, holdings: AccountHoldingRow[], signal?: AbortSignal): Promise<AccountPreview> {
  const response = await fetch("/api/dashboard/account/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ account, holdings }),
    signal
  });
  if (!response.ok) throw new Error(await responseError(response, "账户与持仓校验失败"));
  return (await response.json()) as AccountPreview;
}

export async function applyAccountUpdate(account: AccountFormData, holdings: AccountHoldingRow[], signal?: AbortSignal): Promise<AccountApplyResult> {
  const response = await fetch("/api/dashboard/account/apply", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ account, holdings, confirm: true }),
    signal
  });
  if (!response.ok) throw new Error(await responseError(response, "账户与持仓保存失败"));
  return (await response.json()) as AccountApplyResult;
}

export async function fetchLatestDashboard(signal?: AbortSignal): Promise<DashboardSnapshot> {
  const response = await fetch("/api/dashboard/latest", { signal });
  if (!response.ok) {
    throw new Error(`仪表盘 API 读取失败：${response.status}`);
  }
  return (await response.json()) as DashboardSnapshot;
}

export async function fetchStockDetail(instrument: string, signal?: AbortSignal): Promise<StockDetail> {
  const response = await fetch(`/api/dashboard/stocks/${encodeURIComponent(instrument)}`, {
    cache: "no-store",
    signal
  });
  if (!response.ok) {
    throw new Error(await responseError(response, "股票行情读取失败"));
  }
  return (await response.json()) as StockDetail;
}

export function artifactUrl(id: string): string {
  return `/api/dashboard/artifacts/${encodeURIComponent(id)}`;
}

export async function fetchDashboardJobs(signal?: AbortSignal): Promise<DashboardJobsResponse> {
  const response = await fetch("/api/dashboard/jobs", { signal });
  if (!response.ok) {
    throw new Error(await responseError(response, "任务状态读取失败"));
  }
  return (await response.json()) as DashboardJobsResponse;
}

export async function fetchDashboardWorkflows(signal?: AbortSignal): Promise<DashboardWorkflow[]> {
  const response = await fetch("/api/dashboard/workflows", { signal });
  if (!response.ok) {
    throw new Error(await responseError(response, "工作流目录读取失败"));
  }
  const body = (await response.json()) as { workflows: DashboardWorkflow[] };
  return body.workflows;
}

export async function fetchDashboardPrecheck(signal?: AbortSignal): Promise<DashboardPrecheck> {
  const response = await fetch("/api/dashboard/precheck", { signal });
  if (!response.ok) {
    throw new Error(await responseError(response, "运行前预检查读取失败"));
  }
  return (await response.json()) as DashboardPrecheck;
}

export async function startDashboardJob(
  payload: { action: DashboardJobAction; mode?: DashboardRunMode; parameters?: Record<string, unknown> },
  signal?: AbortSignal
): Promise<DashboardJob> {
  const response = await fetch("/api/dashboard/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal
  });
  if (!response.ok) {
    throw new Error(await responseError(response, "任务启动失败"));
  }
  const body = (await response.json()) as { job: DashboardJob };
  return body.job;
}

export async function stopDashboardJob(id: string, signal?: AbortSignal): Promise<DashboardJob> {
  const response = await fetch(`/api/dashboard/jobs/${encodeURIComponent(id)}/stop`, {
    method: "POST",
    signal
  });
  if (!response.ok) {
    throw new Error(await responseError(response, "任务停止失败"));
  }
  const body = (await response.json()) as { job: DashboardJob };
  return body.job;
}

export async function fetchExecutionWorkspace(signal?: AbortSignal): Promise<ExecutionWorkspaceData> {
  const response = await fetch("/api/dashboard/execution", { signal });
  if (!response.ok) {
    throw new Error(await responseError(response, "交易执行工作区读取失败"));
  }
  return (await response.json()) as ExecutionWorkspaceData;
}

export async function previewExecutionFeedback(
  sourceId: string,
  rows: ExecutionFillRow[],
  signal?: AbortSignal
): Promise<ExecutionPreview> {
  const response = await fetch("/api/dashboard/execution/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_id: sourceId, rows }),
    signal
  });
  if (!response.ok) {
    throw new Error(await responseError(response, "成交回填校验失败"));
  }
  return (await response.json()) as ExecutionPreview;
}

export async function applyExecutionFeedback(
  sourceId: string,
  rows: ExecutionFillRow[],
  signal?: AbortSignal
): Promise<ExecutionApplyResult> {
  const response = await fetch("/api/dashboard/execution/apply", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source_id: sourceId, rows, confirm: true }),
    signal
  });
  if (!response.ok) {
    throw new Error(await responseError(response, "真实持仓更新失败"));
  }
  return (await response.json()) as ExecutionApplyResult;
}

async function responseError(response: Response, fallback: string) {
  try {
    const body = (await response.json()) as { detail?: string };
    if (body.detail) {
      return body.detail;
    }
  } catch {
    // Keep the fallback below when the response is not JSON.
  }
  return `${fallback}: ${response.status}`;
}
