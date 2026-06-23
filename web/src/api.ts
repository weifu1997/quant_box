import type { DashboardJob, DashboardJobAction, DashboardJobsResponse, DashboardRunMode, DashboardSnapshot } from "./types";

export async function fetchLatestDashboard(signal?: AbortSignal): Promise<DashboardSnapshot> {
  const response = await fetch("/api/dashboard/latest", { signal });
  if (!response.ok) {
    throw new Error(`仪表盘 API 读取失败：${response.status}`);
  }
  return (await response.json()) as DashboardSnapshot;
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

export async function startDashboardJob(
  payload: { action: DashboardJobAction; mode?: DashboardRunMode },
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
