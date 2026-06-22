import type { DashboardSnapshot } from "./types";

export async function fetchLatestDashboard(signal?: AbortSignal): Promise<DashboardSnapshot> {
  const response = await fetch("/api/dashboard/latest", { signal });
  if (!response.ok) {
    throw new Error(`Dashboard API failed: ${response.status}`);
  }
  return (await response.json()) as DashboardSnapshot;
}

export function artifactUrl(id: string): string {
  return `/api/dashboard/artifacts/${encodeURIComponent(id)}`;
}
