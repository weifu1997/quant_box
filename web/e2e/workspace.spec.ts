import { expect, test, type Page } from "@playwright/test";

const snapshot = {
  version: 1,
  output_dir: "outputs",
  readiness: { status: "ready", label: "可以复核", summary: "全部质量门槛通过。", is_executable: true },
  latest_run: { generated_at: "2026-07-11T09:00:00", status: "complete", strategy_mode: "annual_state_router", signal_date: "2026-07-10", intended_trade_date: "2026-07-11" },
  gates: [{ id: "data_health", label: "数据健康", status: "pass", summary: "数据完整", issues: [], details: {} }],
  block_reasons: [],
  blocker_actions: [],
  quality_warnings: [],
  freshness_notes: [],
  signal_summary: { BUY: 1, HOLD: 1, SELL: 1 },
  orders: { path: "outputs/manual_orders_2026-07-10.csv", exists: true, valid: true, columns: ["instrument", "name", "action", "order_shares"], rows: [{ instrument: "000001.SZ", name: "平安银行", action: "BUY", order_shares: "200" }], total_rows: 1, preview_limit: 20, action_counts: { BUY: 1 }, actionable_count: 1 },
  artifacts: [{ id: "daily_report", label: "每日报告", kind: "markdown", path: "outputs/daily_signal_report.md", exists: true, downloadable: true }],
  report: { mode: "structured", daily_markdown: { id: "daily_report", label: "每日报告", kind: "markdown", path: "outputs/daily_signal_report.md", exists: true, downloadable: true }, summary: [{ label: "is_executable", value: true }] },
  errors: []
};

const precheck = { version: 1, generated_at: "2026-07-11T09:00:00", status: "pass", summary: "运行前检查通过", can_run_normal: true, target_date_resolution: {}, items: [] };
const workflows = [{ action: "update_market_data", label: "增量更新行情", category: "data", description: "补齐行情数据", duration: "5–60 分钟", parameters: [{ name: "chunk_size", label: "每批股票数", type: "integer", default: 300, min: 1, max: 2000 }, { name: "include_existing", label: "同时刷新已有股票", type: "boolean", default: false }] }, { action: "run_backtest", label: "运行真实化回测", category: "research", description: "运行当前策略回测", duration: "2–20 分钟", parameters: [] }];
const execution = { version: 1, status: "needs_input", message: "请填写成交结果", source_id: "fill_feedback_2026-07-10.csv", source_path: "outputs/fill_feedback/fill_feedback_2026-07-10.csv", signal_date: "2026-07-10", intended_trade_date: "2026-07-11", rows: [{ row_id: 0, signal_date: "2026-07-10", instrument: "000001.SZ", side: "BUY", planned_order_shares: 200, fill_status: "PENDING", actual_trade_date: "2026-07-11", executed_shares: null, executed_price: null, commission_cost: 0, fill_note: "" }], holdings: [{ instrument: "000001.SZ", shares: 100 }], editable_fields: [], issues: ["pending_fill_status:000001.SZ"], pending_count: 1 };
const account = { version: 1, status: "ready", message: "账户与持仓校验通过", account: { total_asset: 1_000_000, cash: 100_000, max_position_pct: 0.2, lot_size: 100, star_market_lot_size: 200 }, holdings: [{ instrument: "000001.SZ", shares: 100 }], issues: [], account_file: "config/account.yaml", holdings_file: "config/current_holdings.csv", account_file_exists: true, holdings_file_exists: true };

test.beforeEach(async ({ page }) => {
  await installApiMocks(page);
});

test("daily review and project overview navigation remain usable", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "自动信号复核" })).toBeVisible();
  await expect(page.getByText("可以进入人工复核")).toBeVisible();
  await expect(page.getByRole("link", { name: "运行控制" })).toBeVisible();

  await page.getByRole("button", { name: "打开项目全景" }).click();
  await expect(page.getByRole("heading", { name: "项目全景", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "它解决什么问题" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "每日信号与人工交易闭环" })).toBeVisible();

  await page.getByRole("button", { name: "进入复核台" }).click();
  await expect(page.getByRole("heading", { name: "自动信号复核" })).toBeVisible();
});

test("operations workflow run and stop buttons match backend payloads", async ({ page }) => {
  let startedAction = "";
  let startedPayload: any = null;
  let stoppedJob = "";
  let currentJob: ReturnType<typeof runningJob> | null = null;
  await page.route("**/api/dashboard/jobs", async (route) => {
    if (route.request().method() === "POST") {
      const body = route.request().postDataJSON();
      startedPayload = body;
      startedAction = body.action;
      currentJob = runningJob(body.action);
      return route.fulfill({ json: { job: currentJob } });
    }
    return route.fulfill({ json: { jobs: currentJob ? [currentJob] : [], active_job: currentJob } });
  });
  await page.route("**/api/dashboard/jobs/*/stop", async (route) => {
    stoppedJob = route.request().url().split("/").slice(-2)[0];
    currentJob = null;
    return route.fulfill({ json: { job: { ...runningJob("update_market_data"), id: stoppedJob, status: "cancelled", message: "任务已停止" } } });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "打开量化操作中心" }).click();
  await expect(page.getByRole("heading", { name: "量化操作中心" })).toBeVisible();
  const card = page.locator(".workflow-card").filter({ hasText: "增量更新行情" });
  await card.getByRole("button", { name: /配置/ }).click();
  await card.getByLabel("每批股票数").fill("500");
  await card.getByText("同时刷新已有股票").click();
  await card.getByRole("button", { name: "运行" }).click();
  await expect.poll(() => startedAction).toBe("update_market_data");
  expect(startedPayload.parameters.chunk_size).toBe("500");
  expect(startedPayload.parameters.include_existing).toBe(true);
  await expect(page.getByText("增量更新行情", { exact: true }).last()).toBeVisible();
  await page.getByRole("button", { name: "停止任务" }).click();
  await expect.poll(() => stoppedJob).toBe("job-1");
});

test("execution preview and apply preserve the backend contract", async ({ page }) => {
  let previewPayload: Record<string, unknown> | null = null;
  let applyPayload: Record<string, unknown> | null = null;
  await page.route("**/api/dashboard/execution/preview", async (route) => {
    previewPayload = route.request().postDataJSON();
    return route.fulfill({ json: { valid: true, issues: [], source_id: execution.source_id, current_holdings: execution.holdings, updated_holdings: [{ instrument: "000001.SZ", shares: 300 }], summary: { fill_rows: 1, applied_fill_rows: 1, executed_shares: 200, fill_status_counts: { FILLED: 1 } } } });
  });
  await page.route("**/api/dashboard/execution/apply", async (route) => {
    applyPayload = route.request().postDataJSON();
    return route.fulfill({ json: { status: "applied", message: "成交回填已保存，真实持仓已更新。", source_id: execution.source_id, holdings_path: "config/current_holdings.csv", audit_path: "outputs/fill_apply_audit.json", holdings: [{ instrument: "000001.SZ", shares: 300 }], summary: { fill_rows: 1, applied_fill_rows: 1, executed_shares: 200, fill_status_counts: { FILLED: 1 } } } });
  });
  page.on("dialog", (dialog) => dialog.accept());

  await page.goto("/");
  await page.getByRole("button", { name: "打开交易执行工作区" }).click();
  const row = page.locator(".execution-table tbody tr").first();
  await row.locator("select").selectOption("FILLED");
  await row.locator('input[type="number"]').nth(0).fill("200");
  await row.locator('input[type="number"]').nth(1).fill("12.5");
  await page.getByRole("button", { name: "预览持仓变化" }).click();
  await expect(page.getByText("校验通过", { exact: true })).toBeVisible();
  expect((previewPayload as any).source_id).toBe(execution.source_id);
  expect((previewPayload as any).rows[0].executed_shares).toBe("200");
  await page.getByRole("button", { name: "确认更新真实持仓" }).click();
  await expect.poll(() => Boolean(applyPayload)).toBe(true);
  expect((applyPayload as any).confirm).toBe(true);
});

test("account preview, add holding, and confirmed save are wired", async ({ page }) => {
  let previewPayload: any = null;
  let applyPayload: any = null;
  await page.route("**/api/dashboard/account/preview", async (route) => {
    previewPayload = route.request().postDataJSON();
    return route.fulfill({ json: { valid: true, issues: [], account: previewPayload.account, holdings: previewPayload.holdings, position_count: previewPayload.holdings.length, holding_shares: 300 } });
  });
  await page.route("**/api/dashboard/account/apply", async (route) => {
    applyPayload = route.request().postDataJSON();
    return route.fulfill({ json: { status: "applied", message: "账户与真实持仓已保存。", account_file: "config/account.yaml", holdings_file: "config/current_holdings.csv", backup_dir: "outputs/account_backups/test", account: applyPayload.account, holdings: applyPayload.holdings } });
  });
  page.on("dialog", (dialog) => dialog.accept());

  await page.goto("/");
  await page.getByRole("button", { name: "打开账户与持仓" }).click();
  await page.getByLabel("总资产").fill("1200000");
  await page.getByRole("button", { name: "新增持仓" }).click();
  await page.getByLabel("第 2 行股票代码").fill("600519.SH");
  await page.getByLabel("第 2 行持仓股数").fill("200");
  await page.getByRole("button", { name: "校验账户与持仓" }).click();
  await expect(page.getByText("校验通过", { exact: true })).toBeVisible();
  expect(previewPayload.account.total_asset).toBe("1200000");
  expect(previewPayload.holdings).toHaveLength(2);
  await page.getByRole("button", { name: "确认保存" }).click();
  await expect.poll(() => Boolean(applyPayload)).toBe(true);
  expect(applyPayload.confirm).toBe(true);
});

test("mobile layout keeps navigation and content inside viewport", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByRole("button", { name: "打开项目全景" }).click();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(1);
  await expect(page.getByRole("button", { name: "打开每日复核台" }).first()).toBeVisible();
});

async function installApiMocks(page: Page) {
  await page.route("**/api/dashboard/latest", (route) => route.fulfill({ json: snapshot }));
  await page.route("**/api/dashboard/precheck", (route) => route.fulfill({ json: precheck }));
  await page.route("**/api/dashboard/jobs", (route) => route.fulfill({ json: { jobs: [], active_job: null } }));
  await page.route("**/api/dashboard/workflows", (route) => route.fulfill({ json: { workflows } }));
  await page.route("**/api/dashboard/execution", (route) => route.fulfill({ json: execution }));
  await page.route("**/api/dashboard/account", (route) => route.fulfill({ json: account }));
}

function runningJob(action: string) {
  return { version: 1, id: "job-1", action, label: action === "update_market_data" ? "增量更新行情" : action, status: "running", message: "任务已启动", command: ["python", "script.py"], started_at: "2026-07-11T09:00:00", completed_at: null, return_code: null, log_path: "outputs/job.log", log_tail: ["starting"], progress: { summary: "运行中", percent: 10, active_step: null, steps: [] } };
}
