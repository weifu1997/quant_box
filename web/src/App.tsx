import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  CircleDot,
  ClipboardList,
  ExternalLink,
  FileText,
  Info,
  Loader2,
  Play,
  RefreshCw,
  ShieldAlert,
  ShoppingCart,
  Square,
  Table2,
  Terminal,
  TrendingUp,
  Wrench,
  XCircle
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { artifactUrl, fetchDashboardJobs, fetchLatestDashboard, startDashboardJob, stopDashboardJob } from "./api";
import type {
  Artifact,
  BlockerAction,
  DashboardJob,
  DashboardJobAction,
  DashboardRunMode,
  DashboardSnapshot,
  Gate,
  GateStatus,
  ReadinessStatus
} from "./types";

const ORDER_COLUMNS = [
  "instrument",
  "action",
  "is_order_actionable",
  "target_weight",
  "target_value",
  "final_target_shares",
  "order_shares",
  "reference_price",
  "suggested_limit_price",
  "note"
];

const ORDER_COLUMN_LABELS: Record<string, string> = {
  instrument: "股票代码",
  action: "动作",
  is_order_actionable: "可执行",
  target_weight: "目标权重",
  target_value: "目标金额",
  final_target_shares: "最终目标股数",
  order_shares: "下单股数",
  reference_price: "参考价",
  suggested_limit_price: "建议限价",
  note: "备注"
};

const ACTION_LABELS: Record<string, string> = {
  BUY: "买入",
  HOLD: "持有",
  SELL: "卖出"
};

const STATUS_LABELS: Record<string, string> = {
  blocked: "已阻塞",
  complete: "已完成",
  completed: "已完成",
  running: "运行中",
  error: "出错",
  skipped: "已跳过",
  planning: "规划中",
  in_progress: "进行中",
  stopping: "正在停止",
  succeeded: "已完成",
  failed: "失败",
  stale: "状态待确认",
  cancelled: "已停止"
};

const STRATEGY_LABELS: Record<string, string> = {
  annual_state_router: "年度状态路由",
  strategy_config: "普通策略配置"
};

export default function App() {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [jobs, setJobs] = useState<DashboardJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [refreshCount, setRefreshCount] = useState(0);
  const [jobsRefreshCount, setJobsRefreshCount] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetchLatestDashboard(controller.signal)
      .then((data) => setSnapshot(data))
      .catch((err: Error) => {
        if (err.name !== "AbortError") {
          setError(err.message);
        }
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [refreshCount]);

  useEffect(() => {
    const controller = new AbortController();
    fetchDashboardJobs(controller.signal)
      .then((data) => {
        setJobs(data.jobs);
        setJobsError(null);
      })
      .catch((err: Error) => {
        if (err.name !== "AbortError") {
          setJobsError(err.message);
        }
      });
    return () => controller.abort();
  }, [jobsRefreshCount]);

  const refresh = useCallback(() => setRefreshCount((value) => value + 1), []);
  const refreshJobs = useCallback(() => setJobsRefreshCount((value) => value + 1), []);
  const activeJob = useMemo(() => jobs.find(isActiveJob) ?? null, [jobs]);

  useEffect(() => {
    if (!activeJob) {
      return;
    }
    let stopped = false;
    const poll = () => {
      fetchDashboardJobs()
        .then((data) => {
          if (stopped) {
            return;
          }
          setJobs(data.jobs);
          setJobsError(null);
          if (!data.jobs.some(isActiveJob)) {
            refresh();
          }
        })
        .catch((err: Error) => {
          if (!stopped) {
            setJobsError(err.message);
          }
        });
    };
    poll();
    const timer = window.setInterval(poll, 2500);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [activeJob?.id, refresh]);

  const recordStartedJob = useCallback((job: DashboardJob) => {
    setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
    setJobsError(null);
  }, []);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-mark">Q</div>
        <nav className="side-nav" aria-label="仪表盘分区">
          <a href="#review" aria-label="复核结论">
            <ShieldAlert size={20} />
          </a>
          <a href="#control" aria-label="运行控制">
            <Play size={20} />
          </a>
          <a href="#gates" aria-label="质量门槛">
            <CheckCircle2 size={20} />
          </a>
          <a href="#orders" aria-label="交易单">
            <Table2 size={20} />
          </a>
          <a href="#artifacts" aria-label="产物">
            <FileText size={20} />
          </a>
        </nav>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">quant_box</p>
            <h1>自动信号复核</h1>
          </div>
          <button className="icon-button command-button" type="button" onClick={refresh} title="刷新仪表盘">
            <RefreshCw size={18} />
            <span>刷新</span>
          </button>
        </header>

        {loading && <StatePanel title="正在读取最新运行" message="正在读取本地 outputs 目录下的复核产物。" />}
        {error && <StatePanel title="仪表盘后端不可用" message={error} tone="danger" />}
        {!loading && !error && snapshot && (
          <Dashboard
            jobs={jobs}
            jobsError={jobsError}
            onJobStarted={recordStartedJob}
            onJobsRefresh={refreshJobs}
            snapshot={snapshot}
          />
        )}
      </main>
    </div>
  );
}

function Dashboard({
  jobs,
  jobsError,
  onJobStarted,
  onJobsRefresh,
  snapshot
}: {
  jobs: DashboardJob[];
  jobsError: string | null;
  onJobStarted: (job: DashboardJob) => void;
  onJobsRefresh: () => void;
  snapshot: DashboardSnapshot;
}) {
  const signalSummary = useMemo(() => actionSummary(snapshot.signal_summary), [snapshot.signal_summary]);
  const qualityItems = useMemo(
    () => [...(snapshot.freshness_notes ?? []), ...snapshot.quality_warnings],
    [snapshot.freshness_notes, snapshot.quality_warnings]
  );
  const readiness = readinessCopy(snapshot);
  return (
    <div className="dashboard-grid">
      <section className={`review-panel status-${snapshot.readiness.status}`} id="review">
        <div className="review-copy">
          <StatusIcon status={snapshot.readiness.status} />
          <div>
            <p className="eyebrow">最新运行结论</p>
            <h2>{readiness.label}</h2>
            <p>{readiness.summary}</p>
          </div>
        </div>
        <div className="date-grid">
          <Metric label="生成时间" value={formatDateTime(snapshot.latest_run.generated_at)} />
          <Metric label="信号日期" value={display(snapshot.latest_run.signal_date)} />
          <Metric label="计划交易日" value={display(snapshot.latest_run.intended_trade_date)} />
          <Metric label="策略模式" value={strategyLabel(snapshot.latest_run.strategy_mode)} />
        </div>
      </section>

      <RunControlPanel jobs={jobs} jobsError={jobsError} onJobStarted={onJobStarted} onJobsRefresh={onJobsRefresh} />

      <section className="panel gates-panel" id="gates">
        <SectionTitle icon={<CheckCircle2 size={18} />} title="质量门槛" aside={statusLabel(snapshot.latest_run.status)} />
        <div className="gate-grid">
          {snapshot.gates.map((gate) => (
            <GateCard gate={gate} key={gate.id} />
          ))}
        </div>
      </section>

      <section className="panel blockers-panel">
        <BlockerActionCenter
          items={snapshot.blocker_actions ?? []}
          jobs={jobs}
          onJobStarted={onJobStarted}
          onJobsRefresh={onJobsRefresh}
        />
      </section>

      <section className="panel quality-warning-panel">
        <QualityWarningsPanel warnings={qualityItems} />
      </section>

      <section className="panel order-summary signal-panel">
        <SectionTitle icon={<BarChart3 size={18} />} title="信号摘要" aside={signalSummary} />
        <SignalSummaryPanel snapshot={snapshot} />
      </section>

      <section className="panel report-panel">
        <SectionTitle icon={<FileText size={18} />} title="复核报告" aside={snapshot.report.daily_markdown.exists ? "可打开" : "缺失"} />
        <div className="report-summary">
          {snapshot.report.summary.map((item) => (
            <Metric key={item.label} label={reportSummaryLabel(item.label)} value={summaryValue(item.label, item.value)} />
          ))}
        </div>
        <ArtifactLink artifact={snapshot.report.daily_markdown} />
      </section>

      <section className="panel table-panel" id="orders">
        <SectionTitle icon={<Table2 size={18} />} title="人工交易单" aside={snapshot.orders.exists ? snapshot.orders.path : "缺失"} />
        {snapshot.orders.exists && snapshot.orders.valid ? <OrdersTable snapshot={snapshot} /> : <EmptyPanel message={snapshot.orders.error || "没有找到人工交易单产物。"} />}
      </section>

      <section className="panel artifacts-panel" id="artifacts">
        <SectionTitle icon={<FileText size={18} />} title="产物文件" aside={snapshot.output_dir} />
        <div className="artifact-list">
          {snapshot.artifacts.map((artifact) => (
            <ArtifactRow artifact={artifact} key={artifact.id} />
          ))}
        </div>
      </section>
    </div>
  );
}

function RunControlPanel({
  jobs,
  jobsError,
  onJobStarted,
  onJobsRefresh
}: {
  jobs: DashboardJob[];
  jobsError: string | null;
  onJobStarted: (job: DashboardJob) => void;
  onJobsRefresh: () => void;
}) {
  const [mode, setMode] = useState<DashboardRunMode>("candidate");
  const [pendingAction, setPendingAction] = useState<DashboardJobAction | null>(null);
  const [pendingStop, setPendingStop] = useState(false);
  const [controlError, setControlError] = useState<string | null>(null);
  const activeJob = jobs.find(isActiveJob) ?? null;
  const latestJob = activeJob ?? jobs[0] ?? null;
  const disabled = Boolean(activeJob || pendingAction || pendingStop);

  const runAction = (action: DashboardJobAction, runMode?: DashboardRunMode) => {
    setPendingAction(action);
    setControlError(null);
    startDashboardJob(runMode ? { action, mode: runMode } : { action })
      .then((job) => {
        onJobStarted(job);
        onJobsRefresh();
      })
      .catch((err: Error) => setControlError(err.message))
      .finally(() => setPendingAction(null));
  };

  const stopActiveJob = () => {
    if (!activeJob) {
      return;
    }
    setPendingStop(true);
    setControlError(null);
    stopDashboardJob(activeJob.id)
      .then((job) => {
        onJobStarted(job);
        onJobsRefresh();
      })
      .catch((err: Error) => setControlError(err.message))
      .finally(() => setPendingStop(false));
  };

  return (
    <section className="panel control-panel" id="control">
      <SectionTitle
        icon={activeJob ? <Loader2 className="spin-icon" size={18} /> : <Play size={18} />}
        title="运行控制"
        aside={activeJob ? statusLabel(activeJob.status) : "就绪"}
      />
      <div className="control-grid">
        <button
          className="control-action"
          disabled={disabled}
          onClick={() => runAction("repair_point_in_time")}
          type="button"
        >
          <Wrench size={17} />
          <span>{pendingAction === "repair_point_in_time" ? "正在启动" : "修复 daily_basic 缺口"}</span>
        </button>
        <div className="signal-run-box">
          <div className="segmented-control" aria-label="自动信号输出模式">
            <button className={mode === "candidate" ? "active" : ""} disabled={disabled} onClick={() => setMode("candidate")} type="button">
              候选输出
            </button>
            <button className={mode === "normal" ? "active" : ""} disabled={disabled} onClick={() => setMode("normal")} type="button">
              正常门槛输出
            </button>
          </div>
          <button className="control-action primary" disabled={disabled} onClick={() => runAction("run_auto_signal", mode)} type="button">
            <Play size={17} />
            <span>{pendingAction === "run_auto_signal" ? "正在启动" : "重跑自动信号"}</span>
          </button>
        </div>
        {activeJob && (
          <button className="control-action danger" disabled={pendingStop || activeJob.status === "stopping"} onClick={stopActiveJob} type="button">
            <Square size={16} />
            <span>{pendingStop || activeJob.status === "stopping" ? "正在停止" : "停止当前任务"}</span>
          </button>
        )}
      </div>
      {controlError && <p className="inline-error">{controlError}</p>}
      {jobsError && <p className="inline-error">{jobsError}</p>}
      {latestJob ? <JobStatusCard job={latestJob} /> : <EmptyPanel message="暂无后台任务记录。" />}
    </section>
  );
}

function JobStatusCard({ job }: { job: DashboardJob }) {
  return (
    <div className={`job-card job-${job.status}`}>
      <div className="job-head">
        <div>
          <strong>{job.label}</strong>
          <span>{statusLabel(job.status)}</span>
        </div>
        <small>{formatDateTime(job.completed_at || job.started_at)}</small>
      </div>
      <p>{job.message}</p>
      {job.progress && <JobProgressView job={job} />}
      <div className="job-command" title={job.command.join(" ")}>
        {job.command.map(commandPart).join(" ")}
      </div>
      <div className="log-tail">
        <div>
          <Terminal size={15} />
          <span>日志尾部</span>
        </div>
        {job.log_tail.length ? (
          <pre>{job.log_tail.join("\n")}</pre>
        ) : (
          <p className="empty-log">{job.status === "running" ? "等待日志输出..." : "没有日志输出。"}</p>
        )}
      </div>
    </div>
  );
}

function JobProgressView({ job }: { job: DashboardJob }) {
  const progress = job.progress;
  if (!progress || !progress.steps.length) {
    return null;
  }
  return (
    <div className="progress-box">
      <div className="progress-head">
        <span>{progress.summary || "正在等待进度更新"}</span>
        <strong>{progress.percent}%</strong>
      </div>
      <div className="progress-bar" aria-label="任务进度">
        <span style={{ width: `${Math.min(Math.max(progress.percent, 0), 100)}%` }} />
      </div>
      <div className="progress-steps">
        {progress.steps.map((step) => (
          <div className={`progress-step step-${step.status}`} key={step.id}>
            <span />
            <div>
              <strong>{step.label}</strong>
              {step.message && <small>{step.message}</small>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function BlockerActionCenter({
  items,
  jobs,
  onJobStarted,
  onJobsRefresh
}: {
  items: BlockerAction[];
  jobs: DashboardJob[];
  onJobStarted: (job: DashboardJob) => void;
  onJobsRefresh: () => void;
}) {
  const [pendingId, setPendingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const activeJob = jobs.find(isActiveJob) ?? null;
  const runnableCount = items.filter((item) => item.action).length;

  const runBlockerAction = (item: BlockerAction) => {
    if (!item.action) {
      return;
    }
    setPendingId(item.id);
    setError(null);
    startDashboardJob({ action: item.action.action, mode: item.action.mode ?? undefined })
      .then((job) => {
        onJobStarted(job);
        onJobsRefresh();
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setPendingId(null));
  };

  return (
    <div className="blocker-center">
      <SectionTitle
        icon={<AlertTriangle size={18} />}
        title="阻塞修复中心"
        aside={items.length ? `${runnableCount}/${items.length} 可一键处理` : "无阻塞"}
      />
      {items.length ? (
        <div className="blocker-stack">
          {items.map((item) => {
            const disabled = Boolean(activeJob || pendingId || !item.action);
            return (
              <article className={`blocker-card blocker-${item.severity}`} key={item.id}>
                <div className="blocker-copy">
                  <strong>{item.title}</strong>
                  <p>{item.detail}</p>
                  <small>{translateReason(item.reason)}</small>
                </div>
                {item.action ? (
                  <button className="blocker-action" disabled={disabled} onClick={() => runBlockerAction(item)} type="button">
                    {pendingId === item.id ? <Loader2 className="spin-icon" size={16} /> : <Wrench size={16} />}
                    <span>{pendingId === item.id ? "正在启动" : item.action.label}</span>
                  </button>
                ) : (
                  <span className="pill muted">查看报告</span>
                )}
              </article>
            );
          })}
        </div>
      ) : (
        <EmptyPanel message="没有记录阻塞原因。" />
      )}
      {activeJob && <p className="helper-text">当前已有后台任务运行，修复按钮会在任务结束后恢复。</p>}
      {error && <p className="inline-error">{error}</p>}
    </div>
  );
}

function QualityWarningsPanel({ warnings }: { warnings: string[] }) {
  return (
    <div className="quality-panel">
      <SectionTitle icon={<Info size={18} />} title="质量提示" aside={warnings.length ? `${warnings.length} 条` : "无"} />
      {warnings.length ? (
        <>
          <div className="quality-overview">
            <div>
              <strong>需要复核数据覆盖与新鲜度</strong>
              <span>这些提示来自自动信号质量门槛，不是前端程序错误。</span>
            </div>
            <span className="pill warning-pill">暂缓复核</span>
          </div>
          <div className="warning-stack">
            {warnings.map((warning) => {
              const item = reasonMeta(warning);
              return (
                <article className="warning-card" key={warning}>
                  <div className="warning-icon">
                    <AlertTriangle size={18} />
                  </div>
                  <div>
                    <strong>{item.title}</strong>
                    <p>{item.detail}</p>
                  </div>
                </article>
              );
            })}
          </div>
        </>
      ) : (
        <EmptyPanel message="没有质量提示。" />
      )}
    </div>
  );
}

function SignalSummaryPanel({ snapshot }: { snapshot: DashboardSnapshot }) {
  const buy = snapshot.signal_summary.BUY ?? 0;
  const hold = snapshot.signal_summary.HOLD ?? 0;
  const sell = snapshot.signal_summary.SELL ?? 0;
  const totalActions = Math.max(buy + hold + sell, 1);
  const buyWidth = `${Math.round((buy / totalActions) * 100)}%`;
  const holdWidth = `${Math.round((hold / totalActions) * 100)}%`;
  const sellWidth = `${Math.round((sell / totalActions) * 100)}%`;
  const actionableText =
    snapshot.orders.total_rows > 0 ? `${snapshot.orders.actionable_count}/${snapshot.orders.total_rows}` : "0/0";
  return (
    <div className="signal-summary-shell">
      <div className="action-cards">
        <ActionCard label="买入" value={buy} tone="buy" icon={<TrendingUp size={18} />} />
        <ActionCard label="持有" value={hold} tone="hold" icon={<CircleDot size={18} />} />
        <ActionCard label="卖出" value={sell} tone="sell" icon={<XCircle size={18} />} />
      </div>
      <div className="action-bar" aria-label="信号动作分布">
        <span className="bar-buy" style={{ width: buyWidth }} />
        <span className="bar-hold" style={{ width: holdWidth }} />
        <span className="bar-sell" style={{ width: sellWidth }} />
      </div>
      <div className="execution-strip">
        <div>
          <ClipboardList size={18} />
          <span>交易单</span>
          <strong>{snapshot.orders.total_rows}</strong>
        </div>
        <div>
          <ShoppingCart size={18} />
          <span>可直接执行</span>
          <strong>{actionableText}</strong>
        </div>
      </div>
    </div>
  );
}

function ActionCard({ label, value, tone, icon }: { label: string; value: number; tone: "buy" | "hold" | "sell"; icon: ReactNode }) {
  return (
    <div className={`action-card action-${tone}`}>
      <div>
        {icon}
        <span>{label}</span>
      </div>
      <strong>{value}</strong>
    </div>
  );
}

function GateCard({ gate }: { gate: Gate }) {
  const view = gateView(gate);
  return (
    <article className={`gate-card gate-${gate.status}`}>
      <div className="gate-head">
        <span className="status-dot" />
        <strong>{view.label}</strong>
      </div>
      <p>{view.summary}</p>
      {gate.issues.length > 0 && <small>{gate.issues.length} {gate.status === "warn" ? "条提示" : "条问题"}</small>}
    </article>
  );
}

function OrdersTable({ snapshot }: { snapshot: DashboardSnapshot }) {
  const columns = ORDER_COLUMNS.filter((column) => snapshot.orders.columns.includes(column));
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{ORDER_COLUMN_LABELS[column] ?? column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {snapshot.orders.rows.map((row, index) => (
            <tr key={`${row.instrument ?? "row"}-${index}`}>
              {columns.map((column) => (
                <td key={column}>{formatOrderCell(column, row[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ArtifactRow({ artifact }: { artifact: Artifact }) {
  return (
    <div className="artifact-row">
      <div>
        <strong>{artifactLabel(artifact)}</strong>
        <span>{artifact.path}</span>
      </div>
      <ArtifactLink artifact={artifact} compact />
    </div>
  );
}

function ArtifactLink({ artifact, compact = false }: { artifact: Artifact; compact?: boolean }) {
  if (!artifact.exists || !artifact.downloadable) {
    return <span className="pill muted">{artifact.exists ? "本地路径" : "缺失"}</span>;
  }
  return (
    <a className={compact ? "icon-link compact" : "icon-link"} href={artifactUrl(artifact.id)} target="_blank" rel="noreferrer" title={`打开${artifactLabel(artifact)}`}>
      <ExternalLink size={16} />
      <span>{compact ? "打开" : artifactLabel(artifact)}</span>
    </a>
  );
}

function IssueList({ items, empty, tone }: { items: string[]; empty: string; tone: "danger" | "warning" }) {
  if (!items.length) {
    return <EmptyPanel message={empty} />;
  }
  return (
    <ul className={`issue-list issue-list-${tone}`}>
      {items.map((item) => (
        <li className={`issue-item-${issueTone(item, tone)}`} key={item}>
          {translateReason(item)}
        </li>
      ))}
    </ul>
  );
}

function SectionTitle({ icon, title, aside }: { icon: ReactNode; title: string; aside?: string }) {
  return (
    <div className="section-title">
      <div>
        {icon}
        <h3>{title}</h3>
      </div>
      {aside && <span>{aside}</span>}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value || "-"}</strong>
    </div>
  );
}

function EmptyPanel({ message }: { message: string }) {
  return <p className="empty-text">{message}</p>;
}

function StatePanel({ title, message, tone = "neutral" }: { title: string; message: string; tone?: "neutral" | "danger" }) {
  return (
    <section className={`state-panel state-${tone}`}>
      <h2>{title}</h2>
      <p>{message}</p>
    </section>
  );
}

function StatusIcon({ status }: { status: ReadinessStatus }) {
  if (status === "ready") {
    return <CheckCircle2 size={36} />;
  }
  if (status === "missing" || status === "error") {
    return <AlertTriangle size={36} />;
  }
  if (status === "blocked") {
    return <XCircle size={36} />;
  }
  return <CircleDot size={36} />;
}

function actionSummary(summary: Record<string, number>) {
  const entries = Object.entries(summary);
  return entries.length ? entries.map(([key, value]) => `${ACTION_LABELS[key] ?? key} ${value}`).join(" / ") : "暂无动作";
}

function formatDateTime(value?: string | null) {
  if (!value) {
    return "-";
  }
  return value.replace("T", " ").replace(/\+.*$/, "");
}

function display(value: unknown) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  return String(value);
}

export function gateStatusLabel(status: GateStatus) {
  return {
    pass: "通过",
    fail: "未通过",
    warn: "有提示",
    missing: "缺失",
    hold: "候选保留"
  }[status];
}

function readinessCopy(snapshot: DashboardSnapshot) {
  const blockers = snapshot.block_reasons.length;
  const copy: Record<ReadinessStatus, { label: string; summary: string }> = {
    ready: {
      label: "可以进入人工复核",
      summary: "最新信号通过了必要门槛，可以继续做人工交易复核。"
    },
    blocked: {
      label: "复核未通过",
      summary: `这不是前端程序报错，而是自动信号被质量门槛拦截。当前有 ${blockers} 条阻塞原因需要先处理。`
    },
    candidate_only: {
      label: "仅候选输出",
      summary: "本次运行被设置为候选模式，不会覆盖正式信号或最新持仓。"
    },
    missing: {
      label: "缺少最新报告",
      summary: "还没有找到 auto_signal_report.json。请先运行自动信号流程。"
    },
    error: {
      label: "报告读取失败",
      summary: "最新报告存在但无法解析，请检查 JSON 文件是否完整。"
    }
  };
  return copy[snapshot.readiness.status];
}

function gateView(gate: Gate) {
  const labels: Record<string, string> = {
    data_health: "数据健康",
    data_governance: "点时数据治理",
    parameter_quality: "参数质量",
    backtest_quality: "回测质量",
    account: "账户与持仓",
    candidate_only: "候选输出模式"
  };
  const defaultSummaries: Record<GateStatus, string> = {
    pass: "已通过。",
    fail: "未通过，请查看阻塞原因。",
    warn: "有提示，不一定阻塞复核。",
    missing: "缺少对应产物。",
    hold: "当前处于候选输出保留状态。"
  };
  const firstReason = gate.issues.length ? translateReason(gate.issues[0]) : "";
  const more = gate.issues.length > 1 ? `（另有 ${gate.issues.length - 1} 条）` : "";
  const governanceSuperseded = Boolean(gate.details.supersedes_auto_report);
  const summaries: Record<string, string> = {
    data_health: gate.status === "pass" ? "原始数据、价格面板和因子覆盖检查通过。" : firstReason + more || defaultSummaries[gate.status],
    data_governance: governanceSuperseded
      ? gate.status === "fail"
        ? firstReason + more || defaultSummaries[gate.status]
        : "最新点时治理报告已不再包含该 daily_basic 缺口；请重跑自动信号刷新复核结论。"
      : gate.status === "pass"
        ? "点时治理输入可用。"
        : firstReason + more || defaultSummaries[gate.status],
    parameter_quality: gate.status === "pass" ? "参数质量达到门槛。" : firstReason + more || defaultSummaries[gate.status],
    backtest_quality: gate.status === "pass" ? "回测质量达到门槛。" : firstReason + more || defaultSummaries[gate.status],
    account: gate.status === "pass" ? "账户与持仓摘要已加载。" : firstReason + more || defaultSummaries[gate.status],
    candidate_only: gate.status === "hold" ? "本次只生成候选产物，不会写入正式信号。" : "未启用候选保留。"
  };
  return {
    label: labels[gate.id] ?? gate.label,
    summary: summaries[gate.id] ?? defaultSummaries[gate.status]
  };
}

function translateReason(reason: string) {
  const trimmed = reason.trim();
  const prefix = trimmed.startsWith("阻塞：") ? "阻塞：" : "";
  const text = trimmed
    .replace(/^阻塞：/, "")
    .replace(/^data:/, "")
    .replace(/^governance:/, "")
    .replace(/^backtest:/, "")
    .replace(/^params?:/, "")
    .replace(/^account:/, "");
  const coverage = text.match(/^factor_latest_coverage_below_threshold:(.+)$/);
  if (coverage) {
    return `${prefix}因子最新覆盖率低于阈值：${coverage[1]}`;
  }
  const factorDate = text.match(/^factor_latest_before_end:(.+)$/);
  if (factorDate) {
    return `${prefix}因子最新日期早于目标日期：${factorDate[1]}`;
  }
  const stCalendar = text.match(/^st_calendar_end_before_factor_end:(.+)$/);
  if (stCalendar) {
    return `${prefix}ST 历史日历早于因子缓存截止日期：${stCalendar[1]}`;
  }
  const dailyBasicCoverage = text.match(/^daily_basic_date_coverage_below_required:(.+)$/);
  if (dailyBasicCoverage) {
    return `${prefix}daily_basic 日期覆盖不足：${dailyBasicCoverage[1]}`;
  }
  if (text === "data_governance_repaired_after_auto_report") {
    return `${prefix}daily_basic 缺口已按最新点时治理报告修复；请重跑自动信号刷新复核结论。`;
  }
  if (text === "candidate_only_requested") {
    return `${prefix}已启用候选输出模式，本次不会生成或覆盖正式交易信号。`;
  }
  return trimmed;
}

function issueTone(reason: string, fallback: "danger" | "warning") {
  const text = reason.trim();
  if (text.includes("candidate_only_requested")) {
    return "hold";
  }
  if (text.includes("daily_basic_date_coverage_below_required")) {
    return "warning";
  }
  return fallback;
}

function reasonMeta(reason: string) {
  const translated = translateReason(reason);
  if (translated.includes("因子最新覆盖率")) {
    return {
      title: "因子覆盖率不足",
      detail: translated
    };
  }
  if (translated.includes("因子最新日期")) {
    return {
      title: "因子日期落后",
      detail: translated
    };
  }
  if (translated.includes("ST 历史日历")) {
    return {
      title: "点时日历提示",
      detail: translated
    };
  }
  if (translated.includes("daily_basic 日期覆盖")) {
    return {
      title: "daily_basic 覆盖不足",
      detail: translated
    };
  }
  if (translated.includes("重跑自动信号刷新复核结论")) {
    return {
      title: "自动信号报告需重跑",
      detail: translated
    };
  }
  return {
    title: "质量门槛提示",
    detail: translated
  };
}

function statusLabel(value?: string | null) {
  if (!value) {
    return "-";
  }
  return STATUS_LABELS[value] ?? value;
}

function isActiveJob(job: DashboardJob) {
  return job.status === "running" || job.status === "stopping";
}

function commandPart(value: string) {
  return value.includes(" ") ? `"${value}"` : value;
}

function strategyLabel(value?: string | null) {
  if (!value) {
    return "-";
  }
  return STRATEGY_LABELS[value] ?? value;
}

function reportSummaryLabel(label: string) {
  return {
    "Strategy mode": "策略模式",
    "Signal date": "信号日期",
    "Intended trade date": "计划交易日"
  }[label] ?? label;
}

function summaryValue(label: string, value: unknown) {
  if (label === "Strategy mode") {
    return strategyLabel(value ? String(value) : "");
  }
  return display(value);
}

function artifactLabel(artifact: Artifact) {
  return {
    auto_signal_report: "自动信号 JSON 报告",
    auto_run_status: "自动运行状态",
    daily_report: "每日 Markdown 报告",
    signal: "信号 CSV",
    holdings: "持仓 CSV",
    manual_orders: "人工交易单 CSV",
    order_confirmation: "订单确认 CSV",
    fill_feedback: "成交回填 CSV",
    data_health: "数据健康 JSON",
    data_governance: "数据治理 JSON",
    parameter_quality: "参数质量 JSON",
    backtest_quality: "回测质量 JSON",
    fundamental_screen_report: "基本面筛选报告"
  }[artifact.id] ?? artifact.label;
}

function formatOrderCell(column: string, value: unknown) {
  if (column === "action") {
    return ACTION_LABELS[String(value)] ?? display(value);
  }
  if (column === "is_order_actionable") {
    const normalized = String(value).toLowerCase();
    if (normalized === "true") {
      return "是";
    }
    if (normalized === "false") {
      return "否";
    }
  }
  if (column === "note") {
    return translateOrderNote(display(value));
  }
  return display(value);
}

function translateOrderNote(value: string) {
  return value
    .replace(/blocked:/g, "阻塞：")
    .replace(/order_not_actionable/g, "订单不可直接执行")
    .split(/[,;]/)
    .map((part) => translateReason(part))
    .join("；");
}
