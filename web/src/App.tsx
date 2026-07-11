import {
  AlertTriangle,
  ArrowDown,
  BarChart3,
  BookOpen,
  CheckCircle2,
  CircleDot,
  ClipboardList,
  Database,
  ExternalLink,
  FileText,
  Gauge,
  GitBranch,
  Info,
  Layers3,
  LayoutDashboard,
  ListChecks,
  Loader2,
  Play,
  RefreshCw,
  Settings2,
  ShieldAlert,
  ShoppingCart,
  Square,
  Table2,
  Terminal,
  TrendingUp,
  Workflow,
  Wrench,
  XCircle
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { artifactUrl, fetchDashboardJobs, fetchDashboardPrecheck, fetchLatestDashboard, startDashboardJob, stopDashboardJob } from "./api";
import ExecutionWorkspace from "./ExecutionWorkspace";
import OperationsWorkspace from "./OperationsWorkspace";
import AccountWorkspace from "./AccountWorkspace";
import StockDetailWorkspace from "./StockDetailWorkspace";
import type {
  Artifact,
  BlockerAction,
  DashboardJob,
  DashboardJobAction,
  DashboardPrecheck,
  DashboardPrecheckItem,
  DashboardRunMode,
  DashboardSnapshot,
  Gate,
  GateStatus,
  ReadinessStatus
} from "./types";

const ORDER_COLUMNS = [
  "instrument",
  "name",
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
  name: "股票名称",
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

type WorkspaceView = "dashboard" | "operations" | "execution" | "account" | "overview";

export default function App() {
  const [workspaceView, setWorkspaceView] = useState<WorkspaceView>("dashboard");
  const [selectedStock, setSelectedStock] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null);
  const [jobs, setJobs] = useState<DashboardJob[]>([]);
  const [precheck, setPrecheck] = useState<DashboardPrecheck | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [precheckError, setPrecheckError] = useState<string | null>(null);
  const [refreshCount, setRefreshCount] = useState(0);
  const [jobsRefreshCount, setJobsRefreshCount] = useState(0);
  const [precheckRefreshCount, setPrecheckRefreshCount] = useState(0);

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

  useEffect(() => {
    const controller = new AbortController();
    fetchDashboardPrecheck(controller.signal)
      .then((data) => {
        setPrecheck(data);
        setPrecheckError(null);
      })
      .catch((err: Error) => {
        if (err.name !== "AbortError") {
          setPrecheckError(err.message);
        }
      });
    return () => controller.abort();
  }, [precheckRefreshCount]);

  const refresh = useCallback(() => {
    setRefreshCount((value) => value + 1);
    setPrecheckRefreshCount((value) => value + 1);
  }, []);
  const refreshJobs = useCallback(() => setJobsRefreshCount((value) => value + 1), []);
  const refreshPrecheck = useCallback(() => setPrecheckRefreshCount((value) => value + 1), []);
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
  const openStock = useCallback((instrument: string) => {
    setSelectedStock(instrument);
  }, []);
  const closeStock = useCallback(() => setSelectedStock(null), []);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-mark">Q</div>
        <nav className="side-nav" aria-label="主要工作区">
          <button
            aria-label="打开每日复核台"
            className={workspaceView === "dashboard" ? "active" : ""}
            onClick={() => setWorkspaceView("dashboard")}
            title="每日复核台"
            type="button"
          >
            <LayoutDashboard size={20} />
          </button>
          <button
            aria-label="打开量化操作中心"
            className={workspaceView === "operations" ? "active" : ""}
            onClick={() => setWorkspaceView("operations")}
            title="量化操作中心"
            type="button"
          >
            <Workflow size={20} />
          </button>
          <button
            aria-label="打开项目全景"
            className={workspaceView === "overview" ? "active" : ""}
            onClick={() => setWorkspaceView("overview")}
            title="项目全景"
            type="button"
          >
            <BookOpen size={20} />
          </button>
          <button
            aria-label="打开交易执行工作区"
            className={workspaceView === "execution" ? "active" : ""}
            onClick={() => setWorkspaceView("execution")}
            title="交易执行"
            type="button"
          >
            <ListChecks size={20} />
          </button>
          <button aria-label="打开账户与持仓" className={workspaceView === "account" ? "active" : ""} onClick={() => setWorkspaceView("account")} title="账户与持仓" type="button"><Settings2 size={20} /></button>
        </nav>
        <span className="sidebar-caption">{workspaceView === "dashboard" ? "复核" : workspaceView === "operations" ? "操作" : workspaceView === "execution" ? "执行" : workspaceView === "account" ? "账户" : "全景"}</span>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">quant_box</p>
            <h1>{workspaceView === "dashboard" ? "自动信号复核" : workspaceView === "operations" ? "量化操作中心" : workspaceView === "execution" ? "交易执行" : workspaceView === "account" ? "账户与持仓" : "项目全景"}</h1>
          </div>
          {workspaceView === "dashboard" ? (
            <button className="icon-button command-button" type="button" onClick={refresh} title="刷新仪表盘">
              <RefreshCw size={18} />
              <span>刷新</span>
            </button>
          ) : workspaceView === "overview" ? (
            <button className="icon-button command-button" type="button" onClick={() => setWorkspaceView("dashboard")}>
              <LayoutDashboard size={18} />
              <span>进入复核台</span>
            </button>
          ) : null}
        </header>

        {workspaceView === "dashboard" && (
          <nav className="dashboard-shortcuts" aria-label="复核台快捷导航">
            <a href="#review"><ShieldAlert size={15} />复核结论</a>
            <a href="#control"><Play size={15} />运行控制</a>
            <a href="#gates"><CheckCircle2 size={15} />质量门槛</a>
            <a href="#orders"><Table2 size={15} />人工交易单</a>
            <a href="#artifacts"><FileText size={15} />产物文件</a>
          </nav>
        )}

        {workspaceView === "overview" && <ProjectOverview onOpenDashboard={() => setWorkspaceView("dashboard")} />}
        {workspaceView === "operations" && <OperationsWorkspace jobs={jobs} onJobStarted={recordStartedJob} onJobsRefresh={refreshJobs} />}
        {workspaceView === "execution" && <ExecutionWorkspace />}
        {workspaceView === "account" && <AccountWorkspace />}
        {workspaceView === "dashboard" && loading && <StatePanel title="正在读取最新运行" message="正在读取本地 outputs 目录下的复核产物。" />}
        {workspaceView === "dashboard" && error && <StatePanel title="仪表盘后端不可用" message={error} tone="danger" />}
        {workspaceView === "dashboard" && !loading && !error && snapshot && (
          <Dashboard
            jobs={jobs}
            jobsError={jobsError}
            onJobStarted={recordStartedJob}
            onJobsRefresh={refreshJobs}
            onOpenStock={openStock}
            onPrecheckRefresh={refreshPrecheck}
            precheck={precheck}
            precheckError={precheckError}
            snapshot={snapshot}
          />
        )}
      </main>
      {selectedStock && <StockDetailWorkspace instrument={selectedStock} key={selectedStock} onClose={closeStock} />}
    </div>
  );
}

function ProjectOverview({ onOpenDashboard }: { onOpenDashboard: () => void }) {
  const questions = [
    "今天的数据是否完整、可靠？",
    "当前策略是否经过足够的样本外验证？",
    "最新信号是否满足正式交易门槛？",
    "如果满足，人工应该买什么、卖什么、买卖多少？"
  ];
  const outputs = ["信号文件", "目标持仓", "人工交易单", "质量检查报告", "阻塞原因", "订单确认模板", "成交回填模板", "Web 复核结论"];
  const pipeline = [
    ["数据源", "Tushare 日线、估值、ST、指数成分、财务数据"],
    ["数据治理", "清洗、复权、交易日历与点时正确性检查"],
    ["因子研究", "Qlib Alpha158、IC 评估、动态权重与基本面筛选"],
    ["策略路由", "综合评分、Top 15、月度调仓与市场状态暴露"],
    ["真实回测", "交易费用、滑点、容量、涨跌停、停牌与风险退出"],
    ["质量门槛", "数据、参数、回测、账户与持仓共同决定可执行性"],
    ["人工闭环", "交易单、订单确认、成交回填与最新持仓更新"]
  ];
  const dataSources = ["A 股主板日线 OHLCV", "复权因子", "daily_basic 估值与市值", "ST 历史状态", "三大指数历史成分", "财务指标与分红", "交易日历"];
  const pointInTimeChecks = ["ST 历史日历覆盖", "指数历史成分覆盖", "daily_basic 日期覆盖", "财务披露滞后", "复权因子版本", "因子缓存新鲜度", "历史股票池来源覆盖"];
  const researchCapabilities = ["Alpha158 计算与缓存", "单因子和年度 IC", "因子分组收益", "rolling IC 动态权重", "相关性过滤与动态选择", "行业、市值中性化", "流动性与基本面筛选", "可选机器学习策略"];
  const tradingConstraints = ["手续费、最低佣金、印花税、过户费", "固定与动态滑点", "成交额参与率和容量限制", "不同板块与 ST 涨跌停", "停牌、无成交与长期停牌退出", "止损、止盈和市场状态仓位", "行业集中度与换手限制"];
  const qualityGroups = [
    ["数据健康", "原始行情、价格面板、因子覆盖和目标日期"],
    ["数据治理", "ST、指数成分、股票池、daily_basic 与元数据"],
    ["参数质量", "验证窗口、正收益率、Sharpe、回撤和样本外收益"],
    ["回测质量", "年化收益、最大回撤、换手和交易成本"],
    ["账户持仓", "账户资产、现金、真实持仓、手数和参考价格"]
  ];
  const workflow = ["运行每日自动流程", "查看复核结论与报告", "确认 is_executable", "检查门槛和阻塞原因", "核对人工交易单", "在券商端手工下单", "填写成交回填 CSV", "校验后更新真实持仓"];

  return (
    <div className="overview-page">
      <section className="overview-intro">
        <div>
          <span className="overview-kicker">本地量化研究与人工交易决策系统</span>
          <h2>从数据可信，到信号可执行</h2>
          <p>quant_box 把数据治理、因子研究、真实化回测、质量门槛和人工交易闭环集中在一个本地工作台中。系统负责给出证据和订单建议，最终交易决定始终由你确认。</p>
          <div className="overview-actions">
            <button className="overview-primary" onClick={onOpenDashboard} type="button">
              <LayoutDashboard size={18} />
              打开每日复核台
            </button>
            <span><ShieldAlert size={16} /> 不自动下单 · 不绕过质量门槛</span>
          </div>
        </div>
        <div className="position-card">
          <small>项目定位</small>
          <strong>本地量化研究平台</strong>
          <ArrowDown size={18} />
          <strong>每日信号生产流水线</strong>
          <ArrowDown size={18} />
          <strong>人工交易决策控制台</strong>
        </div>
      </section>

      <section className="overview-section">
        <OverviewHeading number="01" icon={<Gauge size={19} />} title="它解决什么问题" subtitle="把每天最重要的判断压缩成四个问题" />
        <div className="question-grid">
          {questions.map((question, index) => <article key={question}><span>0{index + 1}</span><strong>{question}</strong></article>)}
        </div>
        <div className="output-strip">
          <span>最终产物</span>
          <div>{outputs.map((output) => <strong key={output}>{output}</strong>)}</div>
        </div>
      </section>

      <section className="overview-section">
        <OverviewHeading number="02" icon={<GitBranch size={19} />} title="整体数据流" subtitle="每一层都为下一层提供可验证的证据" />
        <div className="pipeline-track">
          {pipeline.map(([title, detail], index) => (
            <article key={title}>
              <span>{index + 1}</span>
              <div><strong>{title}</strong><p>{detail}</p></div>
              {index < pipeline.length - 1 && <ArrowDown className="pipeline-arrow" size={17} />}
            </article>
          ))}
        </div>
        <p className="safety-note"><ShieldAlert size={18} /> 任一关键质量门槛未通过，只生成候选结果，不覆盖正式信号和最新持仓。</p>
      </section>

      <div className="overview-two-column">
        <section className="overview-section">
          <OverviewHeading number="03" icon={<Database size={19} />} title="数据层" subtitle="本地缓存、私密配置与点时正确性" />
          <TagCloud items={dataSources} />
          <DirectoryMap />
          <h3 className="overview-subtitle">点时治理检查</h3>
          <CheckList items={pointInTimeChecks} />
        </section>

        <section className="overview-section">
          <OverviewHeading number="04" icon={<Layers3 size={19} />} title="因子与选股研究" subtitle="从 Alpha158 到市场状态路由" />
          <TagCloud items={researchCapabilities} />
          <div className="strategy-card">
            <div><small>默认持仓</small><strong>Top 15</strong></div>
            <div><small>调仓频率</small><strong>月度</strong></div>
            <div><small>换手控制</small><strong>Rank Buffer</strong></div>
          </div>
          <p className="overview-copy">年度状态路由会依据趋势、波动和回撤，在不同因子来源与仓位暴露之间切换，让策略不再依赖一套固定选股逻辑。</p>
        </section>
      </div>

      <section className="overview-section">
        <OverviewHeading number="05" icon={<TrendingUp size={19} />} title="为什么回测更接近真实交易" subtitle="不是无摩擦的收盘价排名游戏" />
        <div className="constraint-grid">{tradingConstraints.map((item) => <article key={item}><CheckCircle2 size={17} /><span>{item}</span></article>)}</div>
        <div className="backtest-facts">
          <Metric label="默认初始资金" value="¥1,000,000" />
          <Metric label="最大成交额参与率" value="5%" />
          <Metric label="正式收益目标" value="约 20% / 年" />
          <Metric label="最大回撤门槛" value="不差于 -20%" />
        </div>
      </section>

      <section className="overview-section quality-architecture">
        <OverviewHeading number="06" icon={<ShieldAlert size={19} />} title="质量门槛是核心安全机制" subtitle="脚本跑完，不等于信号可以交易" />
        <div className="quality-architecture-grid">
          {qualityGroups.map(([title, detail]) => <article key={title}><CheckCircle2 size={18} /><div><strong>{title}</strong><p>{detail}</p></div></article>)}
        </div>
        <div className="candidate-boundary">
          <div><small>门槛失败时仍会生成</small><strong>候选信号 · 候选交易单 · 阻塞原因 · 确认与回填模板</strong></div>
          <div><small>绝不会自动覆盖</small><strong>正式信号 · 正式交易单 · outputs/latest_holdings.csv</strong></div>
        </div>
      </section>

      <section className="overview-section">
        <OverviewHeading number="07" icon={<ShoppingCart size={19} />} title="每日信号与人工交易闭环" subtitle="从复核到真实成交，每一步都可追踪" />
        <div className="workflow-row">
          {workflow.map((item, index) => <article key={item}><span>{index + 1}</span><strong>{item}</strong></article>)}
        </div>
        <p className="safety-note"><ClipboardList size={18} /> 成交回填会拒绝 PENDING、缺失成交数量、超计划成交和超持仓卖出，校验失败不会静默修改真实持仓。</p>
      </section>
    </div>
  );
}

function OverviewHeading({ number, icon, title, subtitle }: { number: string; icon: ReactNode; title: string; subtitle: string }) {
  return <div className="overview-heading"><span>{number}</span><div className="overview-heading-icon">{icon}</div><div><h2>{title}</h2><p>{subtitle}</p></div></div>;
}

function TagCloud({ items }: { items: string[] }) {
  return <div className="tag-cloud">{items.map((item) => <span key={item}>{item}</span>)}</div>;
}

function CheckList({ items }: { items: string[] }) {
  return <ul className="overview-check-list">{items.map((item) => <li key={item}><CheckCircle2 size={15} />{item}</li>)}</ul>;
}

function DirectoryMap() {
  return <div className="directory-map">
    <code>data/raw/</code><span>原始 CSV</span>
    <code>data/qlib_data/</code><span>Qlib provider</span>
    <code>data/prices/</code><span>回测价格面板</span>
    <code>data/factors/</code><span>因子、IC 与估值缓存</span>
    <code>data/fundamentals/</code><span>财务指标与分红</span>
  </div>;
}

function Dashboard({
  jobs,
  jobsError,
  onJobStarted,
  onJobsRefresh,
  onOpenStock,
  onPrecheckRefresh,
  precheck,
  precheckError,
  snapshot
}: {
  jobs: DashboardJob[];
  jobsError: string | null;
  onJobStarted: (job: DashboardJob) => void;
  onJobsRefresh: () => void;
  onOpenStock: (instrument: string) => void;
  onPrecheckRefresh: () => void;
  precheck: DashboardPrecheck | null;
  precheckError: string | null;
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

      <PrecheckPanel
        jobs={jobs}
        onJobStarted={onJobStarted}
        onJobsRefresh={onJobsRefresh}
        onRefresh={onPrecheckRefresh}
        precheck={precheck}
        precheckError={precheckError}
      />

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
        {snapshot.orders.exists && snapshot.orders.valid ? <OrdersTable onOpenStock={onOpenStock} snapshot={snapshot} /> : <EmptyPanel message={snapshot.orders.error || "没有找到人工交易单产物。"} />}
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

function PrecheckPanel({
  jobs,
  onJobStarted,
  onJobsRefresh,
  onRefresh,
  precheck,
  precheckError
}: {
  jobs: DashboardJob[];
  onJobStarted: (job: DashboardJob) => void;
  onJobsRefresh: () => void;
  onRefresh: () => void;
  precheck: DashboardPrecheck | null;
  precheckError: string | null;
}) {
  const [pendingId, setPendingId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const activeJob = jobs.find(isActiveJob) ?? null;
  const status = precheck?.status ?? "missing";
  const aside = precheck ? precheckStatusLabel(status) : "未读取";

  const runPrecheckAction = (item: DashboardPrecheckItem) => {
    if (!item.action) {
      return;
    }
    setPendingId(item.id);
    setActionError(null);
    startDashboardJob({ action: item.action.action, mode: item.action.mode ?? undefined })
      .then((job) => {
        onJobStarted(job);
        onJobsRefresh();
      })
      .catch((err: Error) => setActionError(err.message))
      .finally(() => setPendingId(null));
  };

  return (
    <section className={`panel precheck-panel precheck-${status}`}>
      <SectionTitle icon={<ClipboardList size={18} />} title="运行前预检查" aside={aside} />
      <div className="precheck-head">
        <div>
          <strong>{precheck ? precheck.summary : "正在等待预检查结果"}</strong>
          <span>{precheck ? `检查时间 ${formatDateTime(precheck.generated_at)}` : "读取本地证据，不执行数据下载或信号生成。"}</span>
        </div>
        <button className="icon-button compact-command" onClick={onRefresh} title="刷新预检查" type="button">
          <RefreshCw size={16} />
        </button>
      </div>
      {precheckError && <p className="inline-error">{precheckError}</p>}
      {actionError && <p className="inline-error">{actionError}</p>}
      {precheck ? (
        <div className="precheck-grid">
          {precheck.items.map((item) => (
            <article className={`precheck-item precheck-item-${item.status}`} key={item.id}>
              <div className="precheck-item-head">
                <span className="status-dot" />
                <strong>{precheckItemLabel(item)}</strong>
              </div>
              <p>{precheckItemSummary(item)}</p>
              {item.action ? (
                <button
                  className="precheck-action"
                  disabled={Boolean(activeJob || pendingId)}
                  onClick={() => runPrecheckAction(item)}
                  type="button"
                >
                  {pendingId === item.id ? <Loader2 className="spin-icon" size={15} /> : <Wrench size={15} />}
                  <span>{pendingId === item.id ? "正在启动" : item.action.label}</span>
                </button>
              ) : (
                <small>{precheckStatusLabel(item.status)}</small>
              )}
            </article>
          ))}
        </div>
      ) : (
        <EmptyPanel message="暂无预检查结果。" />
      )}
      {activeJob && <p className="helper-text">后台任务运行中，预检查修复动作会在任务结束后恢复。</p>}
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

function OrdersTable({ onOpenStock, snapshot }: { onOpenStock: (instrument: string) => void; snapshot: DashboardSnapshot }) {
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
              {columns.map((column) => {
                const instrument = String(row.instrument ?? "").trim();
                const clickable = Boolean(
                  (column === "instrument" || column === "name") && instrument && String(row[column] ?? "").trim()
                );
                return (
                  <td key={column}>
                    {clickable ? (
                      <button className="stock-detail-link" onClick={() => onOpenStock(instrument)} type="button">
                        {formatOrderCell(column, row[column])}
                      </button>
                    ) : formatOrderCell(column, row[column])}
                  </td>
                );
              })}
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
  const unconfirmedFactors = text.match(/^factor_symbols_unconfirmed:(\d+)$/);
  if (unconfirmedFactors) {
    return `${prefix}仍有 ${unconfirmedFactors[1]} 只股票未确认是否存在目标日行情。`;
  }
  const stCalendar = text.match(/^st_calendar_end_before_factor_end:(.+)$/);
  if (stCalendar) {
    return `${prefix}ST 历史日历早于因子缓存截止日期：${stCalendar[1]}`;
  }
  const dailyBasicCoverage = text.match(/^daily_basic_date_coverage_below_required:(.+)$/);
  if (dailyBasicCoverage) {
    return `${prefix}daily_basic 日期覆盖不足：${dailyBasicCoverage[1]}`;
  }
  const staleArtifact = text.match(/^artifact_before_target:(.+)$/);
  if (staleArtifact) {
    return `${prefix}证据日期早于当前目标日期：${staleArtifact[1]}`;
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

function precheckStatusLabel(value?: string | null) {
  return {
    pass: "通过",
    warn: "需确认",
    fail: "会阻塞",
    missing: "证据缺失"
  }[value || ""] ?? display(value);
}

function precheckItemLabel(item: DashboardPrecheckItem) {
  return {
    target_date: "目标交易日",
    data_health: "数据健康",
    data_governance: "点时治理",
    factor_freshness: "因子新鲜度",
    account: "账户与持仓"
  }[item.id] ?? item.label;
}

function precheckItemSummary(item: DashboardPrecheckItem) {
  if (item.issues.length) {
    return translateReason(item.issues[0]);
  }
  return item.summary;
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
