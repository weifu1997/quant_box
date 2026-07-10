import { AlertTriangle, CheckCircle2, ClipboardCheck, Loader2, RefreshCw, Save, ShieldCheck, WalletCards } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

import { applyExecutionFeedback, fetchExecutionWorkspace, previewExecutionFeedback } from "./api";
import type { ExecutionFillRow, ExecutionHoldingRow, ExecutionPreview, ExecutionWorkspaceData, FillStatus } from "./types";

const FILL_STATUSES: Array<{ value: FillStatus; label: string }> = [
  { value: "PENDING", label: "待处理" },
  { value: "FILLED", label: "全部成交" },
  { value: "PARTIAL", label: "部分成交" },
  { value: "CANCELLED", label: "已撤单" },
  { value: "SKIPPED", label: "未执行" }
];

export default function ExecutionWorkspace() {
  const [workspace, setWorkspace] = useState<ExecutionWorkspaceData | null>(null);
  const [rows, setRows] = useState<ExecutionFillRow[]>([]);
  const [preview, setPreview] = useState<ExecutionPreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [previewing, setPreviewing] = useState(false);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const load = useCallback(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetchExecutionWorkspace(controller.signal)
      .then((data) => {
        setWorkspace(data);
        setRows(data.rows);
        setPreview(null);
      })
      .catch((err: Error) => {
        if (err.name !== "AbortError") setError(err.message);
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, []);

  useEffect(() => load(), [load]);

  const actionableRows = useMemo(
    () => rows.filter((row) => Math.abs(Number(row.planned_order_shares || 0)) > 0),
    [rows]
  );

  const updateRow = (rowId: number, field: keyof ExecutionFillRow, value: string) => {
    setRows((current) => current.map((row) => row.row_id === rowId ? { ...row, [field]: value } : row));
    setPreview(null);
    setSuccess(null);
  };

  const runPreview = () => {
    if (!workspace?.source_id) return;
    setPreviewing(true);
    setError(null);
    previewExecutionFeedback(workspace.source_id, rows)
      .then(setPreview)
      .catch((err: Error) => setError(err.message))
      .finally(() => setPreviewing(false));
  };

  const apply = () => {
    if (!workspace?.source_id || !preview?.valid) return;
    if (!window.confirm("确认把本次成交回填应用到真实持仓？此操作会更新 config/current_holdings.csv。")) return;
    setApplying(true);
    setError(null);
    applyExecutionFeedback(workspace.source_id, rows)
      .then((result) => {
        setSuccess(result.message);
        load();
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setApplying(false));
  };

  if (loading) return <ExecutionState icon={<Loader2 className="spin-icon" size={24} />} title="正在读取成交回填" detail="正在加载最新正式成交模板和真实持仓。" />;
  if (error && !workspace) return <ExecutionState icon={<AlertTriangle size={24} />} title="交易执行工作区不可用" detail={error} tone="danger" />;
  if (!workspace || workspace.status === "missing" || workspace.status === "error") {
    return <ExecutionState icon={<ClipboardCheck size={24} />} title="暂无可回填的正式交易单" detail={workspace?.message ?? "请先生成正式交易信号。"} />;
  }

  return (
    <div className="execution-page">
      <section className="execution-hero">
        <div>
          <span className="overview-kicker">人工交易执行闭环</span>
          <h2>记录真实成交，再安全更新持仓</h2>
          <p>股票、方向和计划数量来自正式交易单并由后端锁定。你只需要填写实际成交结果，系统会在写入真实持仓前重新执行全部校验。</p>
        </div>
        <div className="execution-hero-actions">
          <button className="icon-button command-button" onClick={load} type="button"><RefreshCw size={17} />刷新模板</button>
          <span className={`execution-status status-${workspace.status}`}><ShieldCheck size={16} />{workspace.pending_count ? `${workspace.pending_count} 条待处理` : "可校验"}</span>
        </div>
      </section>

      <section className="execution-summary-grid">
        <ExecutionFact label="信号日期" value={workspace.signal_date || "—"} />
        <ExecutionFact label="计划交易日" value={workspace.intended_trade_date || "—"} />
        <ExecutionFact label="需要处理" value={`${actionableRows.length} 条`} />
        <ExecutionFact label="当前持仓" value={`${workspace.holdings.length} 只`} />
      </section>

      {error && <p className="inline-error">{error}</p>}
      {success && <p className="execution-success"><CheckCircle2 size={17} />{success}</p>}

      <section className="panel execution-editor-panel">
        <div className="section-heading">
          <div className="section-heading-title"><ClipboardCheck size={18} /><h3>成交回填</h3></div>
          <span>{workspace.source_id}</span>
        </div>
        <div className="execution-table-wrap">
          <table className="execution-table">
            <thead><tr><th>股票</th><th>方向</th><th>计划股数</th><th>成交状态</th><th>实际股数</th><th>成交价格</th><th>交易日期</th><th>费用</th><th>备注</th></tr></thead>
            <tbody>
              {rows.map((row) => <ExecutionRowEditor key={row.row_id} row={row} onChange={updateRow} />)}
            </tbody>
          </table>
        </div>
      </section>

      <section className="execution-review-grid">
        <div className="panel execution-validation-panel">
          <div className="section-heading">
            <div className="section-heading-title"><ShieldCheck size={18} /><h3>更新前校验</h3></div>
          </div>
          {!preview && <p className="execution-muted">填写完成后先点击“预览持仓变化”。存在待处理、超计划成交或超持仓卖出时，系统会拒绝更新。</p>}
          {preview && preview.valid && <div className="execution-valid"><CheckCircle2 size={20} /><div><strong>校验通过</strong><span>{preview.summary.applied_fill_rows} 条成交将应用到持仓</span></div></div>}
          {preview && !preview.valid && <IssuePanel issues={preview.issues} />}
          <div className="execution-action-row">
            <button className="control-action" disabled={previewing || applying} onClick={runPreview} type="button">
              {previewing ? <Loader2 className="spin-icon" size={17} /> : <ClipboardCheck size={17} />}
              {previewing ? "正在校验" : "预览持仓变化"}
            </button>
            <button className="control-action primary" disabled={!preview?.valid || applying} onClick={apply} type="button">
              {applying ? <Loader2 className="spin-icon" size={17} /> : <Save size={17} />}
              {applying ? "正在更新" : "确认更新真实持仓"}
            </button>
          </div>
        </div>
        <HoldingsPreview current={workspace.holdings} preview={preview} />
      </section>
    </div>
  );
}

function ExecutionRowEditor({ row, onChange }: { row: ExecutionFillRow; onChange: (rowId: number, field: keyof ExecutionFillRow, value: string) => void }) {
  const status = String(row.fill_status || "PENDING").toUpperCase();
  const executed = status === "FILLED" || status === "PARTIAL";
  return <tr>
    <td><strong>{row.instrument}</strong></td>
    <td><span className={`execution-side side-${String(row.side).toLowerCase()}`}>{row.side === "BUY" ? "买入" : row.side === "SELL" ? "卖出" : row.side}</span></td>
    <td>{formatNumber(row.planned_order_shares)}</td>
    <td><select value={status} onChange={(event) => onChange(row.row_id, "fill_status", event.target.value)}>{FILL_STATUSES.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></td>
    <td><input disabled={!executed} inputMode="numeric" min="0" type="number" value={fieldValue(row.executed_shares)} onChange={(event) => onChange(row.row_id, "executed_shares", event.target.value)} /></td>
    <td><input disabled={!executed} inputMode="decimal" min="0" step="0.01" type="number" value={fieldValue(row.executed_price)} onChange={(event) => onChange(row.row_id, "executed_price", event.target.value)} /></td>
    <td><input disabled={!executed} type="date" value={fieldValue(row.actual_trade_date)} onChange={(event) => onChange(row.row_id, "actual_trade_date", event.target.value)} /></td>
    <td><input disabled={!executed} inputMode="decimal" min="0" step="0.01" title="佣金" type="number" value={fieldValue(row.commission_cost)} onChange={(event) => onChange(row.row_id, "commission_cost", event.target.value)} /></td>
    <td><input placeholder="成交备注" value={fieldValue(row.fill_note)} onChange={(event) => onChange(row.row_id, "fill_note", event.target.value)} /></td>
  </tr>;
}

function HoldingsPreview({ current, preview }: { current: ExecutionHoldingRow[]; preview: ExecutionPreview | null }) {
  const rows = preview?.updated_holdings ?? current;
  const currentMap = new Map(current.map((row) => [row.instrument, Number(row.shares)]));
  return <div className="panel execution-holdings-panel">
    <div className="section-heading"><div className="section-heading-title"><WalletCards size={18} /><h3>{preview?.valid ? "更新后持仓" : "当前持仓"}</h3></div><span>{rows.length} 只</span></div>
    <div className="holdings-mini-list">
      {rows.length ? rows.map((row) => {
        const before = currentMap.get(row.instrument) ?? 0;
        const delta = Number(row.shares) - before;
        return <div key={row.instrument}><strong>{row.instrument}</strong><span>{formatNumber(row.shares)} 股</span>{preview?.valid && delta !== 0 && <small className={delta > 0 ? "positive" : "negative"}>{delta > 0 ? "+" : ""}{formatNumber(delta)}</small>}</div>;
      }) : <p className="execution-muted">当前没有持仓。</p>}
    </div>
  </div>;
}

function IssuePanel({ issues }: { issues: string[] }) {
  return <div className="execution-issues"><AlertTriangle size={19} /><div><strong>校验未通过</strong>{issues.map((issue) => <span key={issue}>{translateIssue(issue)}</span>)}</div></div>;
}

function ExecutionState({ icon, title, detail, tone = "neutral" }: { icon: ReactNode; title: string; detail: string; tone?: "neutral" | "danger" }) {
  return <section className={`state-panel execution-state state-${tone}`}>{icon}<div><h2>{title}</h2><p>{detail}</p></div></section>;
}

function ExecutionFact({ label, value }: { label: string; value: string }) {
  return <div className="execution-fact"><span>{label}</span><strong>{value}</strong></div>;
}

function fieldValue(value: unknown) {
  return value === null || value === undefined || Number.isNaN(value) ? "" : String(value);
}

function formatNumber(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 2 }).format(number) : "—";
}

function translateIssue(issue: string) {
  if (issue.startsWith("pending_fill_status")) return `仍有待处理订单：${issue.split(":").slice(1).join(":")}`;
  if (issue.startsWith("executed_shares_missing")) return `缺少实际成交股数：${issue.split(":").slice(1).join(":")}`;
  if (issue.startsWith("executed_shares_exceeds_planned")) return `实际成交超过计划数量：${issue.split(":").slice(1).join(":")}`;
  if (issue.startsWith("fill_would_make_negative_position")) return `卖出数量超过当前持仓：${issue.split(":").slice(1).join(":")}`;
  if (issue.startsWith("invalid_fill")) return `成交记录格式无效：${issue}`;
  return issue;
}
