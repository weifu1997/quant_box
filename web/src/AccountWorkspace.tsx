import { AlertTriangle, CheckCircle2, Loader2, Plus, RefreshCw, Save, ShieldCheck, Trash2, WalletCards } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { applyAccountUpdate, fetchAccountWorkspace, previewAccountUpdate } from "./api";
import type { AccountFormData, AccountHoldingRow, AccountPreview, AccountWorkspaceData } from "./types";

export default function AccountWorkspace() {
  const [workspace, setWorkspace] = useState<AccountWorkspaceData | null>(null);
  const [account, setAccount] = useState<AccountFormData | null>(null);
  const [holdings, setHoldings] = useState<AccountHoldingRow[]>([]);
  const [preview, setPreview] = useState<AccountPreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [checking, setChecking] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const load = useCallback(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetchAccountWorkspace(controller.signal)
      .then((data) => {
        setWorkspace(data);
        setAccount(data.account);
        setHoldings(data.holdings);
        setPreview(null);
      })
      .catch((err: Error) => {
        if (err.name !== "AbortError") setError(err.message);
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, []);

  useEffect(() => load(), [load]);

  const invalidate = () => {
    setPreview(null);
    setSuccess(null);
  };

  const updateAccount = (field: keyof AccountFormData, value: string) => {
    setAccount((current) => current ? { ...current, [field]: value } : current);
    invalidate();
  };

  const updateHolding = (index: number, field: keyof AccountHoldingRow, value: string) => {
    setHoldings((current) => current.map((row, rowIndex) => rowIndex === index ? { ...row, [field]: value } : row));
    invalidate();
  };

  const check = () => {
    if (!account) return;
    setChecking(true);
    setError(null);
    previewAccountUpdate(account, holdings)
      .then(setPreview)
      .catch((err: Error) => setError(err.message))
      .finally(() => setChecking(false));
  };

  const save = () => {
    if (!account || !preview?.valid) return;
    if (!window.confirm("确认保存账户与真实持仓？系统会先备份已有文件。")) return;
    setSaving(true);
    setError(null);
    applyAccountUpdate(account, holdings)
      .then((result) => {
        setSuccess(result.backup_dir ? `${result.message} 旧文件已备份。` : result.message);
        load();
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setSaving(false));
  };

  if (loading) return <section className="state-panel execution-state"><Loader2 className="spin-icon" size={24} /><div><h2>正在读取账户设置</h2><p>加载本地账户和真实持仓文件。</p></div></section>;
  if (!workspace || !account) return <section className="state-panel execution-state state-danger"><AlertTriangle size={24} /><div><h2>账户工作区不可用</h2><p>{error || "账户数据读取失败。"}</p></div></section>;

  return <div className="account-page">
    <section className="account-hero"><div><span className="overview-kicker">正式交易输入</span><h2>账户与真实持仓</h2><p>这些信息决定人工交易单是否可执行。Web 端只管理账户规模、现金、仓位上限、交易手数和持仓，不读取或显示 Tushare token。</p></div><button className="icon-button command-button" onClick={load} type="button"><RefreshCw size={17} />重新读取</button></section>
    {error && <p className="inline-error">{error}</p>}
    {success && <p className="execution-success"><CheckCircle2 size={17} />{success}</p>}

    <div className="account-layout">
      <section className="panel account-form-panel">
        <div className="section-heading"><div className="section-heading-title"><WalletCards size={18} /><h3>账户参数</h3></div><span>{workspace.account_file_exists ? "已配置" : "待创建"}</span></div>
        <div className="account-form-grid">
          <AccountField label="总资产" value={account.total_asset} onChange={(value) => updateAccount("total_asset", value)} />
          <AccountField label="可用现金" value={account.cash} onChange={(value) => updateAccount("cash", value)} />
          <AccountField label="单股最大仓位" value={account.max_position_pct ?? ""} onChange={(value) => updateAccount("max_position_pct", value)} placeholder="留空表示不限制" step="0.01" />
          <AccountField label="普通交易手数" value={account.lot_size} onChange={(value) => updateAccount("lot_size", value)} step="1" />
          <AccountField label="科创板交易手数" value={account.star_market_lot_size} onChange={(value) => updateAccount("star_market_lot_size", value)} step="1" />
        </div>
        <p className="account-path">保存位置：{workspace.account_file}</p>
      </section>

      <section className="panel account-holdings-panel">
        <div className="section-heading"><div className="section-heading-title"><ShieldCheck size={18} /><h3>真实持仓</h3></div><button className="account-add" onClick={() => { setHoldings((current) => [...current, { instrument: "", shares: "" }]); invalidate(); }} type="button"><Plus size={15} />新增持仓</button></div>
        <div className="account-holdings-list">
          {holdings.map((row, index) => <div className="account-holding-row" key={`${index}-${row.instrument}`}><input aria-label={`第 ${index + 1} 行股票代码`} placeholder="000001.SZ" value={String(row.instrument ?? "")} onChange={(event) => updateHolding(index, "instrument", event.target.value.toUpperCase())} /><input aria-label={`第 ${index + 1} 行持仓股数`} min="0" placeholder="股数" step="1" type="number" value={String(row.shares ?? "")} onChange={(event) => updateHolding(index, "shares", event.target.value)} /><button aria-label={`删除 ${row.instrument || `第 ${index + 1} 行`}`} onClick={() => { setHoldings((current) => current.filter((_, rowIndex) => rowIndex !== index)); invalidate(); }} type="button"><Trash2 size={15} /></button></div>)}
          {!holdings.length && <p className="execution-muted">当前为空仓。可直接保存空持仓，或添加股票代码和股数。</p>}
        </div>
        <p className="account-path">保存位置：{workspace.holdings_file}</p>
      </section>
    </div>

    <section className="panel account-review-panel">
      <div className="section-heading"><div className="section-heading-title"><ShieldCheck size={18} /><h3>保存前校验</h3></div>{preview && <span>{preview.valid ? "通过" : `${preview.issues.length} 个问题`}</span>}</div>
      {!preview && <p className="execution-muted">校验会检查资产、现金、仓位比例、重复股票、代码格式、负持仓和交易手数倍数。</p>}
      {preview?.valid && <div className="execution-valid"><CheckCircle2 size={20} /><div><strong>校验通过</strong><span>{preview.position_count} 只持仓，共 {preview.holding_shares.toLocaleString("zh-CN")} 股</span></div></div>}
      {preview && !preview.valid && <div className="execution-issues"><AlertTriangle size={19} /><div><strong>校验未通过</strong>{preview.issues.map((issue) => <span key={issue}>{translateAccountIssue(issue)}</span>)}</div></div>}
      <div className="execution-action-row"><button className="control-action" disabled={checking || saving} onClick={check} type="button">{checking ? <Loader2 className="spin-icon" size={17} /> : <ShieldCheck size={17} />}{checking ? "校验中" : "校验账户与持仓"}</button><button className="control-action primary" disabled={!preview?.valid || saving} onClick={save} type="button">{saving ? <Loader2 className="spin-icon" size={17} /> : <Save size={17} />}{saving ? "保存中" : "确认保存"}</button></div>
    </section>
  </div>;
}

function AccountField({ label, value, onChange, placeholder, step = "0.01" }: { label: string; value: unknown; onChange: (value: string) => void; placeholder?: string; step?: string }) {
  return <label><span>{label}</span><input min="0" placeholder={placeholder} step={step} type="number" value={value === null || value === undefined ? "" : String(value)} onChange={(event) => onChange(event.target.value)} /></label>;
}

function translateAccountIssue(issue: string) {
  if (issue === "account_total_asset_not_positive") return "总资产必须大于 0";
  if (issue === "account_cash_negative") return "可用现金不能为负数";
  if (issue.startsWith("invalid_instrument")) return `股票代码格式错误：${issue.split(":").slice(1).join(":")}`;
  if (issue.startsWith("duplicate_instrument")) return `股票代码重复：${issue.split(":").slice(1).join(":")}`;
  if (issue.startsWith("negative_shares")) return `持仓股数不能为负：${issue.split(":").slice(1).join(":")}`;
  if (issue.startsWith("fractional_shares")) return `持仓股数必须是整数：${issue.split(":").slice(1).join(":")}`;
  if (issue.startsWith("shares_not_lot_multiple")) return `持仓不符合交易手数：${issue.split(":").slice(1).join(":")}`;
  return issue;
}
