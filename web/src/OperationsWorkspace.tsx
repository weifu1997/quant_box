import { Activity, AlertTriangle, BarChart3, ChevronDown, ChevronUp, Database, FlaskConical, Loader2, Play, RefreshCw, Settings2, Square, TimerReset, Workflow } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { fetchDashboardWorkflows, startDashboardJob, stopDashboardJob } from "./api";
import type { DashboardJob, DashboardWorkflow, WorkflowParameter } from "./types";

const CATEGORY_META: Record<string, { label: string; description: string }> = {
  data: { label: "数据准备", description: "行情、点时数据、财务缓存和历史股票池" },
  pipeline: { label: "数据加工", description: "价格面板转换和 Alpha158 因子计算" },
  research: { label: "研究验证", description: "因子诊断、参数优化、真实化回测和研究报告" },
  signal: { label: "信号生产", description: "安全生成候选信号和人工交易产物" },
  advanced: { label: "高级研究", description: "有界风险精炼、状态探针和年度路由网格" }
};

export default function OperationsWorkspace({ jobs, onJobStarted, onJobsRefresh }: { jobs: DashboardJob[]; onJobStarted: (job: DashboardJob) => void; onJobsRefresh: () => void }) {
  const [workflows, setWorkflows] = useState<DashboardWorkflow[]>([]);
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [parameterValues, setParameterValues] = useState<Record<string, Record<string, unknown>>>({});
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const activeJob = jobs.find((job) => job.status === "running" || job.status === "stopping") ?? null;
  const latestJob = activeJob ?? jobs[0] ?? null;

  useEffect(() => {
    const controller = new AbortController();
    fetchDashboardWorkflows(controller.signal)
      .then((items) => {
        setWorkflows(items);
        setParameterValues(Object.fromEntries(items.map((item) => [item.action, Object.fromEntries((item.parameters ?? []).map((parameter) => [parameter.name, parameter.default ?? ""]))])));
      })
      .catch((err: Error) => {
        if (err.name !== "AbortError") setError(err.message);
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, []);

  const groups = useMemo(() => Object.entries(CATEGORY_META).map(([id, meta]) => ({ id, ...meta, items: workflows.filter((item) => item.category === id) })), [workflows]);

  const run = (workflow: DashboardWorkflow) => {
    setPending(workflow.action);
    setError(null);
    startDashboardJob({ action: workflow.action as DashboardJob["action"], parameters: parameterValues[workflow.action] ?? {} })
      .then((job) => {
        onJobStarted(job);
        onJobsRefresh();
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setPending(null));
  };

  const stop = () => {
    if (!activeJob) return;
    setStopping(true);
    setError(null);
    stopDashboardJob(activeJob.id)
      .then((job) => {
        onJobStarted(job);
        onJobsRefresh();
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setStopping(false));
  };

  return <div className="operations-page">
    <section className="operations-hero">
      <div><span className="overview-kicker">跨平台量化操作中心</span><h2>从数据更新到候选信号，一站式运行</h2><p>所有按钮都映射到后端固定白名单命令，不接受任意脚本路径或 Shell 参数。Windows 与 Ubuntu 都使用当前 Python 解释器和项目路径启动任务。</p></div>
      <div className="operations-platform"><Workflow size={22} /><strong>Windows · Ubuntu</strong><span>统一任务、日志、停止与状态接口</span></div>
    </section>

    {error && <p className="inline-error">{error}</p>}
    {activeJob && <section className="operations-active"><Activity className="spin-icon" size={19} /><div><strong>{activeJob.label}</strong><span>{activeJob.message}</span></div><button className="control-action danger" disabled={stopping || activeJob.status === "stopping"} onClick={stop} type="button"><Square size={15} />{stopping ? "正在停止" : "停止任务"}</button></section>}

    {loading ? <section className="state-panel execution-state"><Loader2 className="spin-icon" size={24} /><div><h2>正在加载操作目录</h2><p>读取后端白名单工作流。</p></div></section> : groups.map((group) => group.items.length > 0 && <section className="operations-group" key={group.id}>
      <div className="operations-group-heading"><CategoryIcon category={group.id} /><div><h3>{group.label}</h3><p>{group.description}</p></div><span>{group.items.length} 项</span></div>
      <div className="workflow-card-grid">{group.items.map((workflow) => <article className={`workflow-card ${expanded === workflow.action ? "workflow-card-expanded" : ""}`} key={workflow.action}>
        <div><strong>{workflow.label}</strong><p>{workflow.description}</p></div>
        {expanded === workflow.action && workflow.parameters?.length > 0 && <div className="workflow-parameters">{workflow.parameters.map((parameter) => <WorkflowParameterField key={parameter.name} parameter={parameter} value={parameterValues[workflow.action]?.[parameter.name]} onChange={(value) => setParameterValues((current) => ({ ...current, [workflow.action]: { ...(current[workflow.action] ?? {}), [parameter.name]: value } }))} />)}</div>}
        <div className="workflow-card-footer"><span><TimerReset size={14} />{workflow.duration}</span><div>{workflow.parameters?.length > 0 && <button className="workflow-config-button" onClick={() => setExpanded((current) => current === workflow.action ? null : workflow.action)} type="button"><Settings2 size={14} />配置{expanded === workflow.action ? <ChevronUp size={13} /> : <ChevronDown size={13} />}</button>}<button disabled={Boolean(activeJob || pending)} onClick={() => run(workflow)} type="button">{pending === workflow.action ? <Loader2 className="spin-icon" size={15} /> : <Play size={15} />}{pending === workflow.action ? "启动中" : "运行"}</button></div></div>
      </article>)}</div>
    </section>)}

    {latestJob && <section className="panel operations-log"><div className="section-heading"><div className="section-heading-title"><RefreshCw size={18} /><h3>最近任务日志</h3></div><span>{latestJob.status}</span></div><strong>{latestJob.label}</strong><p>{latestJob.message}</p><pre>{latestJob.log_tail?.length ? latestJob.log_tail.join("\n") : "暂无日志输出"}</pre></section>}

    <p className="operations-warning"><AlertTriangle size={17} />预计时长取决于本地缓存、网络和股票数量。首次历史数据、点时数据或财务数据补齐可能达到小时级；重复运行会优先复用已有缓存。</p>
  </div>;
}

function CategoryIcon({ category }: { category: string }) {
  if (category === "data") return <Database size={20} />;
  if (category === "pipeline") return <Workflow size={20} />;
  if (category === "research") return <FlaskConical size={20} />;
  return <BarChart3 size={20} />;
}

function WorkflowParameterField({ parameter, value, onChange }: { parameter: WorkflowParameter; value: unknown; onChange: (value: unknown) => void }) {
  if (parameter.type === "boolean") {
    return <label className="workflow-parameter-toggle"><input checked={Boolean(value)} type="checkbox" onChange={(event) => onChange(event.target.checked)} /><span><strong>{parameter.label}</strong>{parameter.help && <small>{parameter.help}</small>}</span></label>;
  }
  const inputType = parameter.type === "date" ? "date" : parameter.type === "integer" || parameter.type === "number" ? "number" : "text";
  return <label className="workflow-parameter-field"><span>{parameter.label}</span><input max={parameter.max} min={parameter.min} placeholder={parameter.optional ? "留空使用配置" : undefined} step={parameter.type === "integer" ? 1 : parameter.type === "number" ? "any" : undefined} type={inputType} value={value === null || value === undefined ? "" : String(value)} onChange={(event) => onChange(event.target.value)} />{parameter.help && <small>{parameter.help}</small>}</label>;
}
