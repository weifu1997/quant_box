import { AlertTriangle, ArrowDownRight, ArrowUpRight, Clock3, Database, RefreshCw, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { fetchStockDetail } from "./api";
import type { StockDetail } from "./types";

export default function StockDetailModal({ instrument, onClose }: { instrument: string; onClose: () => void }) {
  const [detail, setDetail] = useState<StockDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshCount, setRefreshCount] = useState(0);
  const modalRef = useRef<HTMLElement | null>(null);
  const closeRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetchStockDetail(instrument, controller.signal)
      .then(setDetail)
      .catch((err: Error) => {
        if (err.name !== "AbortError") setError(err.message);
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [instrument, refreshCount]);

  useEffect(() => {
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      } else if (event.key === "Tab" && modalRef.current) {
        trapModalFocus(event, modalRef.current);
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
      previousFocus?.focus();
    };
  }, [onClose]);

  const direction = Number(detail?.change ?? 0);
  const tone = direction > 0 ? "positive" : direction < 0 ? "negative" : "neutral";
  return (
    <div className="stock-modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section
        aria-labelledby="stock-detail-title"
        aria-modal="true"
        className="stock-modal"
        ref={modalRef}
        role="dialog"
      >
        <header className="stock-modal-header">
          <div>
            <span className="overview-kicker">股票详情</span>
            <h2 id="stock-detail-title">{detail?.name || instrument}</h2>
            <p>{instrument}</p>
          </div>
          <button aria-label="关闭股票详情" className="stock-modal-close" onClick={onClose} ref={closeRef} type="button">
            <X size={20} />
          </button>
        </header>

        {loading && !detail ? (
          <StockState title="正在读取股票行情" message={`正在查询 ${instrument} 的最新价格。`} />
        ) : error && !detail ? (
          <StockState title="股票行情暂不可用" message={error} tone="danger">
            <button className="icon-button command-button" onClick={() => setRefreshCount((value) => value + 1)} type="button">
              <RefreshCw size={17} />重新读取
            </button>
          </StockState>
        ) : detail ? (
          <div className="stock-modal-content">
            <div className="stock-modal-price-row">
              <div className="stock-modal-price">
                <span>当前价</span>
                <strong>{formatPrice(detail.price)}</strong>
                <div className={`stock-change stock-change-${tone}`}>
                  {direction > 0 ? <ArrowUpRight size={17} /> : direction < 0 ? <ArrowDownRight size={17} /> : null}
                  <span>{formatSigned(detail.change)} · {formatPercent(detail.change_pct)}</span>
                </div>
              </div>
              <div className="stock-modal-actions">
                <span className={`quote-status quote-status-${detail.status}`}>
                  {detail.is_live ? <RefreshCw size={14} /> : <Database size={14} />}
                  {detail.is_live ? "实时行情接口" : "本地收盘价 · 非实时"}
                </span>
                <button
                  className="icon-button command-button stock-refresh-button"
                  disabled={loading}
                  onClick={() => setRefreshCount((value) => value + 1)}
                  type="button"
                >
                  <RefreshCw className={loading ? "spin-icon" : ""} size={17} />刷新行情
                </button>
              </div>
            </div>

            {error && <div className="stock-inline-warning"><AlertTriangle size={17} />刷新失败，仍显示上一次行情：{error}</div>}

            <div className="stock-quote-grid">
              <QuoteMetric label="昨收" value={formatPrice(detail.pre_close)} />
              <QuoteMetric label="今开" value={formatPrice(detail.open)} />
              <QuoteMetric label="最高" value={formatPrice(detail.high)} />
              <QuoteMetric label="最低" value={formatPrice(detail.low)} />
              <QuoteMetric label="成交量（手）" value={formatVolume(detail.volume)} />
              <QuoteMetric label="行情日期" value={detail.market_date || (detail.is_live ? "当前交易时段" : "—")} />
            </div>

            <div className={`stock-source-card stock-source-${detail.status}`}>
              <Clock3 size={18} />
              <div>
                <strong>获取时间：{formatDateTime(detail.retrieved_at)}</strong>
                <p>{detail.message}</p>
              </div>
            </div>
          </div>
        ) : null}
      </section>
    </div>
  );
}

function trapModalFocus(event: KeyboardEvent, modal: HTMLElement) {
  const focusable = Array.from(
    modal.querySelectorAll<HTMLElement>('button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])')
  );
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

function QuoteMetric({ label, value }: { label: string; value: string }) {
  return <div className="stock-quote-metric"><span>{label}</span><strong>{value}</strong></div>;
}

function StockState({ title, message, tone = "normal", children }: { title: string; message: string; tone?: "normal" | "danger"; children?: ReactNode }) {
  return <div className={`stock-state stock-state-${tone}`}><h3>{title}</h3><p>{message}</p>{children}</div>;
}

function formatPrice(value: number | null | undefined) {
  return value == null || !Number.isFinite(Number(value)) ? "—" : Number(value).toFixed(2);
}

function formatSigned(value: number | null | undefined) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  const numeric = Number(value);
  return `${numeric > 0 ? "+" : ""}${numeric.toFixed(2)}`;
}

function formatPercent(value: number | null | undefined) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  const numeric = Number(value);
  return `${numeric > 0 ? "+" : ""}${numeric.toFixed(2)}%`;
}

function formatVolume(value: number | null | undefined) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 2 }).format(Number(value));
}

function formatDateTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value || "—";
  return new Intl.DateTimeFormat("zh-CN", { dateStyle: "medium", timeStyle: "medium", hour12: false }).format(date);
}
