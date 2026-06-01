# quant_box

本项目实现“tushare HTTP 代理 + Qlib Alpha158 因子 + 简单排序选股”的无训练版量化流程。它不包含 LightGBM 训练，核心流程是：更新沪深300日线数据、转换本地数据、计算 Alpha158 因子、用合成因子排序选股、执行轻量回测并生成每日调仓信号。

## 目录

- `config/settings.yaml`：tushare 代理、数据路径、策略参数和回测参数。
- `src/data_fetcher.py`：通过 tushare HTTP 代理获取日线和沪深300成分股。
- `src/data_converter.py`：把原始 CSV 转换成 Qlib 原生 `.bin` provider 数据，同时生成 parquet 缓存和价格矩阵。
- `src/factor_calculator.py`：调用 Qlib Alpha158 并缓存因子。
- `src/strategy.py`：合成因子、排序选股、每日最多换 1 只等约束。
- `src/backtest.py`：轻量级本地回测和绩效指标。
- `src/signal_generator.py`：生成每日调仓信号与最新持仓。
- `scripts/`：命令行入口。

## 准备

建议使用 Python 3.8 到 3.11。Qlib 在 Windows 上安装可能受版本影响，如果安装失败，优先使用 Python 3.8 环境。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

然后编辑 `config/settings.yaml`：

- `tushare.http_url`：你的第三方 tushare HTTP 代理地址。
- `tushare.token`：代理需要 token 时填写。
- `data.start_date` / `data.end_date`：数据区间。
- `strategy.top_n`：默认持有 7 只。
- `strategy.max_turnover`：默认每日最多换 1 只。

如果代理不能直接返回沪深300成分股，请把本地成分股文件放在 `data/raw/hs300_constituents.csv`，至少包含 `ts_code` 或 `con_code` 列。

## 运行

更新日线数据：

```powershell
python scripts/run_update_data.py
```

只更新指定股票：

```powershell
python scripts/run_update_data.py --codes 000001.SZ 600519.SH
```

转换本地数据：

```powershell
python scripts/run_convert_data.py
```

计算并缓存 Alpha158：

```powershell
python scripts/run_calc_factors.py
```

运行回测：

```powershell
python scripts/run_backtest.py
```

运行参数优化：

```powershell
python scripts/run_optimize.py
```

生成每日信号：

```powershell
python scripts/run_daily_signal.py --date 2026-06-01
```

一键执行完整流程：

```powershell
.\run_all.bat
```

## 输出

- `data/raw/*.csv`：原始日线数据。
- `data/qlib_data/`：转换后的 Qlib provider 数据目录，包含 `calendars`、`instruments` 和 `.bin` 特征文件。
- `data/factors/alpha158.parquet`：Alpha158 因子缓存。
- `data/prices/close.parquet`：回测使用的收盘价矩阵。
- `outputs/backtest_*.csv` / `outputs/backtest_metrics.json`：回测结果。
- `outputs/optimization_results.csv`：参数搜索结果。
- `outputs/factor_ic_summary.csv`：因子 IC 统计。
- `outputs/signal_YYYY-MM-DD.csv`：当日调仓信号。
- `outputs/latest_holdings.csv`：最新持仓。

## 注意

当前转换器会写出 Qlib 0.9.7 可读取的日频 `.bin` 特征文件，包括 `open`、`high`、`low`、`close`、`volume`、`amount` 和 `vwap`。如果未来升级 Qlib 后 provider 格式变化，再根据新版本调整转换器。
