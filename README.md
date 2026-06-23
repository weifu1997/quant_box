# quant_box

`quant_box` 是一个面向 A 股主板股票的本地量化研究与手动交易信号项目。它不直接接入券商 API，也不做自动下单；核心目标是通过本地数据、Qlib Alpha158 因子、IC 加权选股、真实化回测和每日信号文件，辅助你进行手动交易决策。

项目当前支持：

- 通过 Tushare HTTP 代理补齐 A 股主板日线数据
- 将原始 CSV 转换为 Qlib provider 数据和本地价格面板
- 计算并缓存 Alpha158 因子
- 使用 rolling IC 动态权重生成综合因子分数
- 运行考虑滑点、手续费、最低佣金、过户费、容量限制、分板块涨跌停、长期停牌冻结/可选折价退出的回测
- 生成最新调仓信号和最新持仓文件
- 运行 walk-forward 参数优化
- Windows 双击 `.bat` 一键启动常用流程

## 目录结构

```text
quant_box/
  config/
    settings.yaml              默认配置，可提交
    settings.local.yaml        本地私密配置，不提交
  data/
    raw/                       原始日线 CSV，本地缓存，不提交
    qlib_data/                 Qlib provider 数据，不提交
    factors/                   因子和 IC 权重缓存，不提交
    prices/                    回测价格面板，不提交
  outputs/                     回测、优化、信号、进度输出，不提交
  scripts/                     命令行入口
  src/                         核心代码
  tests/                       自动化测试
  *.bat                        Windows 一键脚本
```

## 新电脑快速开始

先拉取项目：

```powershell
git clone https://github.com/weifu1997/quant_box.git
cd quant_box
```

然后双击：

```text
00_安装依赖环境.bat
```

这个脚本会自动执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果需要复现 CI 使用的版本，可以改用直接依赖锁定文件：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

如果 `python` 命令不可用，脚本会自动尝试 `py -3`。

## 本地 Web 仪表盘

Web 仪表盘是本地复核控制台，用来查看最新一次自动信号运行是否可以进入人工交易复核。首屏展示结构化摘要、质量门槛、阻塞原因、订单摘要和报告链接；同时提供受控运行按钮，只能启动白名单后台任务：

- 补齐 `daily_basic` 点时数据缺口：调用 `scripts\run_update_point_in_time_data.py`，并跳过指数成分和 ST 日历更新。
- 重跑自动信号：可选择“候选输出”或“正常门槛输出”；正常模式不会附加 `--candidate-only`，但仍遵守自动流程质量门槛。

后台任务状态写入 `outputs/dashboard_jobs/`，日志写入 `outputs/logs/dashboard_job_*.log`。仪表盘不会编辑配置、推广候选信号、应用成交回填或直接更新持仓。

日常使用可以直接双击：

```text
15_启动Web仪表盘.bat
```

脚本会启动 FastAPI 后端和 React/Vite 前端，检查 `http://127.0.0.1:8000/api/health` 与 `http://127.0.0.1:5173` 可访问，然后打开浏览器。若 `web/node_modules` 不存在，会先在 `web/` 下执行 `npm install`。

启动后端：

```powershell
.\.venv\Scripts\python.exe scripts\run_dashboard.py
```

另开一个终端启动前端：

```powershell
cd web
npm install
npm run dev
```

浏览器打开 `http://127.0.0.1:5173`。前端会把 `/api` 请求代理到 `http://127.0.0.1:8000` 的 FastAPI 后端。

## 配置 Tushare 代理

不要把真实 URL 和 token 写进 `config/settings.yaml` 后提交。建议新建：

```text
config/settings.local.yaml
```

示例：

```yaml
tushare:
  http_url: "http://你的代理地址:端口/"
  token: "你的token"
```

`config/settings.local.yaml` 已经在 `.gitignore` 中，不会被提交。

也可以使用环境变量：

```powershell
setx TUSHARE_HTTP_URL "你的代理地址"
setx TUSHARE_TOKEN "你的token"
```

检查配置是否能被读取，双击：

```text
01_检查Tushare配置.bat
```

这个检查不会发网络请求，也不会打印 token。

## 一键脚本

不用把所有 `.bat` 顺序执行；按目标选入口即可。

| 脚本 | 作用 |
| --- | --- |
| `00_安装依赖环境.bat` | 新电脑首次安装 `.venv` 和依赖 |
| `01_检查Tushare配置.bat` | 检查 Tushare HTTP 代理配置是否可读取 |
| `02_快速更新并生成信号.bat` | 日常快速入口：更新数据、转换、重算因子，跳过重调参与完整回测并生成最新信号 |
| `03_运行测试.bat` | 运行自动化测试 |
| `04_补齐股票数据_持续.bat` | 分步工具：增量补齐缺失或过期的主板股票日线数据 |
| `05_查看补齐进度.bat` | 分步工具：查看 raw 数据是否补到目标日期、最新覆盖率和补齐进度 JSON |
| `06_转换数据.bat` | 分步工具：将 `data/raw/*.csv` 转为 Qlib 数据和价格面板 |
| `07_计算因子.bat` | 分步工具：计算或读取 Alpha158 因子缓存 |
| `08_参数优化.bat` | 分步工具：运行 walk-forward 参数优化 |
| `09_运行回测.bat` | 分步工具：运行当前配置下的回测 |
| `10_生成最新信号.bat` | 分步工具：基于最新因子生成候选手动交易信号；如需覆盖正式持仓，传入 `--official` |
| `12_全量重刷股票数据.bat` | 维护工具：从配置起始日或上市日全量重刷 raw 股票数据，慢于 `04`，仅在历史数据疑似损坏时使用 |
| `13_一键补齐历史数据_无人值守.bat` | 无人值守工具：分批补齐 2012 年以来 raw 日线，转换数据，补齐 daily_basic，并可重算 Alpha158 因子 |
| `14_构建历史股票池.bat` | 分步工具：用 Tushare `index_weight` 构建沪深300 + 中证500 + 中证1000权重前300的历史股票池快照 |
| `15_启动Web仪表盘.bat` | 本地复核控制台入口：启动 FastAPI 后端和 React/Vite 前端，并打开可查看复核结果、受控修复缺口和重跑信号的 Web 仪表盘 |
| `scripts/run_update_point_in_time_data.py` | 命令行工具：补齐 daily_basic、HS300 指数成分权重和 ST 历史日历，并重写点时数据治理报告 |
| `scripts/run_update_fundamentals.py` | 命令行工具：补齐 fina_indicator 和 dividend 基本面缓存，用于质量、分红和负债筛选 |
| `scripts/run_fundamental_screen.py` | 命令行工具：生成基本面筛选 CSV 和 Markdown 解释报告 |
| `run_all.bat` | 命令行自动全流程入口：刷新缺失和过期股票、转换、重算因子、data health、回测、候选信号；不含 walk-forward 参数优化 |

常用入口：

```text
02_快速更新并生成信号.bat      日常更新并生成信号
03_运行测试.bat                只跑自动化测试
13_一键补齐历史数据_无人值守.bat  下班后无人值守补齐长周期历史数据
14_构建历史股票池.bat          生成点时历史股票池快照
15_启动Web仪表盘.bat           打开每日信号复核控制台
run_all.bat                    自动全流程：刷新缺失和过期数据 + data health + 回测 + 候选信号
04 -> 06 -> 07 -> 08 -> 09 -> 10  完整研究流程
```

如果代码更新后或运行异常，再双击 `03_运行测试.bat`。如果需要排查某一步，再使用 `04` 到 `10` 的分步工具。

## 补齐股票数据

双击：

```text
04_补齐股票数据_持续.bat
```

默认参数：

```powershell
--chunk-size 300 --sleep-seconds 0
```

含义：

- 每批补 300 只缺失或过期股票
- 批间等待 1 秒
- 自动记录进度到 `outputs/data_update_progress.json`
- 中断后再次双击，会继续补缺失或过期股票，不会从头开始
- `04` 是增量补齐，不是全量拉取；它会跳过已经补到目标交易日的股票
- 如需从头重刷历史 raw 数据，双击 `12_全量重刷股票数据.bat`，或命令行增加 `--force-full`
- 如果要离开电脑长时间补历史数据，双击 `13_一键补齐历史数据_无人值守.bat`。它会固定补到 `2026-06-04`，循环分批抓取 raw 日线，随后转换数据、补齐 `daily_basic`，并默认重算 Alpha158 因子；日志写入 `outputs/logs/history_data_backfill.log`

命令行等价写法：

```powershell
.\.venv\Scripts\python.exe scripts\run_update_data.py --chunk-size 300 --sleep-seconds 0
```

只跑一批确认状态：

```powershell
.\.venv\Scripts\python.exe scripts\run_update_data.py --chunk-size 300 --sleep-seconds 0 --max-chunks 1
```

查看 raw 数据最新覆盖率和补齐进度：

```powershell
.\.venv\Scripts\python.exe scripts\show_update_progress.py
```

默认会读取最近一次更新写入的 freshness 统计，速度更快；如需现场重新扫描 raw CSV，追加 `--scan-raw`。
如果 `latest_symbols` 略低但 `fresh_or_confirmed_symbols` 等于 `target_symbols`，表示少数停牌或无新行情股票已经查询确认，不会阻塞补齐。

补齐点时治理依赖，包括 `daily_basic`、HS300 成分权重和 ST 历史日历：

```powershell
.\.venv\Scripts\python.exe scripts\run_update_point_in_time_data.py --max-dates 20 --max-index-windows 1
```

构建点时历史股票池快照，默认使用沪深300、中证500 和中证1000成分权重，并保留中证1000权重前300。首次只做接口烟测时，把输出写到 `outputs/`，避免一批试跑数据覆盖正式股票池：

```powershell
.\.venv\Scripts\python.exe scripts\run_build_universe.py --max-index-windows 1 --index-constituents-file outputs\universe_smoke_index_constituents.csv --out-file outputs\historical_universe_smoke.csv
```

正式构建不要加 `--max-index-windows`，生成结果写入 `data/raw/historical_universe.csv`。正式构建默认遇到指数成分接口窗口错误就失败；只有临时烟测或排障时才加 `--skip-index-errors` 允许部分窗口跳过。
需要让回测和候选信号按历史股票池过滤时，在本地配置里启用：

```yaml
universe_builder:
  enabled: true
```

启用后默认要求 `data/raw/historical_universe.csv` 已存在；缺文件会直接报错，避免回测或因子信号在不知情的情况下退回全量股票池。

补齐基础财务依赖，包括 `fina_indicator` 和 `dividend`：

```powershell
.\.venv\Scripts\python.exe scripts\run_update_fundamentals.py
```

首次试跑可以限制股票数量：

```powershell
.\.venv\Scripts\python.exe scripts\run_update_fundamentals.py --max-symbols 50
```

## 数据处理与回测流程

数据补齐后，按顺序双击：

```text
06_转换数据.bat
07_计算因子.bat
08_参数优化.bat
09_运行回测.bat
10_生成最新信号.bat
```

对应命令行：

```powershell
.\.venv\Scripts\python.exe scripts\run_convert_data.py
.\.venv\Scripts\python.exe scripts\run_calc_factors.py
.\.venv\Scripts\python.exe scripts\run_factor_diagnostics.py
.\.venv\Scripts\python.exe scripts\run_optimize.py
.\.venv\Scripts\python.exe scripts\run_backtest.py
.\.venv\Scripts\python.exe scripts\run_quant_diagnostics.py
.\.venv\Scripts\python.exe scripts\run_optimization_review.py
.\.venv\Scripts\python.exe scripts\run_evidence_optimizer.py
.\.venv\Scripts\python.exe scripts\run_daily_signal.py --date latest
# 如需写入正式 signal_*.csv 和 latest_holdings.csv，加 --official
```

`08_参数优化.bat` 和 `scripts\run_optimize.py` 默认使用快速基线 walk-forward 网格：只跑 `momentum` 轻量因子、2 个参数组合、12 个月滚动步长，避免日常调参跑到几十分钟。需要 IC 加权时可传 `--factor-groups ic_weighted,momentum`；需要完整 24 组合网格时，使用：

```powershell
.\.venv\Scripts\python.exe scripts\run_optimize.py --full-grid
```

日常快速一键流程：

```text
02_快速更新并生成信号.bat
```

注意：快速流程会先更新已有股票并补齐缺失股票，再转换数据、重算因子和生成信号；它会显式跳过 walk-forward 重调参与完整回测。如果需要重调参和回测，先运行 `08_参数优化.bat` 和 `09_运行回测.bat`，或直接用命令行去掉 `--skip-optimize --skip-backtest`。

`02_快速更新并生成信号.bat` 会覆盖脚本默认值，使用 `--chunk-size 300 --sleep-seconds 0`；`scripts/run_auto_signal.py` 自身默认值仍来自配置文件。

长时间带 walk-forward 的全流程建议用后台监督入口启动，避免终端或外层工具 1 小时超时中断观察窗口：

```powershell
.\.venv\Scripts\python.exe scripts\run_auto_signal_supervised.py start -- --chunk-size 300 --sleep-seconds 0
.\.venv\Scripts\python.exe scripts\run_auto_signal_supervised.py status
.\.venv\Scripts\python.exe scripts\run_auto_signal_supervised.py tail -n 80
```

后台入口会把日志写到 `outputs/logs/auto_signal_*.log`，并持续复用 `outputs/auto_run_status.json` 作为阶段状态文件。若只想做日常信号，不需要重调参，继续用 `02_快速更新并生成信号.bat` 或 `run_all.bat`。

自动流程会先判断信号是否可执行：

- 数据覆盖率、价格面板和因子缓存必须通过健康检查
- 自动选中的参数必须通过样本外质量门槛
- 如果任一门槛不通过，只输出 `candidate_signal_*.csv` 和 `manual_orders_candidate_*.csv`，不会覆盖 `outputs/latest_holdings.csv`
- 如果门槛全部通过，输出正式 `signal_*.csv`、`manual_orders_*.csv` 和 `latest_holdings.csv`
- `--allow-low-quality` 只允许流程继续生成候选结果；如确需在低质量门槛下覆盖正式文件，必须同时传 `--force-official`
- `--candidate-only` 用于验证运行：即使所有门槛通过也只写候选输出，不覆盖正式信号/最新持仓，且不能和 `--promote-candidate` 同时使用
- 缺少或校验失败的 `config/account.yaml` / `config/current_holdings.csv` 会阻止正式输出，并在交易单备注中列出原因

可选账户与真实持仓文件：

```text
config/account.yaml
config/current_holdings.csv
```

`config/account.yaml` 示例：

```yaml
total_asset: 1000000
cash: 100000
max_position_pct: 0.2
lot_size: 100
star_market_lot_size: 200
```

`config/current_holdings.csv` 示例：

```text
instrument,shares
600519.SH,100
000001.SZ,500
```

这两个文件已加入 `.gitignore`，不会提交到 GitHub。仓库提供 `config/account.example.yaml` 和 `config/current_holdings.example.csv` 作为格式参考。没有真实持仓文件时，系统仍会生成候选交易单，但不会把订单标记为可直接执行。

人工交易单会同时输出估算和最终下单相关字段：

- `indicative_target_shares`：基于信号日可得价格的估算股数
- `final_target_shares` / `order_shares`：仅在交易日参考价格可用且账户/持仓校验通过时给出
- `is_order_actionable`：是否可直接按交易单执行
- `reference_price_source`：价格来自信号日还是预定交易日
- `suggested_limit_price`、`stop_loss_price`、`adv_10d`、`capacity_ratio`、`is_limit_up`、`is_limit_down`、`is_st`：人工执行辅助字段

## 研究诊断与人工执行闭环

自动流程不会放宽质量门槛。只要数据、参数、回测或账户持仓任一门槛不过，系统仍只输出候选信号，并在 `daily_signal_report.md` 和 `auto_signal_report.json` 里列出阻塞原因。

每日流程现在同时输出：

- 研究诊断：`auto_research_diagnostics.json`、基准净值、个股/行业归因、行业/市值暴露，用来判断策略到底赚亏在哪里
- 数据治理：`data_governance_report.json`，检查上市/退市字段、ST 历史日历、指数成分日期/权重、历史股票池来源月度覆盖、复权因子和因子缓存元数据
- 可选基本面筛选：`fundamental_screen_YYYY-MM-DD.csv` 和 `fundamental_screen_report.md`，用质量、分红、负债、估值阈值解释候选公司
- 人工确认：`outputs/order_confirmations/order_confirmation*_YYYY-MM-DD.csv`
- 成交回填：`outputs/fill_feedback/fill_feedback*_YYYY-MM-DD.csv`

建议日常顺序：

```text
1. 打开 outputs/daily_signal_report.md，看质量门槛、研究诊断和数据治理风险
2. 如果 is_executable=false，不按候选信号交易，先修 block_reasons
3. 如果允许人工执行，检查 manual_orders_*.csv 和 order_confirmation_*.csv
4. 手工下单后，在 fill_feedback_*.csv 填入 FILLED/PARTIAL/CANCELLED/SKIPPED、executed_shares、executed_price 等字段；不要保留 PENDING 行
5. 用成交回填更新 config/current_holdings.csv
```

生成基本面筛选报告：

```powershell
.\.venv\Scripts\python.exe scripts\run_fundamental_screen.py --date latest
```

这份报告暂时不改变正式交易信号；它用于先把“好公司、好价格、低负债、能分红”这层长期股权过滤独立看清楚。

成交回填命令：

```powershell
.\.venv\Scripts\python.exe scripts\run_apply_fills.py outputs\fill_feedback\fill_feedback_YYYY-MM-DD.csv
```

先演练不更新持仓：

```powershell
.\.venv\Scripts\python.exe scripts\run_apply_fills.py outputs\fill_feedback\fill_feedback_YYYY-MM-DD.csv --dry-run
```

回填命令会先校验文件：未处理的 PENDING、缺成交股数、成交股数超过计划、卖出超过当前持仓等情况会直接失败，不会静默覆盖 `config/current_holdings.csv`。

## 输出文件

常见输出：

```text
data/raw/*.csv                         原始日线数据
data/qlib_data/                        Qlib provider 数据
data/prices/close.parquet              收盘价面板
data/prices/ohlcv.parquet              OHLCV 价格面板
data/factors/alpha158.parquet          Alpha158 因子缓存
data/factors/alpha158.parquet.meta.json 因子缓存元数据
data/factors/adj_factor_meta.json      复权因子版本元数据
data/factors/rolling_ic_weights.parquet rolling IC 权重缓存
data/fundamentals/fina_indicator.parquet 基础财务指标缓存
data/fundamentals/dividend.parquet      分红缓存
outputs/factor_ic_summary.csv           Factor IC summary
outputs/factor_ic_yearly.csv            Yearly factor IC stability
outputs/factor_group_returns.csv        Factor quantile forward-return spread
outputs/backtest_equity.csv            回测净值
outputs/backtest_holdings.csv          回测持仓
outputs/backtest_trades.csv            回测成交
outputs/backtest_metrics.json          回测指标
outputs/backtest_yearly.csv            Backtest yearly return/drawdown summary
outputs/backtest_run_summary.json      Backtest run inputs, trade, cost, drawdown, and coverage summary
outputs/quant_diagnostic_report.json   Five-layer quant diagnostics summary
outputs/quant_diagnostic_report.md     Five-layer quant diagnostics report
outputs/optimization_review.json       Post-diagnostic style/risk/trading review
outputs/optimization_review.md         Human-readable optimization review
outputs/evidence_optimization_plan.json Evidence-backed style/risk/trading optimization plan
outputs/evidence_optimization_plan.md   Human-readable evidence optimization plan
outputs/logs/backtest_*.log            Backtest run logs
outputs/optimization_results.csv       参数优化结果
outputs/auto_run_status.json           自动流程阶段状态
outputs/auto_run_metrics.prom          自动流程 Prometheus textfile 指标
outputs/data_health_report.json        数据健康检查
outputs/data_health_report.csv         数据健康检查表
outputs/auto_validation_windows.csv    自动选参逐窗口验证
outputs/auto_parameter_summary.csv     自动选参汇总
outputs/auto_selected_params.json      自动选中的策略参数
outputs/auto_parameter_quality.json    自动选参质量门槛判断
outputs/auto_backtest_metrics.json     自动选参后的回测指标
outputs/auto_signal_report.json        自动信号报告
outputs/auto_signal_job.json           后台自动流程任务信息
outputs/logs/auto_signal_*.log         后台自动流程日志
outputs/dashboard_jobs/*.json          Web 仪表盘受控后台任务状态
outputs/logs/dashboard_job_*.log       Web 仪表盘受控后台任务日志
outputs/data_governance_report.json    点时数据治理检查
outputs/auto_research_diagnostics.json 研究诊断汇总
outputs/auto_research_benchmark_curve.csv 基准对比净值
outputs/auto_research_drawdown_periods.csv 回撤分段诊断
outputs/auto_research_regime_returns.csv 市场状态收益拆解
outputs/auto_research_regime_trade_costs.csv 市场状态交易成本拆解
outputs/auto_research_regime_trade_costs_by_reason.csv 市场状态交易原因成本拆解
outputs/auto_research_regime_industry_attribution.csv 市场状态-行业收益归因
outputs/auto_research_regime_instrument_attribution.csv 市场状态-个股收益归因
outputs/auto_research_instrument_attribution.csv 个股收益归因
outputs/auto_research_industry_attribution.csv 行业收益归因
outputs/auto_research_industry_exposure.csv 行业暴露
outputs/auto_research_market_cap_exposure.csv 市值暴露
outputs/fundamental_screen_YYYY-MM-DD.csv 基本面筛选结果
outputs/fundamental_screen_report.md   基本面筛选解释报告
outputs/daily_signal_report.md         每日信号 Markdown 报告
outputs/manual_orders_YYYY-MM-DD.csv   人工执行交易单
outputs/manual_orders_candidate_YYYY-MM-DD.csv 门槛未通过时的候选交易单
outputs/order_confirmations/order_confirmation_YYYY-MM-DD.csv 人工确认模板
outputs/order_confirmations/order_confirmation_candidate_YYYY-MM-DD.csv 候选人工确认模板
outputs/fill_feedback/fill_feedback_YYYY-MM-DD.csv 成交回填模板
outputs/fill_feedback/fill_feedback_candidate_YYYY-MM-DD.csv 候选成交回填模板
outputs/fill_apply_audit_YYYY-MM-DD.json 成交回填应用审计
outputs/candidate_signal_YYYY-MM-DD.csv 门槛未通过时的候选信号
outputs/signal_YYYY-MM-DD.csv          每日信号
outputs/latest_holdings.csv            最新持仓
outputs/data_update_progress.json      数据补齐进度
data/raw/failed_fetches.csv             本轮补数据失败的股票及原因
outputs/history/YYYY-MM-DD/            每次自动运行的归档快照
```

导出自动流程监控指标：

```powershell
.\.venv\Scripts\python.exe scripts\export_auto_status_metrics.py --status-file outputs\auto_run_status.json --output outputs\auto_run_metrics.prom
```

Backtest logs and summaries include run context, sanitized config snapshots, input file/data coverage, price-factor
alignment, yearly target pass/fail flags, failed-year lists, equity/drawdown/trade/cost summaries, output paths, and
exception tracebacks when a run fails.

## 策略与回测要点

当前默认配置位于 `config/settings.yaml`：

- 股票池：A 股主板 `mainboard_a`
- 可选历史股票池：启用 `universe_builder.enabled` 后，回测和候选信号会按 `data/raw/historical_universe.csv` 的点时快照过滤，默认口径为沪深300 + 中证500 + 中证1000权重前300；缺少该文件时默认阻断
- 因子组：`dynamic_ic_selector`，默认每期使用 IC 排名前 3 个候选因子做权重混合
- 调仓频率：monthly
- 默认持仓数：15
- 单次最大换手：1
- 排名缓冲：30
- 流动性候选池：默认剔除 `amount` 10 日均值底部 20% 股票，降低低流动性执行风险但保留量价因子的区分度
- 组合熔断：组合回撤 8% 后降至 30% 目标仓位，并冷却 5 个交易日
- 成本：佣金、最低佣金、印花税、过户费、0.05% 基础滑点；默认开启 ADV 占比动态滑点
- 容量限制：使用交易日前历史 ADV，避免看当日成交额
- 涨跌停：优先用 high/low 判断触板，并按主板、科创/创业、北交所、ST 分别使用阈值
- 止盈止损：默认配置 8% 个股止损，关闭固定止盈；日内触发采用保守成交价，跳空低开按开盘价成交
- 长期缺价/停牌：默认按 `stale_price_haircut` 折价退出，避免长期冻结造成虚高净值

## 可选 ML 依赖

基础 `requirements.txt` 不安装 `xgboost`。默认 ML 模型是 `ridge_numpy`，`model_type: auto` 会优先尝试已安装的模型并在缺少可选依赖时回退；如果显式设置 `model_type: xgboost`，请先手动安装：

```powershell
.\.venv\Scripts\python.exe -m pip install "xgboost>=2.0"
```

## 测试

双击：

```text
03_运行测试.bat
```

或命令行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

GitHub Actions 会在 Windows/Python 3.11 上安装 `requirements-lock.txt`，并运行项目的最小回归测试集。

## 换电脑迁移数据

GitHub 会保存代码、默认配置、脚本和测试，但不会保存：

- `.venv`
- `config/settings.local.yaml`
- `config/account.yaml`
- `config/current_holdings.csv`
- `data/`
- `outputs/`

如果想快速恢复旧电脑状态，可以手动复制旧电脑的：

```text
config/settings.local.yaml
config/account.yaml
config/current_holdings.csv
data/
outputs/
```

如果不复制数据，就在新电脑上重新双击：

```text
00_安装依赖环境.bat
02_快速更新并生成信号.bat
```

## 注意事项

- 本项目只生成手动交易信号，不负责自动下单。
- 不要提交 `config/settings.local.yaml`、`config/account.yaml`、`config/current_holdings.csv`。
- 如果数据补齐窗口长时间没有新增文件，先双击 `05_查看补齐进度.bat` 看 `latest_symbols`、`stale_or_missing_symbols`、`current_symbol` 和 `last_error`。
- 如果 `auto_signal_report.json` 里的 `is_executable` 是 `false`，不要按候选信号交易，先看 `block_reasons`。
- 大批量补齐数据是小时级任务，建议保持小批次可恢复模式运行。
