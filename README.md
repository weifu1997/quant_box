# quant_box

`quant_box` 是一个面向 A 股主板股票的本地量化研究与手动交易信号项目。它不直接接入券商 API，也不做自动下单；核心目标是通过本地数据、Qlib Alpha158 因子、IC 加权选股、真实化回测和每日信号文件，辅助你进行手动交易决策。

项目当前支持：

- 通过 Tushare HTTP 代理补齐 A 股主板日线数据
- 将原始 CSV 转换为 Qlib provider 数据和本地价格面板
- 计算并缓存 Alpha158 因子
- 使用 rolling IC 动态权重生成综合因子分数
- 运行考虑滑点、手续费、最低佣金、过户费、容量限制、涨跌停、长期停牌折价退出的回测
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

如果 `python` 命令不可用，脚本会自动尝试 `py -3`。

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

推荐按编号使用：

| 脚本 | 作用 |
| --- | --- |
| `00_安装依赖环境.bat` | 新电脑首次安装 `.venv` 和依赖 |
| `01_检查Tushare配置.bat` | 检查 Tushare HTTP 代理配置是否可读取 |
| `02_快速更新并生成信号.bat` | 日常快速入口：更新数据、转换、重算因子，跳过重调参与完整回测并生成最新信号 |
| `03_运行测试.bat` | 运行自动化测试 |
| `04_补齐股票数据_持续.bat` | 分步工具：持续补齐缺失主板股票日线数据 |
| `05_查看补齐进度.bat` | 分步工具：查看本地 raw CSV 数量和补齐进度 JSON |
| `06_转换数据.bat` | 分步工具：将 `data/raw/*.csv` 转为 Qlib 数据和价格面板 |
| `07_计算因子.bat` | 分步工具：计算或读取 Alpha158 因子缓存 |
| `08_参数优化.bat` | 运行 walk-forward 参数优化 |
| `09_运行回测.bat` | 分步工具：运行当前配置下的回测 |
| `10_生成最新信号.bat` | 分步工具：基于最新因子生成候选手动交易信号；如需覆盖正式持仓，传入 `--official` |
| `11_旧版全流程_补数据到信号.bat` | 旧版全流程：补数据到信号，但不自动采用调参结果 |

最常用的是：

```text
02_快速更新并生成信号.bat
```

旧入口 `02_自动调参并生成信号.bat` 会直接运行同一套快速流程，避免旧习惯失效。

如果代码更新后或运行异常，再双击 `03_运行测试.bat`。如果需要排查某一步，再使用 `04` 到 `10` 的分步工具。

## 补齐股票数据

双击：

```text
04_补齐股票数据_持续.bat
```

默认参数：

```powershell
--chunk-size 15 --sleep-seconds 10
```

含义：

- 每批补 15 只缺失股票
- 批间等待 10 秒
- 自动记录进度到 `outputs/data_update_progress.json`
- 中断后再次双击，会继续补缺失股票，不会从头开始

命令行等价写法：

```powershell
.\.venv\Scripts\python.exe scripts\run_update_data.py --chunk-size 15 --sleep-seconds 10
```

只跑一批确认状态：

```powershell
.\.venv\Scripts\python.exe scripts\run_update_data.py --chunk-size 15 --sleep-seconds 10 --max-chunks 1
```

查看本地 CSV 数量：

```powershell
(Get-ChildItem data\raw -Filter *.csv).Count
```

查看进度：

```powershell
Get-Content outputs\data_update_progress.json
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
.\.venv\Scripts\python.exe scripts\run_backtest.py
.\.venv\Scripts\python.exe scripts\run_daily_signal.py --date latest
# 如需写入正式 signal_*.csv 和 latest_holdings.csv，加 --official
```

日常快速一键流程：

```text
02_快速更新并生成信号.bat
```

注意：快速流程会先更新已有股票并补齐缺失股票，再转换数据、重算因子和生成信号；它会显式跳过 walk-forward 重调参与完整回测。如果需要重调参和回测，先运行 `08_参数优化.bat` 和 `09_运行回测.bat`，或直接用命令行去掉 `--skip-optimize --skip-backtest`。

`02_快速更新并生成信号.bat` 会覆盖脚本默认值，使用 `--chunk-size 15 --sleep-seconds 10`；`scripts/run_auto_signal.py` 自身默认值仍来自配置文件。

自动流程会先判断信号是否可执行：

- 数据覆盖率、价格面板和因子缓存必须通过健康检查
- 自动选中的参数必须通过样本外质量门槛
- 如果任一门槛不通过，只输出 `candidate_signal_*.csv` 和 `manual_orders_candidate_*.csv`，不会覆盖 `outputs/latest_holdings.csv`
- 如果门槛全部通过，输出正式 `signal_*.csv`、`manual_orders_*.csv` 和 `latest_holdings.csv`

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

这两个文件已加入 `.gitignore`，不会提交到 GitHub。没有真实持仓文件时，系统仍会生成候选交易单，但会在备注中提示 `current_shares_missing`。

## 输出文件

常见输出：

```text
data/raw/*.csv                         原始日线数据
data/qlib_data/                        Qlib provider 数据
data/prices/close.parquet              收盘价面板
data/prices/ohlcv.parquet              OHLCV 价格面板
data/factors/alpha158.parquet          Alpha158 因子缓存
data/factors/alpha158.parquet.meta.json 因子缓存元数据
data/factors/rolling_ic_weights.pkl    rolling IC 权重缓存
outputs/backtest_equity.csv            回测净值
outputs/backtest_holdings.csv          回测持仓
outputs/backtest_trades.csv            回测成交
outputs/backtest_metrics.json          回测指标
outputs/optimization_results.csv       参数优化结果
outputs/auto_run_status.json           自动流程阶段状态
outputs/data_health_report.json        数据健康检查
outputs/data_health_report.csv         数据健康检查表
outputs/auto_validation_windows.csv    自动选参逐窗口验证
outputs/auto_parameter_summary.csv     自动选参汇总
outputs/auto_selected_params.json      自动选中的策略参数
outputs/auto_parameter_quality.json    自动选参质量门槛判断
outputs/auto_backtest_metrics.json     自动选参后的回测指标
outputs/auto_signal_report.json        自动信号报告
outputs/daily_signal_report.md         每日信号 Markdown 报告
outputs/manual_orders_YYYY-MM-DD.csv   人工执行交易单
outputs/manual_orders_candidate_YYYY-MM-DD.csv 门槛未通过时的候选交易单
outputs/candidate_signal_YYYY-MM-DD.csv 门槛未通过时的候选信号
outputs/signal_YYYY-MM-DD.csv          每日信号
outputs/latest_holdings.csv            最新持仓
outputs/data_update_progress.json      数据补齐进度
data/raw/failed_fetches.csv             本轮补数据失败的股票及原因
outputs/history/YYYY-MM-DD/            每次自动运行的归档快照
```

## 策略与回测要点

当前默认配置位于 `config/settings.yaml`：

- 股票池：A 股主板 `mainboard_a`
- 因子组：`ic_weighted`
- 调仓频率：weekly
- 默认持仓数：5
- 单次最大换手：1
- 排名缓冲：10
- 成本：佣金、最低佣金、印花税、过户费、滑点
- 容量限制：使用交易日前历史 ADV，避免看当日成交额
- 涨跌停：优先用 high/low 判断触板
- 止盈止损：优先用 high/low 触发，并按止损价/止盈价或跳空开盘价成交
- 长期缺价/停牌：按配置折价退出，避免回测净值长期使用陈旧价格

## 测试

双击：

```text
03_运行测试.bat
```

或命令行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

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
- 如果数据补齐窗口长时间没有新增文件，先双击 `05_查看补齐进度.bat` 看 `current_symbol`、`last_error` 和 raw CSV 数量。
- 如果 `auto_signal_report.json` 里的 `is_executable` 是 `false`，不要按候选信号交易，先看 `block_reasons`。
- 大批量补齐数据是小时级任务，建议保持小批次可恢复模式运行。
