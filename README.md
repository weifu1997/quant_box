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
| `02_补齐股票数据_持续.bat` | 持续补齐缺失主板股票日线数据 |
| `03_查看补齐进度.bat` | 查看本地 raw CSV 数量和补齐进度 JSON |
| `04_转换数据.bat` | 将 `data/raw/*.csv` 转为 Qlib 数据和价格面板 |
| `05_计算因子.bat` | 计算或读取 Alpha158 因子缓存 |
| `06_运行回测.bat` | 运行当前配置下的回测 |
| `07_生成最新信号.bat` | 基于最新因子生成手动交易信号 |
| `08_参数优化.bat` | 运行 walk-forward 参数优化 |
| `09_运行测试.bat` | 运行自动化测试 |
| `10_全流程_补数据到信号.bat` | 从补数据到生成信号的一键全流程 |

最常用的是：

```text
02_补齐股票数据_持续.bat
03_查看补齐进度.bat
04_转换数据.bat
05_计算因子.bat
06_运行回测.bat
07_生成最新信号.bat
```

## 补齐股票数据

双击：

```text
02_补齐股票数据_持续.bat
```

默认参数：

```powershell
--chunk-size 50 --sleep-seconds 60
```

含义：

- 每批补 50 只缺失股票
- 批间等待 60 秒
- 自动记录进度到 `outputs/data_update_progress.json`
- 中断后再次双击，会继续补缺失股票，不会从头开始

命令行等价写法：

```powershell
.\.venv\Scripts\python.exe scripts\run_update_data.py --chunk-size 50 --sleep-seconds 60
```

只跑一批确认状态：

```powershell
.\.venv\Scripts\python.exe scripts\run_update_data.py --chunk-size 50 --sleep-seconds 60 --max-chunks 1
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
04_转换数据.bat
05_计算因子.bat
06_运行回测.bat
07_生成最新信号.bat
```

对应命令行：

```powershell
.\.venv\Scripts\python.exe scripts\run_convert_data.py
.\.venv\Scripts\python.exe scripts\run_calc_factors.py
.\.venv\Scripts\python.exe scripts\run_backtest.py
.\.venv\Scripts\python.exe scripts\run_daily_signal.py --date latest
```

完整一键流程：

```text
10_全流程_补数据到信号.bat
```

注意：全流程会先补数据。如果当前缺失股票很多，这一步会耗时较久。

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
outputs/signal_YYYY-MM-DD.csv          每日信号
outputs/latest_holdings.csv            最新持仓
outputs/data_update_progress.json      数据补齐进度
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
09_运行测试.bat
```

或命令行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 换电脑迁移数据

GitHub 会保存代码、默认配置、脚本和测试，但不会保存：

- `.venv`
- `config/settings.local.yaml`
- `data/`
- `outputs/`

如果想快速恢复旧电脑状态，可以手动复制旧电脑的：

```text
config/settings.local.yaml
data/
outputs/
```

如果不复制数据，就在新电脑上重新双击：

```text
00_安装依赖环境.bat
02_补齐股票数据_持续.bat
```

## 注意事项

- 本项目只生成手动交易信号，不负责自动下单。
- 不要提交 `config/settings.local.yaml`。
- 如果数据补齐窗口长时间没有新增文件，先双击 `03_查看补齐进度.bat` 看 `current_symbol`、`last_error` 和 raw CSV 数量。
- 大批量补齐数据是小时级任务，建议保持小批次可恢复模式运行。
