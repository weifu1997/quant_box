# quant_box 验证后开发计划

审查日期：2026-06-11

来源文件：`C:\Users\13403\.claude\plans\temporal-rolling-thimble.md`

## 开发进展

- 2026-06-11：完成 P0 运行基线文档，新增 `CLAUDE.md`，明确项目测试应使用 `.\.venv\Scripts\python.exe`。
- 2026-06-11：完成 P1 参数优化超时保护第一阶段，`run_walk_forward_grid_validation()` 支持 `timeout_seconds`、`deadline`、`max_grid_combinations`，自动流程支持 `--optimize-timeout-seconds` 和 `--max-optimize-combinations`，超时时写出 partial validation/summary 并记录状态。
- 2026-06-11：完成 P1 数据校验增强第一阶段，`normalize_daily_frame()` 会拒绝最终规范化结果中的无效 OHLCV（价格缺失/非正、成交量或成交额缺失/为负、OHLC 区间不自洽）。
- 2026-06-11：推进 P1 自动流程主函数拆分，已抽出 `_run_optimization_stage()` 和 `OptimizationStageResult`，优化参数构造、进度回调、超时落盘和参数选择从 `main()` 移出。
- 2026-06-11：完成 P1 性能基准与因子规范化缓存第一阶段，新增 `scripts/benchmark_scoring.py`，并为 `_normalize_factor_frame_for_scoring()` 增加弱引用缓存和规范化 attrs 标记，避免 `composite_factor()` 重复排序/去重同一因子框架。
- 2026-06-11：推进 P2 异常日志质量，非阻塞基本面报告失败会记录完整堆栈，候选信号日期解析改为捕获预期异常类型。
- 2026-06-11：推进 P2 公共规范化工具收敛，`src.common` 新增 `normalize_instruments()`、`normalize_instrument_index()`、`normalize_datetime_index()`、`normalize_multiindex_date_instrument()`，并迁移 `strategy.py`、`signal_generator.py`、`selection_risk.py` 的重复逻辑。
- 2026-06-11：推进 P2 缓存策略统一，`backtest.py` 与 `selection_risk.py` 的价格字段切片缓存增加大小上限和弱引用回调测试。
- 2026-06-11：完成 P2 风控编排抽象第一阶段，新增 `RiskPolicy` 统一读取行业约束、选股风险过滤、止损止盈、滑点与容量配置，并迁移回测、信号、自动优化、研究脚本和人工订单止损读取入口。
- 2026-06-11：继续推进 P1 自动流程主函数拆分，新增 `_run_data_preparation_stage()` 和 `DataPreparationStageResult`，将更新、转换、因子加载、数据健康、复权元数据、数据治理检查从 `main()` 移出。
- 2026-06-11：完成 P3 工程化补齐第一阶段，新增 `requirements-lock.txt`、GitHub Actions CI、`CONTRIBUTING.md`、`CHANGELOG.md`，并在 README/CLAUDE 文档中补充锁定依赖和 CI 测试入口。
- 2026-06-11：继续推进 P1 自动流程主函数拆分，新增 `_run_backtest_stage()` 和 `BacktestStageResult`，将历史回测、回测质量评估、研究诊断从 `main()` 移出，并补充阶段级测试。
- 2026-06-11：推进 P3 监控指标第一阶段，新增 `src.monitoring` 和 `scripts/export_auto_status_metrics.py`，可把 `auto_run_status.json` 导出为 Prometheus textfile 指标。
- 2026-06-11：继续推进 P1 自动流程主函数拆分，新增 `_run_signal_stage()`、`_write_auto_report_stage()`、`SignalStageResult` 和 `ReportStageResult`，将信号生成、质量门槛、失败分析、人工订单、报告和归档从 `main()` 移出。
- 2026-06-12：完成配置 schema 校验第一阶段，`load_config()` 在合并默认/本地配置和展开环境变量后校验已知配置段、关键类型与范围；未知键仍只告警并保留，兼容现有扩展行为。
- 2026-06-12：完成 Magic Numbers 收敛第一阶段，`strategy.py` 将行均值评分的“至少半数因子有效”阈值抽为 `ROW_MEAN_REQUIRED_FACTOR_FRACTION` 和 `_required_row_mean_factor_count()`，并补充评分行为测试。
- 2026-06-12：按用户要求移除容器构建相关产物、静态测试，以及 README/CLAUDE/CHANGELOG 中的容器构建说明。

## 验证方式

- 静态核对源码、配置、测试、文档和仓库结构。
- 使用项目虚拟环境运行相关测试：
  `.\.venv\Scripts\python.exe -m pytest tests/test_data_fetcher.py tests/test_strategy.py tests/test_backtest.py tests/test_signal_generator.py tests/test_selection_risk.py tests/test_selection_constraints.py -q`
- 初始结果：137 passed。
- 本轮补充验证：`tests/test_run_auto_signal.py tests/test_monitoring.py` `20 passed`；CI 同款宽回归 `215 passed`；研究脚本迁移测试 `37 passed`；`requirements-lock.txt` 安装 dry-run 无冲突；2026-06-12 追加 `tests/test_config_loader.py tests/test_strategy.py` `38 passed`、宽回归 `231 passed`、研究脚本测试 `37 passed`、全量 `pytest -q` `543 passed`；`git diff --check` 仅 CRLF 提示。
- 备注：系统默认 `python` 缺少 `pandas/numpy`，不能用于本项目测试；应使用项目 `.venv`。

## 验证结论

| 报告问题 | 结论 | 证据与修正说明 | 计划处理 |
|---|---|---|---|
| `normalize_daily_frame()` 缺少列存在性验证 | 不再成立 | `src/data_fetcher_frames.py` 已检查 `DAILY_FIELDS` 缺失列，空数据返回标准列，非空缺列抛 `ValueError`。 | 改为增强数据范围和一致性校验。 |
| 异常处理过宽 | 部分成立 | `scripts/run_auto_signal.py` 主流程捕获后会记录状态并重新抛出，合理；`_maybe_build_fundamental_screen()` 捕获所有异常且只记录 warning，缺少堆栈。 | 收窄或保留非阻塞捕获但记录完整堆栈。 |
| `run_auto_signal.py` God Function | 成立但报告夸大 | 文件 924 行，`main()` 约 545 行，不是 900+ 行；但仍集中参数解析、数据更新、治理、因子、优化、回测、信号、报告和归档。 | 拆成阶段函数和上下文对象。 |
| `_normalize_factor_frame_for_scoring()` 性能瓶颈 | 潜在成立 | 每次 `composite_factor()` 都经 `_cross_sectional_zscore()` 调用规范化；优化器会多窗口、多组合重复构建分数。尚无基准测试。 | 先加基准，再做缓存或上游一次性规范化。 |
| 日期和 instrument 规范化重复 | 部分成立 | `src.common` 已有 `normalize_instrument()` 和 `parse_datetime_values()`；但 `strategy.py`、`signal_generator.py`、`manual_orders.py` 等仍有重复列表规范化和日期解析。 | 扩展公共工具并渐进迁移。 |
| 参数优化缺少超时保护 | 成立 | `run_auto_signal.py` 直接调用 `run_walk_forward_grid_validation()`；`src/optimizer.py` 窗口和组合循环无 deadline/timeout 检查。 | 加 deadline、最大组合保护和部分结果落盘。 |
| 辅助函数过多 | 成立 | 自动流程、信号、策略模块含大量内部函数，其中不少是核心业务阶段。 | 随主流程重构迁移到阶段模块。 |
| 缓存使用不当 | 部分成立 | `backtest.py` 的规范化价格缓存已有最大大小和弱引用清理；字段切片缓存仍无大小上限。`selection_risk.py` 有类似缓存实现。 | 补统一缓存策略和测试。 |
| Magic Numbers | 已完成第一阶段 | `_row_mean_with_min_count()` 的半数阈值已抽为 `ROW_MEAN_REQUIRED_FACTOR_FRACTION` 和 `_required_row_mean_factor_count()`；`LOT_SIZE` 已有默认常量且可被配置覆盖。 | 后续仅在发现新的隐式业务阈值时继续收敛。 |
| 测试覆盖 517+ | 成立 | 当前全量 `pytest -q` 为 543 passed。 | 保持，新增性能和配置校验测试。 |
| 缺少性能/压力/配置格式测试 | 部分已处理 | 已新增 `scripts/benchmark_scoring.py`；配置 loader 已补 schema/type/range 校验，并覆盖默认配置、当前 settings、未知键保留、段类型错误和范围错误。 | 后续可继续补压力测试和更完整的配置交叉字段校验。 |
| CI/CD、锁定文件缺失 | 已完成第一阶段 | 已新增 `.github/workflows/ci.yml` 和 `requirements-lock.txt`；容器构建相关产物已按用户要求移除。 | 保持 CI 和锁定依赖入口。 |
| 风控模块不足 | 结论需修正 | 行业权重、选股风险过滤、动态 ADV 滑点、容量限制、止损止盈已有实现和测试；独立 `RiskManager` 仍不存在。 | 不重做风控能力，先做风控编排抽象。 |
| README 行数和文档缺失 | 部分成立 | README 当前约 358 行，不是报告 472 行；`CLAUDE.md`、`CONTRIBUTING.md`、`CHANGELOG.md` 缺失。 | 补 AI 上下文和贡献/变更文档。 |

## 开发计划

### P0：运行与验证基线

目标：确保后续改动可验证、可回退。

- 固化推荐测试入口：统一使用 `.\.venv\Scripts\python.exe -m pytest`。
- 在 README 或新建 `CLAUDE.md` 中记录本地测试环境和常用命令。
- 建立变更前基线：保留当前通过的 137 个相关测试，并在每个阶段至少跑对应测试。

验收：
- 文档明确说明不要使用缺依赖的系统默认 `python`。
- 相关测试命令可复现通过。

### P1：参数优化超时与部分结果保护

目标：避免自动流程在优化阶段长时间卡住。

- 为 `src/optimizer.py` 的 walk-forward 验证增加可选 `deadline` 或 `timeout_seconds`。
- 在窗口循环和参数组合循环边界检查 deadline。
- 超时后抛出明确的 `TimeoutError`，包含已完成窗口/组合数。
- `scripts/run_auto_signal.py` 增加 CLI/config 超时参数。
- 超时时写出已有 validation 结果，并把 `auto_run_status.json` 标记为可诊断状态。

验收：
- 新增单元测试覆盖 deadline 命中、未命中、已有结果回调。
- 自动流程不会无状态卡在 `optimize_params`。

### P1：数据校验增强

目标：把“列存在性”升级为“数据质量边界”。

- 在日线规范化或数据健康层增加 OHLC、成交量、成交额的基础校验。
- 明确策略：结构错误抛异常，行情异常进入健康报告或被剔除。
- 新增缺列、负成交量、`high < low`、关键价格缺失等测试。

验收：
- `normalize_daily_frame()` 的当前兼容行为不被破坏。
- 异常数据不会静默进入后续因子和回测。

### P1：拆分自动流程主函数

目标：降低 `run_auto_signal.py` 修改风险。

- 引入轻量 `AutoSignalContext` 或阶段输入对象，集中保存 config、out_dir、status、artifacts、target date。
- 拆出阶段函数：更新数据、转换、因子、数据健康、数据治理、参数优化、回测、信号、报告归档。
- 保持 CLI 行为和输出文件名不变。

验收：
- `tests/test_run_auto_signal.py` 全部通过。
- `main()` 只保留参数解析、上下文初始化和阶段编排。

### P1：性能基准与因子规范化缓存

目标：先量化，再优化热路径。

- 增加小型性能测试或脚本，度量 `build_strategy_scores()` 和 walk-forward validation 的基线耗时。
- 为 `_normalize_factor_frame_for_scoring()` 设计弱引用缓存，或在 factor 加载后一次性规范化并复用。
- 确保缓存对重复 date/instrument、乱序索引、空数据保持现有语义。

验收：
- 有基准结果可对比。
- 相关策略和评分测试通过。
- 重复调用同一因子框架时避免重复完整排序/去重。

### P2：公共规范化工具收敛

目标：减少日期和 instrument 处理不一致。

- 在 `src.common` 增加 `normalize_instruments()`、`normalize_datetime_index()`、`normalize_multiindex_date_instrument()` 等小工具。
- 先迁移 `strategy.py`、`signal_generator.py`、`selection_risk.py` 中重复逻辑。
- 对迁移函数补兼容测试。

验收：
- 重复内部函数减少。
- 股票代码大小写、空值、重复项处理在策略和信号模块一致。

### P2：异常处理和日志质量

目标：保留非阻塞能力，同时让故障可诊断。

- 对 `_maybe_build_fundamental_screen()` 使用 `logger.exception()` 或 `exc_info=True`。
- 日期解析类异常改为捕获 `TypeError`、`ValueError` 等预期异常。
- 自动报告中保留错误类型和阶段。

验收：
- 非核心报告失败不阻断主信号。
- 日志里可看到完整堆栈。

### P2：缓存策略统一

目标：避免缓存无界增长和实现漂移。

- 为字段切片缓存增加大小上限和弱引用回调。
- 统一 `backtest.py` 与 `selection_risk.py` 的价格缓存实现，必要时抽到公共模块。
- 增加缓存淘汰和失效测试。

验收：
- 缓存不会随大量临时 DataFrame 无界增长。
- 已有回测和选股风险测试通过。

### P2：风控编排抽象

目标：不重写现有风控能力，只收拢调用边界。

- 新建 `src/risk_manager.py` 或 `src/risk_policy.py`，封装配置读取和风控步骤编排。
- 首批只包装已有能力：行业权重、选股风险过滤、容量/滑点参数、止损止盈策略。
- 回测、信号、人工交易单逐步改为调用同一风控策略对象。

验收：
- 行业权重、选股风险过滤、止损止盈、动态滑点相关测试继续通过。
- 风控配置入口更集中。

### P3：工程化补齐

目标：补生产协作基础设施。

- 新增 `requirements-lock.txt` 或明确使用 `uv`/pip compile 的锁定流程。
- 新增 GitHub Actions：安装依赖、运行测试、缓存依赖。
- 新增 `CLAUDE.md`、`CONTRIBUTING.md`、`CHANGELOG.md`。
- 监控指标放到后续独立阶段，避免一次性扩大范围。

验收：
- CI 能跑最小测试集。
- 新贡献者或 AI 助手能快速理解运行方式和项目边界。

## 建议实施顺序

1. P0 运行基线。
2. P1 超时保护。
3. P1 数据校验增强。
4. P1 自动流程拆分。
5. P1 性能基准和缓存。
6. P2 公共工具、异常日志、缓存统一。
7. P2 风控编排抽象。
8. P3 CI、锁定文件和文档。
