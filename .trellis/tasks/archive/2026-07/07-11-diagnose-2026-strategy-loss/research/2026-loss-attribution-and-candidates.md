# 2026 Loss Attribution and Controlled Candidate Results

## Baseline

* Rebuilt through 2026-07-10 with engine contract v2: canonical month-end signal dates, latest score on or before the signal date, and route-specific selection schedules.
* Full-history annual return: 24.73%.
* Full-history max drawdown: -17.69%.
* 2026 annualized return: about -6.90% in the canonical grid result (the auto-signal report rounded from its equivalent rebuilt path).
* Formal return/drawdown targets remain 20% / -20%; cost target remains 20% annualized against initial capital.

## Reproducibility Finding

The previously configured evidence file was not reproducible under the current engine. It was generated before the canonical signal-calendar/latest-prior-score implementation landed. Older grid runs could union source-specific partial-month dates and did not carry an engine contract or factor-source provenance. The current auto-signal rebuild correctly rejected that evidence through the backtest gate.

Repairs made before candidate testing:

* Research backtest/grid now use configured per-source factor files.
* Standalone/grid routing uses canonical signal dates.
* Score caches include engine contract, source definition, and date-range fingerprints.
* Candidate evidence includes engine contract and source definitions.
* Auto-signal parameter evidence rejects missing/mismatched engine or source provenance.
* Yearly diagnostic cost ratios use the segment's starting equity rather than the global one-million initial capital.

## 2026 Attribution

### Annual route

All seven 2026 rebalance periods selected:

* source: `beta20`
* reason: `moderate_low_beta20`
* exposure: 40%

The route did not adapt within the year. It protected relative performance in bear/sideways periods but under-captured bull periods.

### Regimes

* Bull days: strategy about +2.31%, benchmark about +20.94%.
* Bear days: strategy about -4.29%, benchmark about -15.76%.
* Sideways days: strategy about -0.81%, benchmark about -7.02%.

The strategy was relatively defensive, but its absolute-return gate failed because bull-market participation was too low and selected beta20 holdings still lost money in several rebalance periods.

### Holdings and industries

Largest 2026 negative gross contributors included:

* `600599.SH`: about -6.17 percentage points of normalized holding contribution.
* `600421.SH`: about -4.08 points.
* `001322.SZ`: about -3.66 points.

Largest negative industry/group labels:

* `UNKNOWN`: about -10.25 points.
* 家居用品: about -4.38 points.
* 化工原料: about -3.69 points.

Directly replacing the route with industry momentum did not pass, so concentration was a symptom rather than a sufficient standalone repair.

### Costs and risk exits

* 2026 trades: 39 (17 buys, 22 sells).
* Total trade cost: about 72,461.
* Slippage: about 57,081, the dominant component.
* Risk exits: 3, all `take_profit`, cost about 2,806.
* Blocked trades: 5; partial trades: 1.

Risk exits were small and profitable exits, not the primary loss driver. The old yearly report's 14.85% annual cost ratio was a diagnostic denominator bug. Against 2026 starting equity of about 12.05 million, the comparable annualized cost ratio is about 1.23%.

## Controlled Candidate Grid

All candidates preserved every parameter except the diagnosed moderate-low source/exposure fields.

| Candidate | Full annual return | Max drawdown | Annual cost ratio | Failed years | Result |
| --- | ---: | ---: | ---: | --- | --- |
| beta20 / 40% baseline | 24.73% | -17.69% | 17.00% | 2026 | Fail |
| default route | 27.25% | -17.69% | 24.77% | 2025 | Fail |
| beta / 40% | 24.82% | -17.69% | 14.93% | 2025, 2026 | Fail |
| roc60 / 40% | 23.78% | -17.69% | 14.65% | 2025, 2026 | Fail |
| industry / 40% | 24.29% | -17.69% | 14.55% | 2025, 2026 | Fail |
| selector / 40% | 24.12% | -17.69% | 15.37% | 2025, 2026 | Fail |
| beta20 / 20% | 23.78% | -17.69% | 14.49% | 2025, 2026 | Fail |
| beta20 / 60% | 25.20% | -17.69% | 22.77% | 2026 | Fail |

No candidate passed every return, drawdown, yearly, and cost gate.

## Walk-Forward Result

Four representative candidates were evaluated by selecting from prior years and testing the next year (2018–2026):

* Return gate passes: 7/9 OOS years.
* Drawdown gate passes: 9/9 OOS years.
* OOS mean yearly return: 24.53%.
* OOS minimum yearly return: -2.36%.
* Failed OOS years: 2025 and 2026.
* 2026 selection based on 2015–2025 chose beta20 / 60%, but its 2026 OOS annualized return was still about -2.36%.

Candidate-family caveat: the small family was chosen after observing the 2026 failure. The rolling selection is point-in-time, but this is not a pristine untouched-family OOS study.

## Promotion Decision

Promotion is forbidden. No formal evidence/config/holding file should be changed. The next automatic run must remain candidate-only/blocked until a separately justified strategy family passes current-engine full-history, yearly, drawdown, cost, and OOS gates.
