"""模块说明：计算回测交易成本和可买股数。"""

from __future__ import annotations


def _commission_cost(gross: float, commission: float, min_commission: float) -> float:
    """函数说明：处理 commission_cost 的内部辅助逻辑。"""
    if gross <= 0 or commission <= 0:
        return 0.0
    cost = gross * commission
    return float(max(cost, min_commission)) if min_commission > 0 else float(cost)


def _transfer_fee_cost(gross: float, transfer_fee: float) -> float:
    """函数说明：处理 transfer_fee_cost 的内部辅助逻辑。"""
    if gross <= 0 or transfer_fee <= 0:
        return 0.0
    return float(gross * transfer_fee)


def _shares_affordable(capital: float, price: float, commission: float, min_commission: float, transfer_fee: float) -> float:
    """函数说明：处理 shares_affordable 的内部辅助逻辑。"""
    if capital <= 0 or price <= 0:
        return 0.0
    variable_rate = max(commission, 0.0) + max(transfer_fee, 0.0)
    variable_shares = capital / (price * (1 + variable_rate))
    if min_commission <= 0:
        return variable_shares
    fixed_shares = (capital - min_commission) / (price * (1 + max(transfer_fee, 0.0)))
    return max(0.0, min(variable_shares, fixed_shares))
