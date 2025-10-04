"""Margin and buying power helpers.

This module provides conservative estimates of required margin for positions. For
production, replace with broker-provided APIs for precise calculations.
"""
from typing import Dict


def required_margin_equity(price: float, qty: int, margin_factor: float = 1.0) -> float:
    """For equity cash/MIS assume full notional or smaller margin_factor for leverage."""
    return abs(price * qty) * margin_factor


def required_margin_fno(notional: float, margin_rate: float = 0.2) -> float:
    """Estimate margin for F&O as a fraction of notional.

    margin_rate: fraction (e.g., 0.2 means 20% of notional required)
    """
    return abs(notional) * margin_rate


def required_margin_fno_with_lot(price: float, lot_size: int, margin_rate: float = 0.2) -> float:
    """Compute margin required for one contract of F&O given lot size and margin_rate.

    Returns margin for one lot: price * lot_size * margin_rate
    """
    return abs(price * lot_size) * margin_rate


def has_buying_power(cash: float, required_margin: float) -> bool:
    return cash >= required_margin
