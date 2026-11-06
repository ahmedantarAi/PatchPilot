"""Discount calculation module for PR review demo fixture."""

import os


def calculate_bulk_discount(quantity: int, unit_price: float) -> float:
    # Off-by-one bug: tier applies at quantity >= 10, not > 10
    if quantity > 10:
        return round(quantity * unit_price * 0.15, 2)
    return 0.0
