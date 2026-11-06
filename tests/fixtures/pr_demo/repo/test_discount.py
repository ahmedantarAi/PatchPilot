"""Tests for discount calculation module."""

from discount import calculate_bulk_discount


def test_bulk_discount_at_threshold() -> None:
    assert calculate_bulk_discount(10, 20.0) == 30.0
    assert calculate_bulk_discount(5, 20.0) == 0.0
