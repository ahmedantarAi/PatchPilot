"""Bug taxonomy definitions and template-based case generators for Phase 3 evaluation datasets."""

from __future__ import annotations

import difflib
import hashlib
import random
from collections.abc import Callable
from dataclasses import dataclass

from patchpilot.evals.models import BugTaxonomy, CaseDefinition, CaseMetadata

# 10 distinct base templates spanning all 7 SPEC §7 taxonomy categories
BASE_TEMPLATES = [
    "inventory_mgmt",
    "order_pricing",
    "content_feed",
    "analytics_tracker",
    "user_auth",
    "payment_gateway",
    "cache_store",
    "data_pipeline",
    "notification_queue",
    "audit_logger",
]


@dataclass(frozen=True)
class GeneratorResult:
    module_name: str
    buggy_code: str
    fixed_code: str
    test_code: str
    description: str


# Generator function signature: (seed: int) -> GeneratorResult
GeneratorFunc = Callable[[int], GeneratorResult]


# ---------------------------------------------------------------------------
# 1. OFF_BY_ONE
# ---------------------------------------------------------------------------


def gen_off_by_one_inventory(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    ps = rng.choice([4, 5, 8, 10])
    salt = rng.randint(100, 999)
    mod = f"inv_page_{salt}"
    buggy = f"""\"\"\"Inventory pagination module {salt}.\"\"\"


def paginate_inventory(items: list[str], page: int, page_size: int = {ps}) -> list[str]:
    # 1-indexed pagination bug: offsets by whole page too far
    offset = page * page_size
    return items[offset : offset + page_size]
"""
    fixed = f"""\"\"\"Inventory pagination module {salt}.\"\"\"


def paginate_inventory(items: list[str], page: int, page_size: int = {ps}) -> list[str]:
    offset = (page - 1) * page_size
    return items[offset : offset + page_size]
"""
    test = f"""\"\"\"Test for inventory pagination module {salt}.\"\"\"
from {mod} import paginate_inventory


def test_pagination_first_page():
    items = [f"item_{{i}}" for i in range(50)]
    page_1 = paginate_inventory(items, page=1, page_size={ps})
    assert len(page_1) == {ps}
    assert page_1[0] == "item_0"
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Fix off-by-one error in 1-indexed inventory pagination offset",
    )


def gen_off_by_one_pricing(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    thresh = rng.choice([10, 15, 20])
    pct = rng.choice([10, 15, 25])
    salt = rng.randint(100, 999)
    mod = f"pricing_tier_{salt}"
    buggy = f"""\"\"\"Pricing tier calculation module {salt}.\"\"\"


def calculate_tier_discount(quantity: int, unit_price: float) -> float:
    # Off-by-one in tier threshold: > instead of >=
    if quantity > {thresh}:
        return round(quantity * unit_price * {pct / 100:.2f}, 2)
    return 0.0
"""
    fixed = f"""\"\"\"Pricing tier calculation module {salt}.\"\"\"


def calculate_tier_discount(quantity: int, unit_price: float) -> float:
    if quantity >= {thresh}:
        return round(quantity * unit_price * {pct / 100:.2f}, 2)
    return 0.0
"""
    expected = round(thresh * 10.0 * (pct / 100), 2)
    test = f"""\"\"\"Test for pricing tier calculation module {salt}.\"\"\"
from {mod} import calculate_tier_discount


def test_tier_threshold_boundary():
    discount = calculate_tier_discount({thresh}, 10.0)
    assert discount == {expected}
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Fix off-by-one threshold boundary in pricing tier discount calculation",
    )


def gen_off_by_one_content(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    limit = rng.choice([5, 6, 7, 8])
    salt = rng.randint(100, 999)
    mod = f"feed_slice_{salt}"
    buggy = f"""\"\"\"Feed slice module {salt}.\"\"\"


def slice_top_posts(posts: list[dict], limit: int = {limit}) -> list[dict]:
    # Slice takes one post too many
    return posts[: limit + 1]
"""
    fixed = f"""\"\"\"Feed slice module {salt}.\"\"\"


def slice_top_posts(posts: list[dict], limit: int = {limit}) -> list[dict]:
    return posts[:limit]
"""
    test = f"""\"\"\"Test for feed slice module {salt}.\"\"\"
from {mod} import slice_top_posts


def test_slice_limit_exact():
    posts = [{{"id": i, "title": f"Post {{i}}"}} for i in range(30)]
    sliced = slice_top_posts(posts, limit={limit})
    assert len(sliced) == {limit}
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Fix slice off-by-one returning one post beyond the requested limit",
    )


def gen_off_by_one_analytics(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    w = rng.choice([3, 4, 5])
    salt = rng.randint(100, 999)
    mod = f"analytics_window_{salt}"
    buggy = f"""\"\"\"Analytics window module {salt}.\"\"\"


def compute_sliding_window_count(data_length: int, window_size: int = {w}) -> int:
    if data_length < window_size:
        return 0
    # Fencepost error: misses the final valid window
    return data_length - window_size
"""
    fixed = f"""\"\"\"Analytics window module {salt}.\"\"\"


def compute_sliding_window_count(data_length: int, window_size: int = {w}) -> int:
    if data_length < window_size:
        return 0
    return data_length - window_size + 1
"""
    test = f"""\"\"\"Test for analytics window module {salt}.\"\"\"
from {mod} import compute_sliding_window_count


def test_window_count():
    data = [10, 20, 30, 40, 50, 60]
    expected_windows = len(data) - {w} + 1
    assert compute_sliding_window_count(len(data), {w}) == expected_windows
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Fix fencepost off-by-one in sliding window count calculation",
    )


# ---------------------------------------------------------------------------
# 2. WRONG_OPERATOR
# ---------------------------------------------------------------------------


def gen_wrong_operator_inventory(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    thresh = rng.choice([10, 15, 25])
    salt = rng.randint(100, 999)
    mod = f"stock_check_{salt}"
    buggy = f"""\"\"\"Stock check module {salt}.\"\"\"


def needs_reorder(current_stock: int, threshold: int = {thresh}) -> bool:
    # Wrong operator: checks > instead of <=
    return current_stock > threshold
"""
    fixed = f"""\"\"\"Stock check module {salt}.\"\"\"


def needs_reorder(current_stock: int, threshold: int = {thresh}) -> bool:
    return current_stock <= threshold
"""
    test = f"""\"\"\"Test for stock check module {salt}.\"\"\"
from {mod} import needs_reorder


def test_reorder_low_stock():
    assert needs_reorder(current_stock=2, threshold={thresh}) is True
    assert needs_reorder(current_stock={thresh} + 10, threshold={thresh}) is False
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Replace inverted comparison operator in inventory reorder predicate",
    )


def gen_wrong_operator_auth(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"perm_eval_{salt}"
    buggy = f"""\"\"\"Permission evaluation module {salt}.\"\"\"


def can_moderate_content(is_admin: bool, is_moderator: bool) -> bool:
    # Wrong operator: requires both admin AND moderator
    return is_admin and is_moderator
"""
    fixed = f"""\"\"\"Permission evaluation module {salt}.\"\"\"


def can_moderate_content(is_admin: bool, is_moderator: bool) -> bool:
    return is_admin or is_moderator
"""
    test = f"""\"\"\"Test for permission evaluation module {salt}.\"\"\"
from {mod} import can_moderate_content


def test_moderator_can_moderate():
    assert can_moderate_content(is_admin=False, is_moderator=True) is True
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Replace boolean 'and' with 'or' in moderation permission check",
    )


def gen_wrong_operator_pricing(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    discount = rng.choice([5.0, 10.0, 15.0])
    salt = rng.randint(100, 999)
    mod = f"voucher_calc_{salt}"
    buggy = f"""\"\"\"Voucher calculation module {salt}.\"\"\"


def apply_voucher(subtotal: float, voucher_value: float = {discount}) -> float:
    # Wrong operator: adds voucher value instead of subtracting
    return round(subtotal + voucher_value, 2)
"""
    fixed = f"""\"\"\"Voucher calculation module {salt}.\"\"\"


def apply_voucher(subtotal: float, voucher_value: float = {discount}) -> float:
    return round(max(0.0, subtotal - voucher_value), 2)
"""
    test = f"""\"\"\"Test for voucher calculation module {salt}.\"\"\"
from {mod} import apply_voucher


def test_voucher_reduces_total():
    subtotal = 100.0
    result = apply_voucher(subtotal, {discount})
    assert result == 100.0 - {discount}
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Fix addition operator used instead of subtraction in voucher discount",
    )


def gen_wrong_operator_payment(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    limit = rng.choice([500.0, 1000.0, 2000.0])
    salt = rng.randint(100, 999)
    mod = f"charge_perm_{salt}"
    buggy = f"""\"\"\"Charge permission module {salt}.\"\"\"


def is_charge_valid(balance: float, amount: float, credit_limit: float = {limit}) -> bool:
    # Wrong operator: inverted condition
    return (balance + credit_limit) < amount
"""
    fixed = f"""\"\"\"Charge permission module {salt}.\"\"\"


def is_charge_valid(balance: float, amount: float, credit_limit: float = {limit}) -> bool:
    return (balance + credit_limit) >= amount
"""
    test = f"""\"\"\"Test for charge permission module {salt}.\"\"\"
from {mod} import is_charge_valid


def test_valid_charge():
    assert is_charge_valid(balance=100.0, amount=200.0, credit_limit={limit}) is True
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Invert comparison operator to correctly authorize charges within limits",
    )


# ---------------------------------------------------------------------------
# 3. NONE_HANDLING
# ---------------------------------------------------------------------------


def gen_none_handling_auth(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"user_fmt_{salt}"
    buggy = f"""\"\"\"User formatting module {salt}.\"\"\"


def format_user_display_name(profile: dict) -> str:
    # Missing None check on profile['full_name']
    return profile["full_name"].strip()
"""
    fixed = f"""\"\"\"User formatting module {salt}.\"\"\"


def format_user_display_name(profile: dict) -> str:
    name = profile.get("full_name")
    if name is not None:
        return name.strip()
    return profile.get("username", "Guest").strip()
"""
    test = f"""\"\"\"Test for user formatting module {salt}.\"\"\"
from {mod} import format_user_display_name


def test_format_display_name_none():
    profile = {{"username": "jdoe", "full_name": None}}
    assert format_user_display_name(profile) == "jdoe"
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Guard against NoneType in user profile full_name before string stripping",
    )


def gen_none_handling_cache(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"cache_ttl_{salt}"
    buggy = f"""\"\"\"Cache TTL module {salt}.\"\"\"


def get_entry_ttl(cache: dict, key: str) -> int:
    entry = cache.get(key)
    # Missing None check before subscripting entry
    return entry["ttl"]
"""
    fixed = f"""\"\"\"Cache TTL module {salt}.\"\"\"


def get_entry_ttl(cache: dict, key: str) -> int:
    entry = cache.get(key)
    if entry is not None:
        return int(entry.get("ttl", 0))
    return 0
"""
    test = f"""\"\"\"Test for cache TTL module {salt}.\"\"\"
from {mod} import get_entry_ttl


def test_missing_cache_key():
    cache = {{"active": {{"ttl": 300}}}}
    assert get_entry_ttl(cache, "non_existent_key") == 0
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Check for None before subscripting cache lookup entry dictionary",
    )


def gen_none_handling_pipeline(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"tag_norm_{salt}"
    buggy = f"""\"\"\"Tag normalization module {salt}.\"\"\"


def normalize_record_tag(record: dict) -> str:
    # Fails with AttributeError when tag value is None
    return record["tag"].lower()
"""
    fixed = f"""\"\"\"Tag normalization module {salt}.\"\"\"


def normalize_record_tag(record: dict) -> str:
    tag = record.get("tag")
    return tag.lower() if tag is not None else ""
"""
    test = f"""\"\"\"Test for tag normalization module {salt}.\"\"\"
from {mod} import normalize_record_tag


def test_none_tag_handling():
    record = {{"id": 42, "tag": None}}
    assert normalize_record_tag(record) == ""
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Handle NoneType gracefully in data pipeline record tag normalization",
    )


def gen_none_handling_content(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"comment_auth_{salt}"
    buggy = f"""\"\"\"Comment author module {salt}.\"\"\"


def extract_comment_author(comment: dict) -> str:
    # Fails with TypeError when comment['author'] is None
    return comment["author"]["username"]
"""
    fixed = f"""\"\"\"Comment author module {salt}.\"\"\"


def extract_comment_author(comment: dict) -> str:
    author = comment.get("author")
    if author is not None:
        return author.get("username", "anonymous")
    return "anonymous"
"""
    test = f"""\"\"\"Test for comment author module {salt}.\"\"\"
from {mod} import extract_comment_author


def test_none_author_in_comment():
    comment = {{"text": "Hello world", "author": None}}
    assert extract_comment_author(comment) == "anonymous"
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Safely handle NoneType in nested comment author dictionary extraction",
    )


# ---------------------------------------------------------------------------
# 4. MISSING_AWAIT
# ---------------------------------------------------------------------------


def gen_missing_await_payment(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"async_pay_{salt}"
    buggy = f"""\"\"\"Async payment module {salt}.\"\"\"


async def process_transaction(gateway, amount: float) -> str:
    # Missing await on coroutine
    response = gateway.authorize(amount)
    return response["status"]
"""
    fixed = f"""\"\"\"Async payment module {salt}.\"\"\"


async def process_transaction(gateway, amount: float) -> str:
    response = await gateway.authorize(amount)
    return response["status"]
"""
    test = f"""\"\"\"Test for async payment module {salt}.\"\"\"
import asyncio
from {mod} import process_transaction


class MockGateway:
    async def authorize(self, amount: float) -> dict:
        return {{"status": "authorized", "amount": amount}}


def test_process_transaction():
    gw = MockGateway()
    status = asyncio.run(process_transaction(gw, 150.0))
    assert status == "authorized"
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Add missing await keyword to asynchronous payment authorization call",
    )


def gen_missing_await_notification(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"async_alert_{salt}"
    buggy = f"""\"\"\"Async notification module {salt}.\"\"\"


async def broadcast_alert(notifier, message: str) -> int:
    # Missing await on async dispatch call
    recipients = notifier.dispatch(message)
    return len(recipients)
"""
    fixed = f"""\"\"\"Async notification module {salt}.\"\"\"


async def broadcast_alert(notifier, message: str) -> int:
    recipients = await notifier.dispatch(message)
    return len(recipients)
"""
    test = f"""\"\"\"Test for async notification module {salt}.\"\"\"
import asyncio
from {mod} import broadcast_alert


class MockNotifier:
    async def dispatch(self, message: str) -> list[str]:
        return ["user1@example.com", "user2@example.com"]


def test_broadcast_alert():
    notifier = MockNotifier()
    count = asyncio.run(broadcast_alert(notifier, "System reboot"))
    assert count == 2
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Await coroutine result before calculating notification recipient count",
    )


def gen_missing_await_pipeline(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"async_batch_{salt}"
    buggy = f"""\"\"\"Async data batching module {salt}.\"\"\"


async def ingest_batch_values(stream, limit: int) -> int:
    # Missing await on stream fetch
    items = stream.fetch_values(limit)
    return sum(items)
"""
    fixed = f"""\"\"\"Async data batching module {salt}.\"\"\"


async def ingest_batch_values(stream, limit: int) -> int:
    items = await stream.fetch_values(limit)
    return sum(items)
"""
    test = f"""\"\"\"Test for async data batching module {salt}.\"\"\"
import asyncio
from {mod} import ingest_batch_values


class MockStream:
    async def fetch_values(self, limit: int) -> list[int]:
        return [10, 20, 30]


def test_ingest_batch_values():
    stream = MockStream()
    total = asyncio.run(ingest_batch_values(stream, 3))
    assert total == 60
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Await async stream batch fetch coroutine before summing values",
    )


def gen_missing_await_audit(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"async_audit_{salt}"
    buggy = f"""\"\"\"Async audit logging module {salt}.\"\"\"


async def record_audit_entry(logger, action: str) -> int:
    # Missing await before arithmetic addition
    code = logger.write_entry(action)
    return code + 1
"""
    fixed = f"""\"\"\"Async audit logging module {salt}.\"\"\"


async def record_audit_entry(logger, action: str) -> int:
    code = await logger.write_entry(action)
    return code + 1
"""
    test = f"""\"\"\"Test for async audit logging module {salt}.\"\"\"
import asyncio
from {mod} import record_audit_entry


class MockLogger:
    async def write_entry(self, action: str) -> int:
        return 100


def test_record_audit_entry():
    logger = MockLogger()
    code = asyncio.run(record_audit_entry(logger, "LOGIN"))
    assert code == 101
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Await async audit logging call before returning incremented response code",
    )


# ---------------------------------------------------------------------------
# 5. RESOURCE_LEAK
# ---------------------------------------------------------------------------


def gen_resource_leak_inventory(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"inv_export_{salt}"
    buggy = f"""\"\"\"Inventory export module {salt}.\"\"\"


class ManagedChannel:
    def __init__(self) -> None:
        self.closed = False

    def write(self, data: str) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def export_inventory_data(channel: ManagedChannel, payload: str) -> bool:
    # Resource leak: channel is never closed
    channel.write(payload)
    return True
"""
    fixed = f"""\"\"\"Inventory export module {salt}.\"\"\"


class ManagedChannel:
    def __init__(self) -> None:
        self.closed = False

    def write(self, data: str) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def export_inventory_data(channel: ManagedChannel, payload: str) -> bool:
    with channel:
        channel.write(payload)
    return True
"""
    test = f"""\"\"\"Test for inventory export module {salt}.\"\"\"
from {mod} import ManagedChannel, export_inventory_data


def test_channel_closed_after_export():
    channel = ManagedChannel()
    success = export_inventory_data(channel, "SKU1,SKU2")
    assert success is True
    assert channel.closed is True
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Use context manager to prevent channel resource leak during inventory export",
    )


def gen_resource_leak_pipeline(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"stream_proc_{salt}"
    buggy = f"""\"\"\"Stream processor module {salt}.\"\"\"


class DataStream:
    def __init__(self, data: list[int]) -> None:
        self.data = data
        self.closed = False

    def read_all(self) -> list[int]:
        return list(self.data)

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def process_stream_batch(stream: DataStream) -> int:
    # Resource leak: stream left open
    values = stream.read_all()
    return sum(values)
"""
    fixed = f"""\"\"\"Stream processor module {salt}.\"\"\"


class DataStream:
    def __init__(self, data: list[int]) -> None:
        self.data = data
        self.closed = False

    def read_all(self) -> list[int]:
        return list(self.data)

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def process_stream_batch(stream: DataStream) -> int:
    with stream:
        values = stream.read_all()
    return sum(values)
"""
    test = f"""\"\"\"Test for stream processor module {salt}.\"\"\"
from {mod} import DataStream, process_stream_batch


def test_stream_closed_after_batch():
    stream = DataStream([1, 2, 3])
    total = process_stream_batch(stream)
    assert total == 6
    assert stream.closed is True
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Ensure data stream is properly closed to prevent file descriptor leaks",
    )


def gen_resource_leak_audit(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"audit_conn_{salt}"
    buggy = f"""\"\"\"Audit connection module {salt}.\"\"\"


class AuditConnection:
    def __init__(self) -> None:
        self.closed = False

    def commit(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def record_audit_log(conn: AuditConnection, event: str) -> None:
    # Resource leak: connection not closed
    conn.commit()
"""
    fixed = f"""\"\"\"Audit connection module {salt}.\"\"\"


class AuditConnection:
    def __init__(self) -> None:
        self.closed = False

    def commit(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def record_audit_log(conn: AuditConnection, event: str) -> None:
    with conn:
        conn.commit()
"""
    test = f"""\"\"\"Test for audit connection module {salt}.\"\"\"
from {mod} import AuditConnection, record_audit_log


def test_connection_closed():
    conn = AuditConnection()
    record_audit_log(conn, "AUTH_FAILURE")
    assert conn.closed is True
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Close audit database connection via context manager to avoid resource leak",
    )


def gen_resource_leak_notification(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"socket_trans_{salt}"
    buggy = f"""\"\"\"Notification socket module {salt}.\"\"\"


class SocketTransport:
    def __init__(self) -> None:
        self.closed = False

    def send(self, data: bytes) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def transmit_notification(transport: SocketTransport, payload: bytes) -> None:
    # Resource leak: socket left unclosed
    transport.send(payload)
"""
    fixed = f"""\"\"\"Notification socket module {salt}.\"\"\"


class SocketTransport:
    def __init__(self) -> None:
        self.closed = False

    def send(self, data: bytes) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def transmit_notification(transport: SocketTransport, payload: bytes) -> None:
    with transport:
        transport.send(payload)
"""
    test = f"""\"\"\"Test for notification socket module {salt}.\"\"\"
from {mod} import SocketTransport, transmit_notification


def test_socket_closed():
    transport = SocketTransport()
    transmit_notification(transport, b"ALERT")
    assert transport.closed is True
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Properly close notification socket transport to prevent socket leak",
    )


# ---------------------------------------------------------------------------
# 6. BAD_DEFAULT_ARG
# ---------------------------------------------------------------------------


def gen_bad_default_inventory(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"inv_item_{salt}"
    buggy = f"""\"\"\"Inventory item creation module {salt}.\"\"\"


def register_inventory_item(name: str, tags: list[str] = []) -> list[str]:
    # Mutable default list argument leaks across calls
    tags.append(name)
    return tags
"""
    fixed = f"""\"\"\"Inventory item creation module {salt}.\"\"\"


def register_inventory_item(name: str, tags: list[str] | None = None) -> list[str]:
    if tags is None:
        tags = []
    tags.append(name)
    return tags
"""
    test = f"""\"\"\"Test for inventory item creation module {salt}.\"\"\"
from {mod} import register_inventory_item


def test_no_shared_mutable_default():
    first = register_inventory_item("Alpha")
    second = register_inventory_item("Beta")
    assert second == ["Beta"]
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Replace mutable list default argument with None sentinel in item registration",
    )


def gen_bad_default_pricing(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"order_items_{salt}"
    buggy = f"""\"\"\"Order item accumulator module {salt}.\"\"\"


def add_order_line(sku: str, items: list[str] = []) -> list[str]:
    # Mutable default list argument
    items.append(sku)
    return items
"""
    fixed = f"""\"\"\"Order item accumulator module {salt}.\"\"\"


def add_order_line(sku: str, items: list[str] | None = None) -> list[str]:
    if items is None:
        items = []
    items.append(sku)
    return items
"""
    test = f"""\"\"\"Test for order item accumulator module {salt}.\"\"\"
from {mod} import add_order_line


def test_independent_order_lines():
    order_a = add_order_line("SKU-100")
    order_b = add_order_line("SKU-200")
    assert order_b == ["SKU-200"]
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Fix mutable default argument state leakage across separate orders",
    )


def gen_bad_default_cache(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"cache_entry_{salt}"
    buggy = f"""\"\"\"Cache entry initializer module {salt}.\"\"\"


def init_cache_metadata(key: str, meta: dict = {{}}) -> dict:
    # Mutable default dictionary leaks across keys
    meta[key] = True
    return meta
"""
    fixed = f"""\"\"\"Cache entry initializer module {salt}.\"\"\"


def init_cache_metadata(key: str, meta: dict | None = None) -> dict:
    if meta is None:
        meta = {{}}
    meta[key] = True
    return meta
"""
    test = f"""\"\"\"Test for cache entry initializer module {salt}.\"\"\"
from {mod} import init_cache_metadata


def test_isolated_metadata_dicts():
    first = init_cache_metadata("session_a")
    second = init_cache_metadata("session_b")
    assert "session_a" not in second
    assert second == {{"session_b": True}}
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Use None default instead of mutable dictionary in cache initialization",
    )


def gen_bad_default_auth(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"user_roles_{salt}"
    buggy = f"""\"\"\"User role assignment module {salt}.\"\"\"


def assign_base_roles(username: str, roles: list[str] = []) -> list[str]:
    # Mutable list default leaks assigned roles to future calls
    roles.append(f"user_{{username}}")
    return roles
"""
    fixed = f"""\"\"\"User role assignment module {salt}.\"\"\"


def assign_base_roles(username: str, roles: list[str] | None = None) -> list[str]:
    if roles is None:
        roles = []
    roles.append(f"user_{{username}}")
    return roles
"""
    test = f"""\"\"\"Test for user role assignment module {salt}.\"\"\"
from {mod} import assign_base_roles


def test_role_assignment_isolation():
    u1_roles = assign_base_roles("alice")
    u2_roles = assign_base_roles("bob")
    assert u2_roles == ["user_bob"]
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Prevent role pollution across users caused by mutable default list argument",
    )


# ---------------------------------------------------------------------------
# 7. SQL_INJECTION
# ---------------------------------------------------------------------------


def gen_sql_injection_audit(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"audit_query_{salt}"
    buggy = f"""\"\"\"Audit query builder module {salt}.\"\"\"


def build_audit_query(user_id: str) -> tuple[str, tuple]:
    # Vulnerable SQL construction via string formatting
    query = f"SELECT * FROM audit_logs WHERE user_id = '{{user_id}}'"
    return query, ()
"""
    fixed = f"""\"\"\"Audit query builder module {salt}.\"\"\"


def build_audit_query(user_id: str) -> tuple[str, tuple]:
    query = "SELECT * FROM audit_logs WHERE user_id = ?"
    return query, (user_id,)
"""
    test = f"""\"\"\"Test for audit query builder module {salt}.\"\"\"
from {mod} import build_audit_query


def test_parameterized_audit_query():
    raw_input = "admin' OR '1'='1"
    query, params = build_audit_query(raw_input)
    assert "?" in query
    assert raw_input not in query
    assert params == (raw_input,)
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Replace formatted SQL string with parameterized query and placeholder tuple",
    )


def gen_sql_injection_auth(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"login_query_{salt}"
    buggy = f"""\"\"\"User login query module {salt}.\"\"\"


def build_login_query(username: str) -> tuple[str, tuple]:
    # String interpolation leads to SQL injection vulnerability
    query = f"SELECT id, pass_hash FROM accounts WHERE username = '{{username}}'"
    return query, ()
"""
    fixed = f"""\"\"\"User login query module {salt}.\"\"\"


def build_login_query(username: str) -> tuple[str, tuple]:
    query = "SELECT id, pass_hash FROM accounts WHERE username = ?"
    return query, (username,)
"""
    test = f"""\"\"\"Test for user login query module {salt}.\"\"\"
from {mod} import build_login_query


def test_parameterized_login_query():
    target_user = "guest' UNION SELECT 1, 'pw'--"
    query, params = build_login_query(target_user)
    assert "?" in query
    assert target_user not in query
    assert params == (target_user,)
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Use parameterized SQL query to secure login verification against injection",
    )


def gen_sql_injection_content(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"post_search_{salt}"
    buggy = f"""\"\"\"Post search query module {salt}.\"\"\"


def build_search_query(tag: str) -> tuple[str, tuple]:
    # Raw SQL formatting susceptible to SQL injection
    query = f"SELECT id, title FROM posts WHERE tag = '{{tag}}'"
    return query, ()
"""
    fixed = f"""\"\"\"Post search query module {salt}.\"\"\"


def build_search_query(tag: str) -> tuple[str, tuple]:
    query = "SELECT id, title FROM posts WHERE tag = ?"
    return query, (tag,)
"""
    test = f"""\"\"\"Test for post search query module {salt}.\"\"\"
from {mod} import build_search_query


def test_parameterized_search_query():
    malicious_tag = "python'; DROP TABLE posts;--"
    query, params = build_search_query(malicious_tag)
    assert "?" in query
    assert malicious_tag not in query
    assert params == (malicious_tag,)
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Convert string-formatted post query into safe parameterized SQL statement",
    )


def gen_sql_injection_pricing(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"order_lookup_{salt}"
    buggy = f"""\"\"\"Order lookup query module {salt}.\"\"\"


def build_order_lookup_query(status: str) -> tuple[str, tuple]:
    # Vulnerable f-string query formatting
    query = f"SELECT order_id, total FROM orders WHERE status = '{{status}}'"
    return query, ()
"""
    fixed = f"""\"\"\"Order lookup query module {salt}.\"\"\"


def build_order_lookup_query(status: str) -> tuple[str, tuple]:
    query = "SELECT order_id, total FROM orders WHERE status = ?"
    return query, (status,)
"""
    test = f"""\"\"\"Test for order lookup query module {salt}.\"\"\"
from {mod} import build_order_lookup_query


def test_parameterized_order_lookup():
    status_input = "PENDING' OR '1'='1"
    query, params = build_order_lookup_query(status_input)
    assert "?" in query
    assert status_input not in query
    assert params == (status_input,)
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Parameterize order status lookup query to eliminate SQL injection threat",
    )


# ---------------------------------------------------------------------------
# Registry of (Taxonomy, Template) -> GeneratorFunc
# ---------------------------------------------------------------------------

TAXONOMY_TEMPLATE_GENERATORS: dict[tuple[BugTaxonomy, str], GeneratorFunc] = {
    # OFF_BY_ONE
    (BugTaxonomy.OFF_BY_ONE, "inventory_mgmt"): gen_off_by_one_inventory,
    (BugTaxonomy.OFF_BY_ONE, "order_pricing"): gen_off_by_one_pricing,
    (BugTaxonomy.OFF_BY_ONE, "content_feed"): gen_off_by_one_content,
    (BugTaxonomy.OFF_BY_ONE, "analytics_tracker"): gen_off_by_one_analytics,
    # WRONG_OPERATOR
    (BugTaxonomy.WRONG_OPERATOR, "inventory_mgmt"): gen_wrong_operator_inventory,
    (BugTaxonomy.WRONG_OPERATOR, "user_auth"): gen_wrong_operator_auth,
    (BugTaxonomy.WRONG_OPERATOR, "order_pricing"): gen_wrong_operator_pricing,
    (BugTaxonomy.WRONG_OPERATOR, "payment_gateway"): gen_wrong_operator_payment,
    # NONE_HANDLING
    (BugTaxonomy.NONE_HANDLING, "user_auth"): gen_none_handling_auth,
    (BugTaxonomy.NONE_HANDLING, "cache_store"): gen_none_handling_cache,
    (BugTaxonomy.NONE_HANDLING, "data_pipeline"): gen_none_handling_pipeline,
    (BugTaxonomy.NONE_HANDLING, "content_feed"): gen_none_handling_content,
    # MISSING_AWAIT
    (BugTaxonomy.MISSING_AWAIT, "payment_gateway"): gen_missing_await_payment,
    (BugTaxonomy.MISSING_AWAIT, "notification_queue"): gen_missing_await_notification,
    (BugTaxonomy.MISSING_AWAIT, "data_pipeline"): gen_missing_await_pipeline,
    (BugTaxonomy.MISSING_AWAIT, "audit_logger"): gen_missing_await_audit,
    # RESOURCE_LEAK
    (BugTaxonomy.RESOURCE_LEAK, "inventory_mgmt"): gen_resource_leak_inventory,
    (BugTaxonomy.RESOURCE_LEAK, "data_pipeline"): gen_resource_leak_pipeline,
    (BugTaxonomy.RESOURCE_LEAK, "audit_logger"): gen_resource_leak_audit,
    (BugTaxonomy.RESOURCE_LEAK, "notification_queue"): gen_resource_leak_notification,
    # BAD_DEFAULT_ARG
    (BugTaxonomy.BAD_DEFAULT_ARG, "inventory_mgmt"): gen_bad_default_inventory,
    (BugTaxonomy.BAD_DEFAULT_ARG, "order_pricing"): gen_bad_default_pricing,
    (BugTaxonomy.BAD_DEFAULT_ARG, "cache_store"): gen_bad_default_cache,
    (BugTaxonomy.BAD_DEFAULT_ARG, "user_auth"): gen_bad_default_auth,
    # SQL_INJECTION
    (BugTaxonomy.SQL_INJECTION, "audit_logger"): gen_sql_injection_audit,
    (BugTaxonomy.SQL_INJECTION, "user_auth"): gen_sql_injection_auth,
    (BugTaxonomy.SQL_INJECTION, "content_feed"): gen_sql_injection_content,
    (BugTaxonomy.SQL_INJECTION, "order_pricing"): gen_sql_injection_pricing,
}


def compute_case_content_hash(files: dict[str, str], patch_diff: str) -> str:
    """Compute deterministic SHA-256 over repository files and unified patch diff."""
    hasher = hashlib.sha256()
    for path, content in sorted(files.items()):
        hasher.update(path.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(content.encode("utf-8"))
        hasher.update(b"\x00")
    hasher.update(b"__patch__\x00")
    hasher.update(patch_diff.encode("utf-8"))
    return hasher.hexdigest()


def compute_source_hash(source_code: str) -> str:
    """Compute normalized SHA-256 over Python source code."""
    lines = [line.rstrip() for line in source_code.strip().splitlines() if line.strip()]
    normalized = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def compute_unified_diff(buggy_code: str, fixed_code: str, filepath: str) -> str:
    """Produce canonical unified diff matching standard git/patch format."""
    diff_lines = list(
        difflib.unified_diff(
            buggy_code.splitlines(keepends=True),
            fixed_code.splitlines(keepends=True),
            fromfile=f"a/{filepath}",
            tofile=f"b/{filepath}",
            lineterm="\n",
        )
    )
    return "".join(diff_lines)


def get_template_distribution() -> dict[str, list[str]]:
    """Return dictionary of taxonomy class -> list of unique templates."""
    distribution: dict[str, list[str]] = {}
    for tax, tpl in sorted(TAXONOMY_TEMPLATE_GENERATORS.keys(), key=lambda x: (x[0].value, x[1])):
        distribution.setdefault(tax.value, []).append(tpl)
    return distribution


def build_seeded_case(
    *,
    case_id: str,
    pool: str,
    taxonomy: BugTaxonomy,
    template: str,
    seed: int,
) -> CaseDefinition:
    """Generate a single reproducible case given taxonomy, template, and seed."""
    gen_func = TAXONOMY_TEMPLATE_GENERATORS.get((taxonomy, template))
    if gen_func is None:
        raise ValueError(f"No generator found for ({taxonomy.value}, {template})")

    gen_res = gen_func(seed)
    filename = f"{gen_res.module_name}.py"
    test_filename = f"test_{gen_res.module_name}.py"

    patch_diff = compute_unified_diff(gen_res.buggy_code, gen_res.fixed_code, filename)
    files = {
        filename: gen_res.buggy_code,
        test_filename: gen_res.test_code,
        "pyproject.toml": (
            "[project]\nname = 'case'\nversion = '0.0.0'\n\n"
            "[tool.pytest.ini_options]\npythonpath = ['.']\n"
        ),
    }

    content_hash = compute_case_content_hash(files, patch_diff)
    sig_salt = hashlib.sha256(f"{seed}:{content_hash}".encode()).hexdigest()[:12]
    mutation_signature = f"{template}::{taxonomy.value}::{sig_salt}"

    metadata = CaseMetadata(
        case_id=case_id,
        pool=pool,
        taxonomy=taxonomy.value,
        template=template,
        mutation_signature=mutation_signature,
        seed=seed,
        content_hash=content_hash,
        patch_diff=patch_diff,
        pytest_args=["-q"],
        license="MIT",
        license_source="self-generated",
        description=gen_res.description,
    )

    return CaseDefinition(
        metadata=metadata,
        files=files,
        symlinks={},
        patch_diff=patch_diff,
    )
