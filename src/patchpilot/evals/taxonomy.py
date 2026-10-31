"""Bug taxonomy definitions and template-based case generators for Phase 3 evaluation datasets."""

from __future__ import annotations

import difflib
import hashlib
import random
from collections.abc import Callable
from dataclasses import dataclass

from patchpilot.evals.models import BugTaxonomy, CaseDefinition, CaseMetadata


@dataclass(frozen=True)
class GeneratorResult:
    module_name: str
    buggy_code: str
    fixed_code: str
    test_code: str
    description: str


GeneratorFunc = Callable[[int], GeneratorResult]


# ---------------------------------------------------------------------------
# 1. OFF_BY_ONE (8 templates: 2 eval, 6 train)
# ---------------------------------------------------------------------------


# Template 1 (eval): pagination_offset
def gen_off_by_one_pagination_offset(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    ps = rng.choice([4, 5, 8, 10])
    salt = rng.randint(100, 999)
    mod = f"inv_page_{salt}"
    buggy = f"""\"\"\"Pagination offset module {salt}.\"\"\"


def paginate_inventory(items: list[str], page: int, page_size: int = {ps}) -> list[str]:
    # 1-indexed pagination bug: offsets by whole page too far
    offset = page * page_size
    return items[offset : offset + page_size]
"""
    fixed = f"""\"\"\"Pagination offset module {salt}.\"\"\"


def paginate_inventory(items: list[str], page: int, page_size: int = {ps}) -> list[str]:
    offset = (page - 1) * page_size
    return items[offset : offset + page_size]
"""
    test = f"""\"\"\"Test for pagination offset module {salt}.\"\"\"
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
        description="Fix off-by-one error in 1-indexed pagination offset calculation",
    )


# Template 2 (eval): tier_threshold
def gen_off_by_one_tier_threshold(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    thresh = rng.choice([10, 15, 20])
    pct = rng.choice([10, 15, 25])
    salt = rng.randint(100, 999)
    mod = f"pricing_tier_{salt}"
    buggy = f"""\"\"\"Tier threshold module {salt}.\"\"\"


def calculate_tier_discount(quantity: int, unit_price: float) -> float:
    # Off-by-one in tier threshold: > instead of >=
    if quantity > {thresh}:
        return round(quantity * unit_price * {pct / 100:.2f}, 2)
    return 0.0
"""
    fixed = f"""\"\"\"Tier threshold module {salt}.\"\"\"


def calculate_tier_discount(quantity: int, unit_price: float) -> float:
    if quantity >= {thresh}:
        return round(quantity * unit_price * {pct / 100:.2f}, 2)
    return 0.0
"""
    expected = round(thresh * 10.0 * (pct / 100), 2)
    test = f"""\"\"\"Test for tier threshold module {salt}.\"\"\"
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


# Template 3 (train): slice_limit
def gen_off_by_one_slice_limit(seed: int) -> GeneratorResult:
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
        description="Fix slice off-by-one returning one post beyond requested limit",
    )


# Template 4 (train): sliding_window
def gen_off_by_one_sliding_window(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    w = rng.choice([3, 4, 5])
    salt = rng.randint(100, 999)
    mod = f"window_count_{salt}"
    buggy = f"""\"\"\"Sliding window module {salt}.\"\"\"


def compute_sliding_window_count(data_length: int, window_size: int = {w}) -> int:
    if data_length < window_size:
        return 0
    # Fencepost error: misses final valid window
    return data_length - window_size
"""
    fixed = f"""\"\"\"Sliding window module {salt}.\"\"\"


def compute_sliding_window_count(data_length: int, window_size: int = {w}) -> int:
    if data_length < window_size:
        return 0
    return data_length - window_size + 1
"""
    test = f"""\"\"\"Test for sliding window module {salt}.\"\"\"
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


# Template 5 (train): binary_search_range
def gen_off_by_one_binary_search(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"bin_search_{salt}"
    buggy = f"""\"\"\"Binary search range module {salt}.\"\"\"


def find_index(arr: list[int], target: int) -> int:
    low = 0
    high = len(arr) - 1
    # Bug: terminates before checking low == high
    while low < high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
"""
    fixed = f"""\"\"\"Binary search range module {salt}.\"\"\"


def find_index(arr: list[int], target: int) -> int:
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
"""
    test = f"""\"\"\"Test for binary search range module {salt}.\"\"\"
from {mod} import find_index


def test_single_element_search():
    arr = [10, 20, 30, 40, 50]
    assert find_index(arr, 50) == 4
    assert find_index([42], 42) == 0
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Fix binary search boundary condition while low <= high",
    )


# Template 6 (train): chunk_partition
def gen_off_by_one_chunk_partition(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    chunk = rng.choice([3, 4, 6])
    salt = rng.randint(100, 999)
    mod = f"chunk_part_{salt}"
    buggy = f"""\"\"\"Chunk partition module {salt}.\"\"\"


def count_batches(total_items: int, chunk_size: int = {chunk}) -> int:
    if total_items <= 0:
        return 0
    # Bug: floor division drops final partial chunk
    return total_items // chunk_size
"""
    fixed = f"""\"\"\"Chunk partition module {salt}.\"\"\"


def count_batches(total_items: int, chunk_size: int = {chunk}) -> int:
    if total_items <= 0:
        return 0
    return (total_items + chunk_size - 1) // chunk_size
"""
    test = f"""\"\"\"Test for chunk partition module {salt}.\"\"\"
from {mod} import count_batches


def test_chunk_counting_with_remainder():
    total = {chunk} * 3 + 1
    assert count_batches(total, {chunk}) == 4
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Fix ceiling division in batch chunk count for partial final batch",
    )


# Template 7 (train): inclusive_range
def gen_off_by_one_inclusive_range(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"range_span_{salt}"
    buggy = f"""\"\"\"Inclusive range module {salt}.\"\"\"


def days_between_inclusive(start_day: int, end_day: int) -> int:
    if start_day > end_day:
        return 0
    # Bug: excludes endpoints count
    return end_day - start_day
"""
    fixed = f"""\"\"\"Inclusive range module {salt}.\"\"\"


def days_between_inclusive(start_day: int, end_day: int) -> int:
    if start_day > end_day:
        return 0
    return end_day - start_day + 1
"""
    test = f"""\"\"\"Test for inclusive range module {salt}.\"\"\"
from {mod} import days_between_inclusive


def test_inclusive_day_span():
    assert days_between_inclusive(5, 5) == 1
    assert days_between_inclusive(1, 10) == 10
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Account for both endpoints in inclusive date range duration calculation",
    )


# Template 8 (train): buffer_stride
def gen_off_by_one_buffer_stride(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    stride = rng.choice([2, 3, 4])
    salt = rng.randint(100, 999)
    mod = f"buf_stride_{salt}"
    buggy = f"""\"\"\"Buffer stride module {salt}.\"\"\"


def extract_stride_samples(buffer: list[int], stride: int = {stride}) -> list[int]:
    # Bug: stops one stride too early
    limit = len(buffer) - stride
    return [buffer[i] for i in range(0, max(0, limit), stride)]
"""
    fixed = f"""\"\"\"Buffer stride module {salt}.\"\"\"


def extract_stride_samples(buffer: list[int], stride: int = {stride}) -> list[int]:
    limit = len(buffer) - stride + 1
    return [buffer[i] for i in range(0, max(0, limit), stride)]
"""
    test = f"""\"\"\"Test for buffer stride module {salt}.\"\"\"
from {mod} import extract_stride_samples


def test_buffer_stride_exact():
    buf = list(range({stride} * 3))
    samples = extract_stride_samples(buf, {stride})
    assert len(samples) == 3
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Include boundary window in buffer stride sampling loop",
    )


# ---------------------------------------------------------------------------
# 2. WRONG_OPERATOR (8 templates: 2 eval, 6 train)
# ---------------------------------------------------------------------------


# Template 1 (eval): reorder_threshold
def gen_wrong_operator_reorder_threshold(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    thresh = rng.choice([10, 15, 25])
    salt = rng.randint(100, 999)
    mod = f"stock_check_{salt}"
    buggy = f"""\"\"\"Stock check module {salt}.\"\"\"


def needs_reorder(current_stock: int, threshold: int = {thresh}) -> bool:
    # Wrong operator: > instead of <=
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


# Template 2 (eval): boolean_permission
def gen_wrong_operator_boolean_permission(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"perm_eval_{salt}"
    buggy = f"""\"\"\"Permission evaluation module {salt}.\"\"\"


def can_moderate_content(is_admin: bool, is_moderator: bool) -> bool:
    # Wrong operator: and instead of or
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


# Template 3 (train): voucher_reduction
def gen_wrong_operator_voucher_reduction(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    discount = rng.choice([5.0, 10.0, 15.0])
    salt = rng.randint(100, 999)
    mod = f"voucher_calc_{salt}"
    buggy = f"""\"\"\"Voucher calculation module {salt}.\"\"\"


def apply_voucher(subtotal: float, voucher_value: float = {discount}) -> float:
    # Wrong operator: + instead of -
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


# Template 4 (train): credit_limit_check
def gen_wrong_operator_credit_limit(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    limit = rng.choice([500.0, 1000.0, 2000.0])
    salt = rng.randint(100, 999)
    mod = f"charge_perm_{salt}"
    buggy = f"""\"\"\"Charge permission module {salt}.\"\"\"


def is_charge_valid(balance: float, amount: float, credit_limit: float = {limit}) -> bool:
    # Wrong operator: < instead of >=
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


# Template 5 (train): role_membership
def gen_wrong_operator_role_membership(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"role_gate_{salt}"
    buggy = f"""\"\"\"Role gate module {salt}.\"\"\"


def is_role_denied(user_role: str, blocked_roles: list[str]) -> bool:
    # Wrong operator: not in instead of in
    return user_role not in blocked_roles
"""
    fixed = f"""\"\"\"Role gate module {salt}.\"\"\"


def is_role_denied(user_role: str, blocked_roles: list[str]) -> bool:
    return user_role in blocked_roles
"""
    test = f"""\"\"\"Test for role gate module {salt}.\"\"\"
from {mod} import is_role_denied


def test_blocked_role_denied():
    assert is_role_denied("banned_user", ["banned_user", "suspended"]) is True
    assert is_role_denied("normal_user", ["banned_user", "suspended"]) is False
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Use 'in' instead of 'not in' for blocked role evaluation",
    )


# Template 6 (train): surcharge_math
def gen_wrong_operator_surcharge(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    rate = rng.choice([0.05, 0.08, 0.12])
    salt = rng.randint(100, 999)
    mod = f"tax_surcharge_{salt}"
    buggy = f"""\"\"\"Tax surcharge module {salt}.\"\"\"


def calculate_surcharged_total(base: float, rate: float = {rate}) -> float:
    # Wrong operator: - instead of +
    return round(base * (1.0 - rate), 2)
"""
    fixed = f"""\"\"\"Tax surcharge module {salt}.\"\"\"


def calculate_surcharged_total(base: float, rate: float = {rate}) -> float:
    return round(base * (1.0 + rate), 2)
"""
    test = f"""\"\"\"Test for tax surcharge module {salt}.\"\"\"
from {mod} import calculate_surcharged_total


def test_surcharge_increases_total():
    total = calculate_surcharged_total(100.0, {rate})
    assert total == round(100.0 * (1.0 + {rate}), 2)
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Apply addition for tax surcharge instead of subtraction",
    )


# Template 7 (train): retry_budget
def gen_wrong_operator_retry_budget(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    max_retries = rng.choice([3, 4, 5])
    salt = rng.randint(100, 999)
    mod = f"retry_ctrl_{salt}"
    buggy = f"""\"\"\"Retry control module {salt}.\"\"\"


def can_retry_attempt(attempt_count: int, max_retries: int = {max_retries}) -> bool:
    # Wrong operator: > instead of <
    return attempt_count > max_retries
"""
    fixed = f"""\"\"\"Retry control module {salt}.\"\"\"


def can_retry_attempt(attempt_count: int, max_retries: int = {max_retries}) -> bool:
    return attempt_count < max_retries
"""
    test = f"""\"\"\"Test for retry control module {salt}.\"\"\"
from {mod} import can_retry_attempt


def test_can_retry():
    assert can_retry_attempt(0, {max_retries}) is True
    assert can_retry_attempt({max_retries}, {max_retries}) is False
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Fix comparison operator governing attempt retry eligibility",
    )


# Template 8 (train): bitmask_flag
def gen_wrong_operator_bitmask(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    flag_val = rng.choice([1, 2, 4, 8])
    salt = rng.randint(100, 999)
    mod = f"bitmask_eval_{salt}"
    buggy = f"""\"\"\"Bitmask evaluation module {salt}.\"\"\"


def has_flag(bitmask: int, flag: int = {flag_val}) -> bool:
    # Wrong operator: | instead of &
    return bool((bitmask | flag) == 0)
"""
    fixed = f"""\"\"\"Bitmask evaluation module {salt}.\"\"\"


def has_flag(bitmask: int, flag: int = {flag_val}) -> bool:
    return bool((bitmask & flag) != 0)
"""
    test = f"""\"\"\"Test for bitmask evaluation module {salt}.\"\"\"
from {mod} import has_flag


def test_bitmask_presence():
    mask = {flag_val} | 16
    assert has_flag(mask, {flag_val}) is True
    assert has_flag(32, {flag_val}) is False
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Use bitwise AND operator to evaluate flag presence in bitmask",
    )


# ---------------------------------------------------------------------------
# 3. NONE_HANDLING (8 templates: 2 eval, 6 train)
# ---------------------------------------------------------------------------


# Template 1 (eval): profile_display_name
def gen_none_handling_profile_display(seed: int) -> GeneratorResult:
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


# Template 2 (eval): cache_subscript
def gen_none_handling_cache_subscript(seed: int) -> GeneratorResult:
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


# Template 3 (train): pipeline_normalize
def gen_none_handling_pipeline_normalize(seed: int) -> GeneratorResult:
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


# Template 4 (train): comment_author
def gen_none_handling_comment_author(seed: int) -> GeneratorResult:
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


# Template 5 (train): config_timeout_fallback
def gen_none_handling_config_timeout(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    default_t = rng.choice([30, 45, 60])
    salt = rng.randint(100, 999)
    mod = f"cfg_timeout_{salt}"
    buggy = f"""\"\"\"Config timeout module {salt}.\"\"\"


def get_timeout_millis(settings: dict, default_s: int = {default_t}) -> int:
    # Fails with TypeError when settings['timeout'] is explicitly None
    val = settings.get("timeout")
    return int(val * 1000)
"""
    fixed = f"""\"\"\"Config timeout module {salt}.\"\"\"


def get_timeout_millis(settings: dict, default_s: int = {default_t}) -> int:
    val = settings.get("timeout")
    effective = val if val is not None else default_s
    return int(effective * 1000)
"""
    test = f"""\"\"\"Test for config timeout module {salt}.\"\"\"
from {mod} import get_timeout_millis


def test_none_timeout_setting():
    settings = {{"timeout": None}}
    assert get_timeout_millis(settings, {default_t}) == {default_t * 1000}
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Fallback to default timeout when config dictionary timeout is None",
    )


# Template 6 (train): header_split
def gen_none_handling_header_split(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"header_split_{salt}"
    buggy = f"""\"\"\"Header parsing module {salt}.\"\"\"


def parse_scopes(headers: dict) -> list[str]:
    # Fails with AttributeError when header value is None
    scope_str = headers.get("X-Scope")
    return [s.strip() for s in scope_str.split(",") if s.strip()]
"""
    fixed = f"""\"\"\"Header parsing module {salt}.\"\"\"


def parse_scopes(headers: dict) -> list[str]:
    scope_str = headers.get("X-Scope")
    if scope_str is None:
        return []
    return [s.strip() for s in scope_str.split(",") if s.strip()]
"""
    test = f"""\"\"\"Test for header parsing module {salt}.\"\"\"
from {mod} import parse_scopes


def test_missing_or_none_scope_header():
    assert parse_scopes({{"X-Scope": None}}) == []
    assert parse_scopes({{}}) == []
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Handle None header value safely before splitting scope string",
    )


# Template 7 (train): list_element_none_filter
def gen_none_handling_list_filter(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"sensor_agg_{salt}"
    buggy = f"""\"\"\"Sensor aggregation module {salt}.\"\"\"


def sum_valid_readings(readings: list[int | None]) -> int:
    # Fails with TypeError: unsupported operand type for +: 'int' and 'NoneType'
    return sum(readings)  # type: ignore[arg-type]
"""
    fixed = f"""\"\"\"Sensor aggregation module {salt}.\"\"\"


def sum_valid_readings(readings: list[int | None]) -> int:
    return sum(r for r in readings if r is not None)
"""
    test = f"""\"\"\"Test for sensor aggregation module {salt}.\"\"\"
from {mod} import sum_valid_readings


def test_sum_with_none_elements():
    data = [10, None, 20, None, 30]
    assert sum_valid_readings(data) == 60
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Filter None elements from numerical list before sum aggregation",
    )


# Template 8 (train): optional_callback
def gen_none_handling_optional_callback(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"callback_exec_{salt}"
    buggy = f"""\"\"\"Callback execution module {salt}.\"\"\"


def transform_data(payload: str, callback=None) -> str:
    # Fails with TypeError: 'NoneType' object is not callable
    return callback(payload)
"""
    fixed = f"""\"\"\"Callback execution module {salt}.\"\"\"


def transform_data(payload: str, callback=None) -> str:
    if callback is not None:
        return callback(payload)
    return payload
"""
    test = f"""\"\"\"Test for callback execution module {salt}.\"\"\"
from {mod} import transform_data


def test_optional_callback_none():
    assert transform_data("raw_content", callback=None) == "raw_content"
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Check if callback is not None before invoking transformation hook",
    )


# ---------------------------------------------------------------------------
# 4. MISSING_AWAIT (8 templates: 2 eval, 6 train)
# ---------------------------------------------------------------------------


# Template 1 (eval): payment_authorize
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


# Template 2 (eval): alert_broadcast
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


# Template 3 (train): pipeline_batch_fetch
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


# Template 4 (train): audit_entry_write
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


# Template 5 (train): http_fetch_json
def gen_missing_await_http_client(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"http_fetch_{salt}"
    buggy = f"""\"\"\"Async HTTP fetch module {salt}.\"\"\"


async def fetch_item_identifier(client, endpoint: str) -> int:
    # Missing await on async client get_json call
    data = client.get_json(endpoint)
    return int(data["id"])
"""
    fixed = f"""\"\"\"Async HTTP fetch module {salt}.\"\"\"


async def fetch_item_identifier(client, endpoint: str) -> int:
    data = await client.get_json(endpoint)
    return int(data["id"])
"""
    test = f"""\"\"\"Test for async HTTP fetch module {salt}.\"\"\"
import asyncio
from {mod} import fetch_item_identifier


class MockHttpClient:
    async def get_json(self, endpoint: str) -> dict:
        return {{"id": 42, "endpoint": endpoint}}


def test_fetch_identifier():
    client = MockHttpClient()
    item_id = asyncio.run(fetch_item_identifier(client, "/api/item"))
    assert item_id == 42
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Await async HTTP JSON fetch coroutine before reading dictionary key",
    )


# Template 6 (train): cache_async_lookup
def gen_missing_await_cache_async(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"async_cache_{salt}"
    buggy = f"""\"\"\"Async cache lookup module {salt}.\"\"\"


async def lookup_cached_name(cache_client, key: str) -> str:
    # Missing await on async cache get
    val = cache_client.get(key)
    return str(val.upper())
"""
    fixed = f"""\"\"\"Async cache lookup module {salt}.\"\"\"


async def lookup_cached_name(cache_client, key: str) -> str:
    val = await cache_client.get(key)
    return str(val.upper())
"""
    test = f"""\"\"\"Test for async cache lookup module {salt}.\"\"\"
import asyncio
from {mod} import lookup_cached_name


class MockCache:
    async def get(self, key: str) -> str:
        return "cached_token"


def test_lookup_cached_name():
    cache = MockCache()
    res = asyncio.run(lookup_cached_name(cache, "auth_key"))
    assert res == "CACHED_TOKEN"
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Await async cache get coroutine before string manipulation",
    )


# Template 7 (train): queue_pop_payload
def gen_missing_await_queue_pop(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"async_queue_{salt}"
    buggy = f"""\"\"\"Async queue consumer module {salt}.\"\"\"


async def read_next_task(queue) -> str:
    # Missing await on async queue pop
    msg = queue.pop()
    return msg["task"]
"""
    fixed = f"""\"\"\"Async queue consumer module {salt}.\"\"\"


async def read_next_task(queue) -> str:
    msg = await queue.pop()
    return msg["task"]
"""
    test = f"""\"\"\"Test for async queue consumer module {salt}.\"\"\"
import asyncio
from {mod} import read_next_task


class MockQueue:
    async def pop(self) -> dict:
        return {{"task": "process_job_1"}}


def test_read_next_task():
    q = MockQueue()
    task = asyncio.run(read_next_task(q))
    assert task == "process_job_1"
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Await async queue pop coroutine before accessing message dictionary",
    )


# Template 8 (train): db_async_query
def gen_missing_await_db_query(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"async_db_{salt}"
    buggy = f"""\"\"\"Async database query module {salt}.\"\"\"


async def count_query_results(session, sql: str) -> int:
    # Missing await on async session execute
    rows = session.execute(sql)
    return len(rows)
"""
    fixed = f"""\"\"\"Async database query module {salt}.\"\"\"


async def count_query_results(session, sql: str) -> int:
    rows = await session.execute(sql)
    return len(rows)
"""
    test = f"""\"\"\"Test for async database query module {salt}.\"\"\"
import asyncio
from {mod} import count_query_results


class MockDbSession:
    async def execute(self, sql: str) -> list[dict]:
        return [{{"id": 1}}, [{{"id": 2}}]]


def test_count_query_results():
    s = MockDbSession()
    count = asyncio.run(count_query_results(s, "SELECT 1"))
    assert count == 2
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Await async database session execute coroutine before counting row count",
    )


# ---------------------------------------------------------------------------
# 5. RESOURCE_LEAK (8 templates: 2 eval, 6 train)
# ---------------------------------------------------------------------------


# Template 1 (eval): export_channel
def gen_resource_leak_export_channel(seed: int) -> GeneratorResult:
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


# Template 2 (eval): data_stream_reader
def gen_resource_leak_stream_reader(seed: int) -> GeneratorResult:
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


# Template 3 (train): audit_db_conn
def gen_resource_leak_audit_conn(seed: int) -> GeneratorResult:
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


# Template 4 (train): socket_transport
def gen_resource_leak_socket_transport(seed: int) -> GeneratorResult:
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


# Template 5 (train): file_handle_writer
def gen_resource_leak_file_handle(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"file_writer_{salt}"
    buggy = f"""\"\"\"File handle writer module {salt}.\"\"\"


class LogHandle:
    def __init__(self) -> None:
        self.closed = False

    def write_line(self, line: str) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def write_system_log(handle: LogHandle, text: str) -> bool:
    # Bug: handle not closed
    handle.write_line(text)
    return True
"""
    fixed = f"""\"\"\"File handle writer module {salt}.\"\"\"


class LogHandle:
    def __init__(self) -> None:
        self.closed = False

    def write_line(self, line: str) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def write_system_log(handle: LogHandle, text: str) -> bool:
    with handle:
        handle.write_line(text)
    return True
"""
    test = f"""\"\"\"Test for file handle writer module {salt}.\"\"\"
from {mod} import LogHandle, write_system_log


def test_file_handle_closed():
    h = LogHandle()
    write_system_log(h, "event_occurred")
    assert h.closed is True
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Wrap file log handle in with statement to prevent handle descriptor leak",
    )


# Template 6 (train): temp_buffer_cleanup
def gen_resource_leak_temp_buffer(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"scratch_buf_{salt}"
    buggy = f"""\"\"\"Scratch buffer module {salt}.\"\"\"


class ScratchBuffer:
    def __init__(self) -> None:
        self.cleaned_up = False

    def populate(self, size: int) -> None:
        pass

    def cleanup(self) -> None:
        self.cleaned_up = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def allocate_and_compute(buf: ScratchBuffer, size: int) -> int:
    # Bug: buffer cleanup omitted
    buf.populate(size)
    return size * 2
"""
    fixed = f"""\"\"\"Scratch buffer module {salt}.\"\"\"


class ScratchBuffer:
    def __init__(self) -> None:
        self.cleaned_up = False

    def populate(self, size: int) -> None:
        pass

    def cleanup(self) -> None:
        self.cleaned_up = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def allocate_and_compute(buf: ScratchBuffer, size: int) -> int:
    with buf:
        buf.populate(size)
    return size * 2
"""
    test = f"""\"\"\"Test for scratch buffer module {salt}.\"\"\"
from {mod} import ScratchBuffer, allocate_and_compute


def test_scratch_cleaned_up():
    buf = ScratchBuffer()
    res = allocate_and_compute(buf, 10)
    assert res == 20
    assert buf.cleaned_up is True
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Ensure scratch buffer cleanup method is called upon exit",
    )


# Template 7 (train): lock_release
def gen_resource_leak_lock_release(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"managed_lock_{salt}"
    buggy = f"""\"\"\"Managed lock module {salt}.\"\"\"


class ManagedLock:
    def __init__(self) -> None:
        self.held = False

    def acquire(self) -> None:
        self.held = True

    def release(self) -> None:
        self.held = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def perform_guarded_action(lock: ManagedLock, val: int) -> int:
    # Bug: acquires lock but never releases it
    lock.acquire()
    return val * 3
"""
    fixed = f"""\"\"\"Managed lock module {salt}.\"\"\"


class ManagedLock:
    def __init__(self) -> None:
        self.held = False

    def acquire(self) -> None:
        self.held = True

    def release(self) -> None:
        self.held = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def perform_guarded_action(lock: ManagedLock, val: int) -> int:
    with lock:
        return val * 3
"""
    test = f"""\"\"\"Test for managed lock module {salt}.\"\"\"
from {mod} import ManagedLock, perform_guarded_action


def test_lock_released_after_action():
    lock = ManagedLock()
    res = perform_guarded_action(lock, 5)
    assert res == 15
    assert lock.held is False
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Release acquired synchronization lock via context manager",
    )


# Template 8 (train): session_pool_checkout
def gen_resource_leak_session_checkout(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"pool_checkout_{salt}"
    buggy = f"""\"\"\"Pool checkout module {salt}.\"\"\"


class PooledSession:
    def __init__(self) -> None:
        self.in_use = True

    def close(self) -> None:
        self.in_use = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def query_and_release(session: PooledSession, item_id: int) -> str:
    # Bug: checkout session not closed/returned to pool
    return f"result_{{item_id}}"
"""
    fixed = f"""\"\"\"Pool checkout module {salt}.\"\"\"


class PooledSession:
    def __init__(self) -> None:
        self.in_use = True

    def close(self) -> None:
        self.in_use = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def query_and_release(session: PooledSession, item_id: int) -> str:
    with session:
        return f"result_{{item_id}}"
"""
    test = f"""\"\"\"Test for pool checkout module {salt}.\"\"\"
from {mod} import PooledSession, query_and_release


def test_session_released():
    sess = PooledSession()
    res = query_and_release(sess, 99)
    assert res == "result_99"
    assert sess.in_use is False
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Return checked-out pooled session on function return",
    )


# ---------------------------------------------------------------------------
# 6. BAD_DEFAULT_ARG (8 templates: 2 eval, 6 train)
# ---------------------------------------------------------------------------


# Template 1 (eval): item_tags
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
    register_inventory_item("Alpha")
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


# Template 2 (eval): order_items
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
    add_order_line("SKU-100")
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


# Template 3 (train): cache_metadata
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
    init_cache_metadata("session_a")
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


# Template 4 (train): user_roles
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
    assign_base_roles("alice")
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


# Template 5 (train): metrics_buffer
def gen_bad_default_metrics_buffer(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"metric_buf_{salt}"
    buggy = f"""\"\"\"Metrics buffer module {salt}.\"\"\"


def record_metric_sample(value: float, buffer: list[float] = []) -> list[float]:
    # Mutable default list buffer
    buffer.append(value)
    return buffer
"""
    fixed = f"""\"\"\"Metrics buffer module {salt}.\"\"\"


def record_metric_sample(value: float, buffer: list[float] | None = None) -> list[float]:
    if buffer is None:
        buffer = []
    buffer.append(value)
    return buffer
"""
    test = f"""\"\"\"Test for metrics buffer module {salt}.\"\"\"
from {mod} import record_metric_sample


def test_metrics_buffer_isolation():
    record_metric_sample(10.5)
    second = record_metric_sample(20.0)
    assert second == [20.0]
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Prevent cross-call metric leakage via mutable default buffer argument",
    )


# Template 6 (train): headers_dict
def gen_bad_default_headers_dict(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"headers_dict_{salt}"
    buggy = f"""\"\"\"Headers dict module {salt}.\"\"\"


def build_request_headers(token: str, headers: dict = {{}}) -> dict:
    # Mutable default dict argument
    headers["Authorization"] = f"Bearer {{token}}"
    return headers
"""
    fixed = f"""\"\"\"Headers dict module {salt}.\"\"\"


def build_request_headers(token: str, headers: dict | None = None) -> dict:
    if headers is None:
        headers = {{}}
    headers["Authorization"] = f"Bearer {{token}}"
    return headers
"""
    test = f"""\"\"\"Test for headers dict module {salt}.\"\"\"
from {mod} import build_request_headers


def test_headers_isolation():
    h1 = build_request_headers("secret_token_a")
    h1["X-Extra"] = "injected"
    h2 = build_request_headers("secret_token_b")
    assert "X-Extra" not in h2
    assert h2 == {{"Authorization": "Bearer secret_token_b"}}
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Use None sentinel instead of mutable dict default in HTTP headers builder",
    )


# Template 7 (train): category_filter_list
def gen_bad_default_category_filter(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"cat_filter_{salt}"
    buggy = f"""\"\"\"Category filter module {salt}.\"\"\"


def add_active_category(category: str, active: list[str] = []) -> list[str]:
    # Mutable default list
    active.append(category)
    return active
"""
    fixed = f"""\"\"\"Category filter module {salt}.\"\"\"


def add_active_category(category: str, active: list[str] | None = None) -> list[str]:
    if active is None:
        active = []
    active.append(category)
    return active
"""
    test = f"""\"\"\"Test for category filter module {salt}.\"\"\"
from {mod} import add_active_category


def test_active_categories_isolation():
    add_active_category("electronics")
    second = add_active_category("books")
    assert second == ["books"]
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Isolate active categories list using None sentinel default",
    )


# Template 8 (train): listener_registry
def gen_bad_default_listeners(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"listener_reg_{salt}"
    buggy = f"""\"\"\"Listener registry module {salt}.\"\"\"


def register_handler(name: str, registry: list[str] = []) -> list[str]:
    # Mutable default list
    registry.append(name)
    return registry
"""
    fixed = f"""\"\"\"Listener registry module {salt}.\"\"\"


def register_handler(name: str, registry: list[str] | None = None) -> list[str]:
    if registry is None:
        registry = []
    registry.append(name)
    return registry
"""
    test = f"""\"\"\"Test for listener registry module {salt}.\"\"\"
from {mod} import register_handler


def test_handler_registry_isolation():
    register_handler("on_start")
    second = register_handler("on_stop")
    assert second == ["on_stop"]
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Prevent handler pollution across registrations via mutable default argument",
    )


# ---------------------------------------------------------------------------
# 7. SQL_INJECTION (8 templates: 2 eval, 6 train)
# ---------------------------------------------------------------------------


# Template 1 (eval): audit_query
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


# Template 2 (eval): login_query
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


# Template 3 (train): post_search_query
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


# Template 4 (train): order_lookup_query
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


# Template 5 (train): product_category_query
def gen_sql_injection_product_category(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"prod_filter_{salt}"
    buggy = f"""\"\"\"Product category query module {salt}.\"\"\"


def build_category_filter_query(category: str) -> tuple[str, tuple]:
    # String formatting SQL injection
    query = f"SELECT id, name, price FROM products WHERE category = '{{category}}'"
    return query, ()
"""
    fixed = f"""\"\"\"Product category query module {salt}.\"\"\"


def build_category_filter_query(category: str) -> tuple[str, tuple]:
    query = "SELECT id, name, price FROM products WHERE category = ?"
    return query, (category,)
"""
    test = f"""\"\"\"Test for product category query module {salt}.\"\"\"
from {mod} import build_category_filter_query


def test_parameterized_category_query():
    cat = "tools' OR 'x'='x"
    query, params = build_category_filter_query(cat)
    assert "?" in query
    assert cat not in query
    assert params == (cat,)
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Use parameter binding for product category SQL filtering",
    )


# Template 6 (train): user_email_query
def gen_sql_injection_user_email(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"email_query_{salt}"
    buggy = f"""\"\"\"User email query module {salt}.\"\"\"


def build_email_lookup_query(email: str) -> tuple[str, tuple]:
    # Vulnerable email lookup formatting
    query = f"SELECT user_id, active FROM users WHERE email = '{{email}}'"
    return query, ()
"""
    fixed = f"""\"\"\"User email query module {salt}.\"\"\"


def build_email_lookup_query(email: str) -> tuple[str, tuple]:
    query = "SELECT user_id, active FROM users WHERE email = ?"
    return query, (email,)
"""
    test = f"""\"\"\"Test for user email query module {salt}.\"\"\"
from {mod} import build_email_lookup_query


def test_parameterized_email_query():
    raw_email = "test@example.com' OR 1=1--"
    query, params = build_email_lookup_query(raw_email)
    assert "?" in query
    assert raw_email not in query
    assert params == (raw_email,)
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Parameterize email lookup query to prevent SQL authentication bypass",
    )


# Template 7 (train): transaction_ref_query
def gen_sql_injection_transaction_ref(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"txn_ref_{salt}"
    buggy = f"""\"\"\"Transaction reference query module {salt}.\"\"\"


def build_txn_query(reference: str) -> tuple[str, tuple]:
    # String format injection vulnerability
    query = f"SELECT id, amount, status FROM transactions WHERE ref = '{{reference}}'"
    return query, ()
"""
    fixed = f"""\"\"\"Transaction reference query module {salt}.\"\"\"


def build_txn_query(reference: str) -> tuple[str, tuple]:
    query = "SELECT id, amount, status FROM transactions WHERE ref = ?"
    return query, (reference,)
"""
    test = f"""\"\"\"Test for transaction reference query module {salt}.\"\"\"
from {mod} import build_txn_query


def test_parameterized_txn_query():
    ref_code = "TX-999' UNION SELECT 1, 2, 'paid'--"
    query, params = build_txn_query(ref_code)
    assert "?" in query
    assert ref_code not in query
    assert params == (ref_code,)
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Use parameterized query for financial transaction reference lookups",
    )


# Template 8 (train): metric_name_query
def gen_sql_injection_metric_name(seed: int) -> GeneratorResult:
    rng = random.Random(seed)
    salt = rng.randint(100, 999)
    mod = f"metric_sql_{salt}"
    buggy = f"""\"\"\"Metric SQL query module {salt}.\"\"\"


def build_metric_query(metric_name: str) -> tuple[str, tuple]:
    # Vulnerable metric query
    query = f"SELECT timestamp, value FROM metric_events WHERE name = '{{metric_name}}'"
    return query, ()
"""
    fixed = f"""\"\"\"Metric SQL query module {salt}.\"\"\"


def build_metric_query(metric_name: str) -> tuple[str, tuple]:
    query = "SELECT timestamp, value FROM metric_events WHERE name = ?"
    return query, (metric_name,)
"""
    test = f"""\"\"\"Test for metric SQL query module {salt}.\"\"\"
from {mod} import build_metric_query


def test_parameterized_metric_query():
    m_name = "cpu_usage'; DROP TABLE metrics;--"
    query, params = build_metric_query(m_name)
    assert "?" in query
    assert m_name not in query
    assert params == (m_name,)
"""
    return GeneratorResult(
        module_name=mod,
        buggy_code=buggy,
        fixed_code=fixed,
        test_code=test,
        description="Parameterize telemetry metric event lookup query",
    )


# ---------------------------------------------------------------------------
# Template Partitioning: exactly 25% (2) for seeded_eval, 75% (6) for seeded_train
# ---------------------------------------------------------------------------

EVAL_TEMPLATES_PER_CLASS: dict[BugTaxonomy, list[str]] = {
    BugTaxonomy.OFF_BY_ONE: ["pagination_offset", "tier_threshold"],
    BugTaxonomy.WRONG_OPERATOR: ["reorder_threshold", "boolean_permission"],
    BugTaxonomy.NONE_HANDLING: ["profile_display_name", "cache_subscript"],
    BugTaxonomy.MISSING_AWAIT: ["payment_authorize", "alert_broadcast"],
    BugTaxonomy.RESOURCE_LEAK: ["export_channel", "data_stream_reader"],
    BugTaxonomy.BAD_DEFAULT_ARG: ["item_tags", "order_items"],
    BugTaxonomy.SQL_INJECTION: ["audit_query", "login_query"],
}

TRAIN_TEMPLATES_PER_CLASS: dict[BugTaxonomy, list[str]] = {
    BugTaxonomy.OFF_BY_ONE: [
        "slice_limit",
        "sliding_window",
        "binary_search",
        "chunk_partition",
        "inclusive_range",
        "buffer_stride",
    ],
    BugTaxonomy.WRONG_OPERATOR: [
        "voucher_reduction",
        "credit_limit_check",
        "role_membership",
        "surcharge_math",
        "retry_budget",
        "bitmask_flag",
    ],
    BugTaxonomy.NONE_HANDLING: [
        "pipeline_normalize",
        "comment_author",
        "config_timeout",
        "header_split",
        "list_element_none_filter",
        "optional_callback",
    ],
    BugTaxonomy.MISSING_AWAIT: [
        "pipeline_batch_fetch",
        "audit_entry_write",
        "http_client_get",
        "cache_async_lookup",
        "queue_pop_payload",
        "db_async_query",
    ],
    BugTaxonomy.RESOURCE_LEAK: [
        "audit_db_conn",
        "socket_transport",
        "file_handle_writer",
        "temp_buffer_cleanup",
        "lock_release",
        "session_pool_checkout",
    ],
    BugTaxonomy.BAD_DEFAULT_ARG: [
        "cache_metadata",
        "user_roles",
        "metrics_buffer",
        "headers_dict",
        "category_filter_list",
        "listener_registry",
    ],
    BugTaxonomy.SQL_INJECTION: [
        "post_search_query",
        "order_lookup_query",
        "product_category_query",
        "user_email_query",
        "transaction_ref_query",
        "metric_name_query",
    ],
}

# Registry mapping: (Taxonomy, Template) -> GeneratorFunc
TAXONOMY_TEMPLATE_GENERATORS: dict[tuple[BugTaxonomy, str], GeneratorFunc] = {
    # OFF_BY_ONE (2 eval, 6 train)
    (BugTaxonomy.OFF_BY_ONE, "pagination_offset"): gen_off_by_one_pagination_offset,
    (BugTaxonomy.OFF_BY_ONE, "tier_threshold"): gen_off_by_one_tier_threshold,
    (BugTaxonomy.OFF_BY_ONE, "slice_limit"): gen_off_by_one_slice_limit,
    (BugTaxonomy.OFF_BY_ONE, "sliding_window"): gen_off_by_one_sliding_window,
    (BugTaxonomy.OFF_BY_ONE, "binary_search"): gen_off_by_one_binary_search,
    (BugTaxonomy.OFF_BY_ONE, "chunk_partition"): gen_off_by_one_chunk_partition,
    (BugTaxonomy.OFF_BY_ONE, "inclusive_range"): gen_off_by_one_inclusive_range,
    (BugTaxonomy.OFF_BY_ONE, "buffer_stride"): gen_off_by_one_buffer_stride,
    # WRONG_OPERATOR (2 eval, 6 train)
    (BugTaxonomy.WRONG_OPERATOR, "reorder_threshold"): gen_wrong_operator_reorder_threshold,
    (BugTaxonomy.WRONG_OPERATOR, "boolean_permission"): gen_wrong_operator_boolean_permission,
    (BugTaxonomy.WRONG_OPERATOR, "voucher_reduction"): gen_wrong_operator_voucher_reduction,
    (BugTaxonomy.WRONG_OPERATOR, "credit_limit_check"): gen_wrong_operator_credit_limit,
    (BugTaxonomy.WRONG_OPERATOR, "role_membership"): gen_wrong_operator_role_membership,
    (BugTaxonomy.WRONG_OPERATOR, "surcharge_math"): gen_wrong_operator_surcharge,
    (BugTaxonomy.WRONG_OPERATOR, "retry_budget"): gen_wrong_operator_retry_budget,
    (BugTaxonomy.WRONG_OPERATOR, "bitmask_flag"): gen_wrong_operator_bitmask,
    # NONE_HANDLING (2 eval, 6 train)
    (BugTaxonomy.NONE_HANDLING, "profile_display_name"): gen_none_handling_profile_display,
    (BugTaxonomy.NONE_HANDLING, "cache_subscript"): gen_none_handling_cache_subscript,
    (BugTaxonomy.NONE_HANDLING, "pipeline_normalize"): gen_none_handling_pipeline_normalize,
    (BugTaxonomy.NONE_HANDLING, "comment_author"): gen_none_handling_comment_author,
    (BugTaxonomy.NONE_HANDLING, "config_timeout"): gen_none_handling_config_timeout,
    (BugTaxonomy.NONE_HANDLING, "header_split"): gen_none_handling_header_split,
    (BugTaxonomy.NONE_HANDLING, "list_element_none_filter"): gen_none_handling_list_filter,
    (BugTaxonomy.NONE_HANDLING, "optional_callback"): gen_none_handling_optional_callback,
    # MISSING_AWAIT (2 eval, 6 train)
    (BugTaxonomy.MISSING_AWAIT, "payment_authorize"): gen_missing_await_payment,
    (BugTaxonomy.MISSING_AWAIT, "alert_broadcast"): gen_missing_await_notification,
    (BugTaxonomy.MISSING_AWAIT, "pipeline_batch_fetch"): gen_missing_await_pipeline,
    (BugTaxonomy.MISSING_AWAIT, "audit_entry_write"): gen_missing_await_audit,
    (BugTaxonomy.MISSING_AWAIT, "http_client_get"): gen_missing_await_http_client,
    (BugTaxonomy.MISSING_AWAIT, "cache_async_lookup"): gen_missing_await_cache_async,
    (BugTaxonomy.MISSING_AWAIT, "queue_pop_payload"): gen_missing_await_queue_pop,
    (BugTaxonomy.MISSING_AWAIT, "db_async_query"): gen_missing_await_db_query,
    # RESOURCE_LEAK (2 eval, 6 train)
    (BugTaxonomy.RESOURCE_LEAK, "export_channel"): gen_resource_leak_export_channel,
    (BugTaxonomy.RESOURCE_LEAK, "data_stream_reader"): gen_resource_leak_stream_reader,
    (BugTaxonomy.RESOURCE_LEAK, "audit_db_conn"): gen_resource_leak_audit_conn,
    (BugTaxonomy.RESOURCE_LEAK, "socket_transport"): gen_resource_leak_socket_transport,
    (BugTaxonomy.RESOURCE_LEAK, "file_handle_writer"): gen_resource_leak_file_handle,
    (BugTaxonomy.RESOURCE_LEAK, "temp_buffer_cleanup"): gen_resource_leak_temp_buffer,
    (BugTaxonomy.RESOURCE_LEAK, "lock_release"): gen_resource_leak_lock_release,
    (BugTaxonomy.RESOURCE_LEAK, "session_pool_checkout"): gen_resource_leak_session_checkout,
    # BAD_DEFAULT_ARG (2 eval, 6 train)
    (BugTaxonomy.BAD_DEFAULT_ARG, "item_tags"): gen_bad_default_inventory,
    (BugTaxonomy.BAD_DEFAULT_ARG, "order_items"): gen_bad_default_pricing,
    (BugTaxonomy.BAD_DEFAULT_ARG, "cache_metadata"): gen_bad_default_cache,
    (BugTaxonomy.BAD_DEFAULT_ARG, "user_roles"): gen_bad_default_auth,
    (BugTaxonomy.BAD_DEFAULT_ARG, "metrics_buffer"): gen_bad_default_metrics_buffer,
    (BugTaxonomy.BAD_DEFAULT_ARG, "headers_dict"): gen_bad_default_headers_dict,
    (BugTaxonomy.BAD_DEFAULT_ARG, "category_filter_list"): gen_bad_default_category_filter,
    (BugTaxonomy.BAD_DEFAULT_ARG, "listener_registry"): gen_bad_default_listeners,
    # SQL_INJECTION (2 eval, 6 train)
    (BugTaxonomy.SQL_INJECTION, "audit_query"): gen_sql_injection_audit,
    (BugTaxonomy.SQL_INJECTION, "login_query"): gen_sql_injection_auth,
    (BugTaxonomy.SQL_INJECTION, "post_search_query"): gen_sql_injection_content,
    (BugTaxonomy.SQL_INJECTION, "order_lookup_query"): gen_sql_injection_pricing,
    (BugTaxonomy.SQL_INJECTION, "product_category_query"): gen_sql_injection_product_category,
    (BugTaxonomy.SQL_INJECTION, "user_email_query"): gen_sql_injection_user_email,
    (BugTaxonomy.SQL_INJECTION, "transaction_ref_query"): gen_sql_injection_transaction_ref,
    (BugTaxonomy.SQL_INJECTION, "metric_name_query"): gen_sql_injection_metric_name,
}

ALL_BASE_TEMPLATES = sorted({tpl for (_, tpl) in TAXONOMY_TEMPLATE_GENERATORS})
BASE_TEMPLATES = ALL_BASE_TEMPLATES


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
    """Return dictionary of taxonomy class -> list of all unique templates."""
    distribution: dict[str, list[str]] = {}
    for tax, tpl in sorted(TAXONOMY_TEMPLATE_GENERATORS.keys(), key=lambda x: (x[0].value, x[1])):
        distribution.setdefault(tax.value, []).append(tpl)
    return distribution


def get_eval_templates_per_class() -> dict[str, list[str]]:
    """Return dictionary of taxonomy class -> eval-only reserved templates (25%)."""
    return {tax.value: list(tpls) for tax, tpls in EVAL_TEMPLATES_PER_CLASS.items()}


def get_train_templates_per_class() -> dict[str, list[str]]:
    """Return dictionary of taxonomy class -> train-only templates (75%)."""
    return {tax.value: list(tpls) for tax, tpls in TRAIN_TEMPLATES_PER_CLASS.items()}


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
