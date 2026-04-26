import json
import os
from datetime import datetime, date
from dataclasses import dataclass, asdict
from pathlib import Path

# ── Groq pricing per million tokens ──────────────────────────────
# Update these if Groq changes their pricing
PRICING = {
    "llama3-70b-8192": {
        "input":  0.59 / 1_000_000,   # $ per token
        "output": 0.79 / 1_000_000,
    },
    "default": {
        "input":  0.59 / 1_000_000,
        "output": 0.79 / 1_000_000,
    }
}

# ── Alert thresholds ──────────────────────────────────────────────
DAILY_BUDGET_USD    = 1.00    # alert if total daily spend exceeds $1
PER_USER_BUDGET_USD = 0.10    # alert if single user spends over $0.10/day

# ── Where to store usage logs ─────────────────────────────────────
LOG_FILE = Path(__file__).parent / "usage_logs.json"

@dataclass
class UsageRecord:
    timestamp:     str
    username:      str
    role:          str
    query:         str
    input_tokens:  int
    output_tokens: int
    total_tokens:  int
    cost_usd:      float
    model:         str
    blocked:       bool

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a given token usage."""
    rates = PRICING.get(model, PRICING["default"])
    return round(
        (input_tokens * rates["input"]) + (output_tokens * rates["output"]),
        6
    )

def log_usage(
    username:      str,
    role:          str,
    query:         str,
    input_tokens:  int,
    output_tokens: int,
    model:         str,
    blocked:       bool = False,
) -> UsageRecord:
    """Create a usage record and append it to the log file."""
    cost = calculate_cost(model, input_tokens, output_tokens)

    record = UsageRecord(
        timestamp     = datetime.utcnow().isoformat(),
        username      = username,
        role          = role,
        query         = query[:100],   # truncate long queries
        input_tokens  = input_tokens,
        output_tokens = output_tokens,
        total_tokens  = input_tokens + output_tokens,
        cost_usd      = cost,
        model         = model,
        blocked       = blocked,
    )

    # Load existing logs
    logs = []
    if LOG_FILE.exists():
        with open(LOG_FILE, "r") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []

    # Append new record
    logs.append(asdict(record))

    # Save back
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

    return record

def get_today_logs() -> list[dict]:
    """Return all usage records from today."""
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE, "r") as f:
        try:
            logs = json.load(f)
        except json.JSONDecodeError:
            return []
    today = date.today().isoformat()
    return [r for r in logs if r["timestamp"].startswith(today)]

def check_alerts(username: str) -> list[str]:
    """
    Check if any budget thresholds have been exceeded.
    Returns a list of alert messages (empty = no alerts).
    """
    alerts = []
    today_logs = get_today_logs()

    if not today_logs:
        return alerts

    # Check total daily spend
    daily_total = sum(r["cost_usd"] for r in today_logs)
    if daily_total >= DAILY_BUDGET_USD:
        alerts.append(
            f"DAILY BUDGET ALERT: Total spend today is "
            f"${daily_total:.4f} — exceeds ${DAILY_BUDGET_USD:.2f} limit"
        )

    # Check per-user spend
    user_logs   = [r for r in today_logs if r["username"] == username]
    user_total  = sum(r["cost_usd"] for r in user_logs)
    if user_total >= PER_USER_BUDGET_USD:
        alerts.append(
            f"USER BUDGET ALERT: {username} has spent "
            f"${user_total:.4f} today — exceeds ${PER_USER_BUDGET_USD:.2f} limit"
        )

    return alerts

def get_summary() -> dict:
    """Return a summary of all usage for the dashboard."""
    if not LOG_FILE.exists():
        return {}

    with open(LOG_FILE, "r") as f:
        try:
            logs = json.load(f)
        except json.JSONDecodeError:
            return {}

    if not logs:
        return {}

    # Per user totals
    user_totals = {}
    for r in logs:
        u = r["username"]
        if u not in user_totals:
            user_totals[u] = {"tokens": 0, "cost": 0.0, "requests": 0}
        user_totals[u]["tokens"]   += r["total_tokens"]
        user_totals[u]["cost"]     += r["cost_usd"]
        user_totals[u]["requests"] += 1

    # Per role totals
    role_totals = {}
    for r in logs:
        role = r["role"]
        if role not in role_totals:
            role_totals[role] = {"tokens": 0, "cost": 0.0, "requests": 0}
        role_totals[role]["tokens"]   += r["total_tokens"]
        role_totals[role]["cost"]     += r["cost_usd"]
        role_totals[role]["requests"] += 1

    # Daily totals
    daily_totals = {}
    for r in logs:
        day = r["timestamp"][:10]
        if day not in daily_totals:
            daily_totals[day] = {"tokens": 0, "cost": 0.0, "requests": 0}
        daily_totals[day]["tokens"]   += r["total_tokens"]
        daily_totals[day]["cost"]     += r["cost_usd"]
        daily_totals[day]["requests"] += 1

    return {
        "total_requests": len(logs),
        "total_tokens":   sum(r["total_tokens"] for r in logs),
        "total_cost_usd": round(sum(r["cost_usd"] for r in logs), 6),
        "by_user":        user_totals,
        "by_role":        role_totals,
        "by_day":         daily_totals,
    }