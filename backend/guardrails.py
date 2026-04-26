import re
from dataclasses import dataclass

# ── What a guardrail check returns ───────────────────────────────
@dataclass
class GuardrailResult:
    passed: bool          # True = safe to continue
    reason: str = ""      # Human-readable explanation if blocked
    cleaned_text: str = ""  # Sanitised version of the text (used in output guardrail)

# ── PII regex patterns ────────────────────────────────────────────
# Each entry is (label, compiled_regex)
PII_PATTERNS = [
    ("email address",    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", re.I)),
    ("phone number",     re.compile(r"\b(\+?\d[\d\s\-().]{7,14}\d)\b")),
    ("Aadhaar number",   re.compile(r"\b[2-9]{1}\d{3}\s?\d{4}\s?\d{4}\b")),   # India
    ("PAN number",       re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),            # India
    ("credit card",      re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("date of birth",    re.compile(r"\b(0?[1-9]|[12]\d|3[01])[\/\-](0?[1-9]|1[0-2])[\/\-](\d{2}|\d{4})\b")),
    ("salary figure",    re.compile(r"(salary|ctc|package)\s*:?\s*₹?\$?\d[\d,\.]+", re.I)),
]

# ── Allowed topic keywords (scope check) ─────────────────────────
# If a query contains NONE of these broad topics, it's likely out of scope.
# Keep this list fairly generous — it's a coarse filter.
IN_SCOPE_KEYWORDS = [
    # business operations
    "revenue", "sales", "growth", "profit", "expense", "budget", "cost",
    "forecast", "report", "quarter", "financial", "invoice", "reimbursement",
    # people / hr
    "employee", "salary", "payroll", "leave", "attendance", "performance",
    "hire", "onboard", "policy", "hr", "team", "department",
    # marketing
    "campaign", "customer", "feedback", "metric", "conversion", "leads",
    "marketing", "brand", "engagement", "channel",
    # engineering / tech
    "architecture", "system", "deployment", "api", "database", "server",
    "code", "technical", "infrastructure", "process", "workflow",
    # general
    "company", "event", "announcement", "faq", "guideline", "procedure",
    "meeting", "project", "update", "status", "who", "what", "when",
    "how", "why", "explain", "summarise", "summarize", "list", "show",
]

def check_input(query: str) -> GuardrailResult:
    """
    Run two checks on the raw user query:
    1. Does it contain PII the user accidentally typed?
    2. Is it related to company business at all?
    """
    query_lower = query.lower()

    # ── Check 1: PII in the query ─────────────────────────────────
    for label, pattern in PII_PATTERNS:
        if pattern.search(query):
            return GuardrailResult(
                passed=False,
                reason=(
                    f"Your message appears to contain a {label}. "
                    f"Please remove personal information before sending."
                ),
            )

    # ── Check 2: Out-of-scope ─────────────────────────────────────
    # Short messages (greetings, clarifications) are always allowed
    if len(query.split()) <= 5:
        return GuardrailResult(passed=True)

    has_topic = any(kw in query_lower for kw in IN_SCOPE_KEYWORDS)
    if not has_topic:
        return GuardrailResult(
            passed=False,
            reason=(
                "I can only help with company-related questions — "
                "finance, HR, marketing, engineering, or general policies. "
                "Your question seems outside that scope."
            ),
        )

    return GuardrailResult(passed=True)

# ── Redaction placeholder ─────────────────────────────────────────
REDACT_PLACEHOLDER = "[redacted]"

def check_output(answer: str) -> GuardrailResult:
    """
    Scan the LLM's answer for PII that leaked from source documents.
    Instead of blocking entirely, we redact and return the cleaned text.
    """
    cleaned = answer
    found_labels = []

    for label, pattern in PII_PATTERNS:
        if pattern.search(cleaned):
            cleaned = pattern.sub(REDACT_PLACEHOLDER, cleaned)
            found_labels.append(label)

    if found_labels:
        unique = list(dict.fromkeys(found_labels))   # preserve order, remove duplicates
        return GuardrailResult(
            passed=False,                             # "failed" = something was found
            reason=f"Redacted: {', '.join(unique)}",
            cleaned_text=cleaned,
        )

    return GuardrailResult(passed=True, cleaned_text=answer)
