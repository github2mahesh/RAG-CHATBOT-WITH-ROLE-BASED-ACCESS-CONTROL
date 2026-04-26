import pytest
import requests

# ── The running FastAPI server ────────────────────────────────────
# Locally: http://localhost:8000
# In GitHub Actions: http://localhost:8000 (we start the server in the pipeline)
BASE_URL = "http://localhost:8000"

# ── Test users ────────────────────────────────────────────────────
USERS = {
    "finance":     ("Sam",     "financepass"),
    "hr":          ("Natasha", "hrpass123"),
    "engineering": ("Tony",    "password123"),
    "marketing":   ("Bruce",   "securepass"),
    "c-level":     ("Nick",    "ceopass"),
    "employee":    ("Happy",   "emppass"),
}

# ─────────────────────────────────────────────────────────────────
# SECTION 1 — AUTHENTICATION EVALS
# ─────────────────────────────────────────────────────────────────

def test_valid_login_returns_200():
    """Correct credentials should return 200 and the user's role."""
    username, password = USERS["finance"]
    response = requests.get(f"{BASE_URL}/login", auth=(username, password))
    assert response.status_code == 200
    assert response.json()["role"] == "finance"

def test_wrong_password_returns_401():
    """Wrong password should be rejected."""
    response = requests.get(f"{BASE_URL}/login", auth=("Sam", "wrongpassword"))
    assert response.status_code == 401

def test_unknown_user_returns_401():
    """Unknown username should be rejected."""
    response = requests.get(f"{BASE_URL}/login", auth=("Ghost", "password123"))
    assert response.status_code == 401

def test_all_roles_can_login():
    """Every role in the system should be able to log in."""
    for role, (username, password) in USERS.items():
        response = requests.get(f"{BASE_URL}/login", auth=(username, password))
        assert response.status_code == 200, f"{role} login failed"
        assert response.json()["role"] == role

# ─────────────────────────────────────────────────────────────────
# SECTION 2 — RBAC EVALS
# ─────────────────────────────────────────────────────────────────

def _chat(username, password, message):
    """Helper — send a chat message and return the response JSON."""
    response = requests.post(
        f"{BASE_URL}/chat",
        auth=(username, password),
        json={"message": message, "history": []},
        timeout=30,
    )
    assert response.status_code == 200, f"Chat failed: {response.text}"
    return response.json()

def test_finance_user_cannot_access_hr():
    """Finance user asking HR question should be told no access."""
    data = _chat(*USERS["finance"], "Show me employee attendance records")
    answer_lower = data["answer"].lower()
    # LLM may say any of these — all mean correctly refused
    refused = (
        data["blocked"] is True or
        "don't have access" in answer_lower or
        "no access" in answer_lower or
        "i don't know" in answer_lower or      
        "i do not know" in answer_lower        
    )
    assert refused, f"Expected refusal but got: {data['answer']}"

def test_clevel_gets_answer_for_any_department():
    """C-level user should be able to ask about any department."""
    data = _chat(*USERS["c-level"], "What is the revenue?")
    assert data["blocked"] is False
    assert len(data["answer"]) > 0

def test_employee_limited_to_general():
    """Employee role should only see general data."""
    data = _chat(*USERS["employee"], "What are the company policies?")
    assert data["blocked"] is False

# ─────────────────────────────────────────────────────────────────
# SECTION 3 — GUARDRAIL EVALS
# ─────────────────────────────────────────────────────────────────

def test_email_in_query_is_blocked():
    """Query containing an email address should be blocked."""
    data = _chat(*USERS["finance"], "My email is test@gmail.com what is revenue?")
    assert data["blocked"] is True
    assert "email" in data["answer"].lower()

def test_pan_number_in_query_is_blocked():
    """Query containing a PAN number should be blocked."""
    data = _chat(*USERS["hr"], "What is salary for ABCDE1234F?")
    assert data["blocked"] is True

def test_out_of_scope_query_is_blocked():
    """Query unrelated to company business should be blocked."""
    data = _chat(*USERS["finance"], "What is the best pizza recipe in the world?")
    assert data["blocked"] is True

def test_greeting_is_not_blocked():
    """A simple greeting should never be blocked."""
    data = _chat(*USERS["finance"], "Hello")
    assert data["blocked"] is False

def test_short_query_is_not_blocked():
    """Short queries should pass the scope check."""
    data = _chat(*USERS["finance"], "Hi there")
    assert data["blocked"] is False

# ─────────────────────────────────────────────────────────────────
# SECTION 4 — RAG QUALITY EVALS
# ─────────────────────────────────────────────────────────────────

def test_answer_is_not_empty():
    """Every valid query should return a non-empty answer."""
    data = _chat(*USERS["finance"], "What is the revenue?")
    assert len(data["answer"].strip()) > 0

def test_sources_are_returned():
    """Valid queries should return source references."""
    data = _chat(*USERS["finance"], "What is the revenue?")
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0

def test_sources_match_user_department():
    """Sources returned should be from the user's department."""
    data = _chat(*USERS["finance"], "What is the revenue?")
    for source in data["sources"]:
        assert "finance" in source.lower(), f"Wrong dept in source: {source}"

def test_response_time_under_15_seconds():
    """The API should respond within 15 seconds."""
    import time
    start = time.time()
    _chat(*USERS["finance"], "What is the revenue?")
    elapsed = time.time() - start
    assert elapsed < 15, f"Response took {elapsed:.1f}s — too slow"

def test_empty_message_does_not_crash():
    """Sending an empty message should not crash the server."""
    response = requests.post(
        f"{BASE_URL}/chat",
        auth=(*USERS["finance"],),
        json={"message": "", "history": []},
        timeout=30,
    )
    # Should return 200 (handled) not 500 (crashed)
    assert response.status_code != 500