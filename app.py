import streamlit as st
import requests

# ── FastAPI base URL ──────────────────────────────────────────────
API_URL = "http://localhost:8000"

# ── Page config (must be the FIRST Streamlit call) ───────────────
st.set_page_config(
    page_title="Company Chatbot",
    page_icon="💬",
    layout="centered",
)

# ── Session state initialisation ─────────────────────────────────
# These keys are created once; on every rerun they already exist.
if "logged_in" not in st.session_state:
    st.session_state.logged_in  = False
if "username" not in st.session_state:
    st.session_state.username   = ""
if "role" not in st.session_state:
    st.session_state.role       = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role": "user"|"assistant", "content": ..., "sources": [...]}

# ── Role badge colours ────────────────────────────────────────────
ROLE_COLOURS = {
    "finance":     "#1D9E75",   # teal
    "marketing":   "#7F77DD",   # purple
    "hr":          "#D85A30",   # coral
    "engineering": "#378ADD",   # blue
    "c-level":     "#BA7517",   # amber
    "employee":    "#888780",   # gray
}

def get_password() -> str:
    """Return the password from session state (stored only during this session)."""
    return st.session_state.get("password", "")

def show_login():
    st.title("💬 Company Chatbot")
    st.write("Please log in to continue.")

    # st.form groups widgets so the whole form submits together
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        # Call FastAPI /login with HTTP Basic Auth
        response = requests.get(
            f"{API_URL}/login",
            auth=(username, password),   # requests handles the Basic Auth header
        )

        if response.status_code == 200:
            data = response.json()
            # Write into session_state so the next rerun knows we're logged in
            st.session_state.logged_in = True
            st.session_state.password = password 
            st.session_state.username  = username
            st.session_state.role      = data["role"]
            st.rerun()   # force an immediate rerun so the chat screen shows up
        else:
            st.error("❌ Invalid username or password.")

def show_chat():
    # ── Header with user info ─────────────────────────────────────
    role_colour = ROLE_COLOURS.get(st.session_state.role, "#888780")

    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("💬 Company Chatbot")
    with col2:
        # Logout button aligned to the right
        if st.button("Log out"):
            # Clear all session state and return to login screen
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Role badge — built with markdown + inline HTML colour
    st.markdown(
        f"Logged in as **{st.session_state.username}** &nbsp;"
        f"<span style='background:{role_colour};color:white;"
        f"padding:2px 10px;border-radius:12px;font-size:13px;'>"
        f"{st.session_state.role}</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Replay chat history ───────────────────────────────────────
    # st.chat_message creates a styled bubble (user or assistant)
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            # Show sources only on assistant messages
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📄 Sources"):
                    for src in msg["sources"]:
                        st.write(f"• {src}")

    # ── Input box at the bottom ───────────────────────────────────
    # st.chat_input pins a text box to the bottom of the page
    user_input = st.chat_input("Ask me anything...")

    if user_input:
        # 1. Show the user's message immediately
        with st.chat_message("user"):
            st.write(user_input)

        # 2. Save it to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "sources": [],
        })

        # 3. Call FastAPI /chat
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/chat",
                auth=(st.session_state.username, get_password()),
                json={"message": user_input},
            )

        if response.status_code == 200:
            data    = response.json()
            answer  = data["answer"]
            sources = data.get("sources", [])
            blocked  = data.get("blocked", False)
            redacted = data.get("redacted", False)
        else:
            answer  = "⚠️ Something went wrong. Please try again."
            sources = []
            blocked = False
            redacted = False
        
        # 4. Show the assistant reply
        with st.chat_message("assistant"):
            if blocked:
                # Guardrail blocked the query — show as a warning, not an error
                st.warning(answer)
            else:
                st.write(answer)
                if redacted:
                    st.caption("⚠️ Some sensitive information was redacted from this response.")
                if sources:
                    with st.expander("📄 Sources"):
                        for src in sources:
                            st.write(f"• {src}")

        # 5. Save assistant reply to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

# ── Router — decides which screen to show ────────────────────────
if st.session_state.logged_in:
    show_chat()
else:
    show_login()