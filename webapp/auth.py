"""
Email OTP authentication for the Signal Dashboard.

Flow:
  1. User enters their email address
  2. If email is on the allowed list → a 6-digit OTP is sent and the code
     input form is shown
  3. User enters the OTP within 10 minutes (3 attempts allowed)
  4. On success → st.session_state["authenticated"] = True for the session

Secrets required in .streamlit/secrets.toml (local) or Streamlit Cloud settings:

    SMTP_EMAIL    = "you@gmail.com"
    SMTP_PASSWORD = "xxxx xxxx xxxx xxxx"   # Gmail App Password
    SMTP_HOST     = "smtp.gmail.com"         # optional, default shown
    SMTP_PORT     = 587                      # optional, default shown
    ALLOWED_EMAILS = ["you@example.com", "colleague@example.com"]
"""

import random
import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import streamlit as st

OTP_EXPIRY_SECONDS = 600   # 10 minutes
MAX_ATTEMPTS = 3


# ── Helpers ────────────────────────────────────────────────────────────────────

def _generate_otp() -> str:
    return str(random.randint(100_000, 999_999))


def _send_otp_email(to_email: str, otp: str) -> bool:
    smtp_email    = st.secrets["SMTP_EMAIL"]
    smtp_password = st.secrets["SMTP_PASSWORD"]
    smtp_host     = st.secrets.get("SMTP_HOST", "smtp.gmail.com")
    smtp_port     = int(st.secrets.get("SMTP_PORT", 587))

    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;
                border:1px solid #e0e0e0;border-radius:8px;padding:32px">
      <h2 style="color:#1a1a2e;margin-top:0">📈 Signal Dashboard</h2>
      <p style="color:#555">Your one-time login code is:</p>
      <div style="font-size:40px;font-weight:bold;letter-spacing:10px;
                  color:#0066cc;text-align:center;padding:20px 0">
        {otp}
      </div>
      <p style="color:#888;font-size:13px">
        This code expires in <strong>10 minutes</strong>.<br>
        If you didn't request this, ignore this email.
      </p>
    </div>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"{otp} — Your Signal Dashboard Login Code"
    msg["From"]    = smtp_email
    msg["To"]      = to_email
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, to_email, msg.as_string())
        return True
    except Exception as exc:
        st.error(f"Failed to send email: {exc}")
        return False


def _allowed_emails() -> list[str]:
    raw = st.secrets.get("ALLOWED_EMAILS", [])
    return [e.strip().lower() for e in raw]


def _otp_expired() -> bool:
    sent_at = st.session_state.get("otp_sent_at", 0)
    return (time.time() - sent_at) > OTP_EXPIRY_SECONDS


def _reset_otp():
    for key in ("otp_code", "otp_email", "otp_sent_at", "otp_attempts", "otp_step"):
        st.session_state.pop(key, None)


# ── Public API ─────────────────────────────────────────────────────────────────

def is_authenticated() -> bool:
    return st.session_state.get("authenticated", False)


def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def login_page():
    """Render the full login UI. Call st.stop() after this if not authenticated."""

    st.markdown(
        "<h2 style='text-align:center;margin-top:60px'>📈 S&P 500 Signal Dashboard</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:#888'>Enter your email to receive a login code</p>",
        unsafe_allow_html=True,
    )

    col = st.columns([1, 1.4, 1])[1]   # centre the form

    step = st.session_state.get("otp_step", "email")

    # ── Step 1: email entry ────────────────────────────────────────────────────
    if step == "email":
        with col:
            with st.form("email_form"):
                email = st.text_input("Email address", placeholder="you@example.com")
                submitted = st.form_submit_button("Send code", use_container_width=True, type="primary")

            if submitted:
                email = email.strip().lower()
                if not email:
                    st.warning("Please enter your email address.")
                elif email not in _allowed_emails():
                    st.error("This email is not authorised. Contact the dashboard owner.")
                else:
                    otp = _generate_otp()
                    with st.spinner("Sending code…"):
                        ok = _send_otp_email(email, otp)
                    if ok:
                        st.session_state["otp_code"]     = otp
                        st.session_state["otp_email"]    = email
                        st.session_state["otp_sent_at"]  = time.time()
                        st.session_state["otp_attempts"] = 0
                        st.session_state["otp_step"]     = "otp"
                        st.rerun()

    # ── Step 2: OTP entry ──────────────────────────────────────────────────────
    elif step == "otp":
        email      = st.session_state["otp_email"]
        attempts   = st.session_state.get("otp_attempts", 0)
        remaining  = max(0, int(OTP_EXPIRY_SECONDS - (time.time() - st.session_state["otp_sent_at"])))
        mins, secs = divmod(remaining, 60)

        with col:
            st.info(f"Code sent to **{email}** · expires in {mins}m {secs}s")

            if _otp_expired():
                st.error("Code expired. Request a new one.")
                if st.button("Try again", use_container_width=True):
                    _reset_otp()
                    st.rerun()
                return

            with st.form("otp_form"):
                entered = st.text_input("6-digit code", max_chars=6, placeholder="123456")
                verify  = st.form_submit_button("Verify", use_container_width=True, type="primary")

            if verify:
                if entered.strip() == st.session_state["otp_code"]:
                    _reset_otp()
                    st.session_state["authenticated"] = True
                    st.session_state["auth_email"]    = email
                    st.rerun()
                else:
                    attempts += 1
                    st.session_state["otp_attempts"] = attempts
                    left = MAX_ATTEMPTS - attempts
                    if left > 0:
                        st.error(f"Incorrect code. {left} attempt{'s' if left > 1 else ''} remaining.")
                    else:
                        st.error("Too many incorrect attempts. Request a new code.")
                        _reset_otp()
                        st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("← Use a different email", use_container_width=True):
                _reset_otp()
                st.rerun()
