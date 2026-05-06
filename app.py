"""
AutoGuard AI — Streamlit Frontend Client
=========================================
Sends uploaded images to the FastAPI backend at http://127.0.0.1:8000/predict
and renders a professional Bento Grid results dashboard.

Run with:
    streamlit run app.py --server.port 8501
"""

import io
import requests
import streamlit as st
from PIL import Image, ImageOps

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoGuard AI — Car Damage Assessment",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# API endpoint
# ─────────────────────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000/predict"

# ─────────────────────────────────────────────────────────────────────────────
# Premium CSS injection
# ─────────────────────────────────────────────────────────────────────────────
PREMIUM_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Root reset & dark background ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: radial-gradient(ellipse at 20% 20%, #0d1b4b 0%, #050a1a 55%, #000510 100%) !important;
    font-family: 'Inter', sans-serif !important;
    color: #e2e8f0;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { visibility: hidden; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ── Reduce default top padding to bring content up ── */
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 0rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 4px; }

/* ── Glassmorphism card ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    padding: 28px 32px;
    margin-bottom: 20px;
    transition: all 0.3s ease;
}
.glass-card:hover {
    background: rgba(255,255,255,0.07);
    border-color: rgba(255,255,255,0.18);
}

/* ── Hero header ── */
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.18);
    border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 100px;
    margin-bottom: 14px;
}
.hero-sub {
    color: rgba(148,163,184,0.85);
    font-size: 1.05rem;
    font-weight: 400;
    margin-top: 8px;
    letter-spacing: 0.01em;
}

/* ── Prediction result ── */
.result-label {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1;
}
.result-meta {
    font-size: 0.88rem;
    color: rgba(148,163,184,0.8);
    margin-top: 4px;
}

/* ── Severity badge ── */
.severity-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    border-radius: 100px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 12px;
}

/* ── Stat tile (Bento Grid) ── */
.stat-tile {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 18px 22px;
    text-align: center;
}
.stat-value {
    font-size: 1.45rem;
    font-weight: 700;
    color: #f8fafc;
}
.stat-label {
    font-size: 0.75rem;
    color: rgba(148,163,184,0.7);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 2px;
}

/* ── Probability bar ── */
.prob-row { margin-bottom: 11px; }
.prob-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}
.prob-name { font-size: 0.82rem; font-weight: 500; color: #cbd5e1; }
.prob-pct  { font-size: 0.82rem; font-weight: 600; color: #94a3b8; }
.prob-track {
    height: 6px;
    background: rgba(255,255,255,0.07);
    border-radius: 100px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.6s ease;
}

/* ── Upload zone override ── */
[data-testid="stFileUploader"] > div {
    background: rgba(99,102,241,0.07) !important;
    border: 2px dashed rgba(99,102,241,0.40) !important;
    border-radius: 16px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: rgba(99,102,241,0.70) !important;
    background: rgba(99,102,241,0.12) !important;
}

/* ── Section heading ── */
.section-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(148,163,184,0.6);
    margin-bottom: 14px;
}

/* ── Idle state ── */
.idle-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 320px;
    text-align: center;
    gap: 12px;
}
.idle-icon { font-size: 3.5rem; opacity: 0.35; }
.idle-text  { color: rgba(148,163,184,0.5); font-size: 0.95rem; }

/* ── Footer strip ── */
.footer-strip {
    text-align: center;
    color: rgba(148,163,184,0.35);
    font-size: 0.72rem;
    padding-top: 32px;
    letter-spacing: 0.06em;
}
</style>
"""

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: probability bar renderer
# ─────────────────────────────────────────────────────────────────────────────
def _prob_bar_color(class_name: str) -> str:
    l = class_name.lower()
    if "crushed" in l:
        return "linear-gradient(90deg, #ef4444, #f97316)"
    elif "breakage" in l:
        return "linear-gradient(90deg, #f59e0b, #fbbf24)"
    else:
        return "linear-gradient(90deg, #22c55e, #34d399)"


def render_prob_bars(all_probs: dict) -> str:
    bars = ""
    for cls, pct in sorted(all_probs.items(), key=lambda x: -x[1]):
        color = _prob_bar_color(cls)
        bars += f"""
        <div class="prob-row">
            <div class="prob-label-row">
                <span class="prob-name">{cls.replace('_', ' ')}</span>
                <span class="prob-pct">{pct:.1f}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{pct}%;background:{color};"></div>
            </div>
        </div>"""
    return bars


# ─────────────────────────────────────────────────────────────────────────────
# Large centered title
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 18px 0 22px 0;">
    <div class="hero-badge">🛡️ &nbsp; AutoGuard AI &nbsp; v2.0</div>
    <h1 style="
        font-family: 'Inter', sans-serif;
        font-size: 2.55rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin: 8px 0 10px 0;
        background: linear-gradient(135deg, #818cf8 0%, #60a5fa 45%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: drop-shadow(0 0 18px rgba(96,165,250,0.55))
                drop-shadow(0 0 40px rgba(129,140,248,0.30));
    ">
        AutoGuard AI: Advanced Vehicle Damage Assessment
    </h1>
    <p class="hero-sub" style="text-align:center;">
        ResNet-50 &nbsp;·&nbsp; 6-Class Classifier &nbsp;·&nbsp; FastAPI-Powered Inference
    </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main Bento Grid
# ─────────────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.05, 1], gap="large")

# ═══════════════════════════════════════════════════════════════════
# LEFT: Upload + Image preview
# ═══════════════════════════════════════════════════════════════════
with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📁 &nbsp; Upload Vehicle Image</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        label="Drop image here",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
        key="uploader",
    )

    if uploaded:
        image = Image.open(uploaded)
        display_image = ImageOps.exif_transpose(image)
        st.image(display_image, use_column_width=True, caption="Uploaded vehicle image")
    else:
        st.markdown("""
        <div class="idle-card">
            <div class="idle-icon">🚗</div>
            <div class="idle-text">Upload a front or rear vehicle photo<br>to begin damage assessment</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# RIGHT: Results panel
# ═══════════════════════════════════════════════════════════════════
with right_col:

    if uploaded:
        # ── Send image to FastAPI backend ──────────────────────────
        with st.spinner("Analyzing damage via AutoGuard AI…"):
            try:
                uploaded.seek(0)
                response = requests.post(
                    API_URL,
                    files={"file": (uploaded.name, uploaded.read(), uploaded.type)},
                    timeout=30,
                )
                if response.status_code == 200:
                    result = response.json()
                elif response.status_code == 503:
                    st.error(
                        "🔴 **Backend unavailable:** Model file not found on server. "
                        "Ensure `model_resnetup.pth` is in the project directory and "
                        "restart the FastAPI server."
                    )
                    st.stop()
                else:
                    detail = response.json().get("detail", response.text)
                    st.error(f"🔴 **API Error {response.status_code}:** {detail}")
                    st.stop()
            except requests.exceptions.ConnectionError:
                st.error(
                    "🔴 **Cannot connect to the AutoGuard AI backend.**\n\n"
                    "Please start the FastAPI server first:\n"
                    "```\nuvicorn main:app --host 0.0.0.0 --port 8000 --reload\n```"
                )
                st.stop()
            except requests.exceptions.Timeout:
                st.error("🔴 **Request timed out.** The server took too long to respond.")
                st.stop()
            except Exception as exc:
                st.error(f"🔴 **Unexpected error:** {exc}")
                st.stop()

        # ── Parse result ──────────────────────────────────────────
        a        = result
        LOW_CONF = a["confidence"] < 50.0
        accent   = a["accent"]    if not LOW_CONF else "#94a3b8"
        glow     = a["glow"]      if not LOW_CONF else "rgba(148,163,184,0.20)"
        badge_bg = a["badge_bg"]  if not LOW_CONF else "rgba(148,163,184,0.12)"
        badge_txt= a["badge_text"]if not LOW_CONF else "#cbd5e1"

        # ── Low-confidence fallback (< 50%) ───────────────────────
        if LOW_CONF:
            st.warning(
                "**⚠️ Low Confidence — Result is Inconclusive**\n\n"
                "The model cannot definitively identify the damage type. "
                "Please upload a **clearer, well-lit image** of the affected area "
                "for a reliable assessment."
            )

        # ── Prediction card ────────────────────────────────────────
        display_label = (
            f"Inconclusive — {a['label'].replace('_', ' ')} (Preliminary)"
            if LOW_CONF else
            a['label'].replace('_', ' ')
        )
        result_icon   = "🔘" if LOW_CONF else a["icon"]
        severity_text = f"Preliminary · {a['severity']}" if LOW_CONF else f"{a['severity']} damage"
        label_size    = "1.55rem" if LOW_CONF else "2.2rem"

        st.markdown(f"""
        <div class="glass-card" style="border-color:{accent}55;
             box-shadow: 0 0 32px {glow};">
            <p class="section-title">🔍 &nbsp; Prediction Result</p>
            <p class="result-label" style="color:{accent};font-size:{label_size};">
                {display_label}
            </p>
            <p class="result-meta">Confidence &nbsp;<strong style="color:#f8fafc;">{a['confidence']:.1f}%</strong>
                &nbsp;|&nbsp; {result_icon} &nbsp;
                <span class="severity-badge"
                      style="background:{badge_bg};color:{badge_txt};border:1px solid {accent}55;">
                    {severity_text}
                </span>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Bento KPI strip ────────────────────────────────────────
        k1, k2, k3 = st.columns(3)
        k1.markdown(f"""
        <div class="stat-tile" style="border-color:{accent}44;">
            <div class="stat-value" style="color:{accent};">{a['icon']}</div>
            <div class="stat-label">Severity</div>
            <div style="font-size:0.82rem;font-weight:600;color:{badge_txt};margin-top:2px;">{a['severity']}</div>
        </div>""", unsafe_allow_html=True)

        k2.markdown(f"""
        <div class="stat-tile">
            <div class="stat-value">{a['confidence']:.1f}%</div>
            <div class="stat-label">Confidence</div>
        </div>""", unsafe_allow_html=True)

        k3.markdown(f"""
        <div class="stat-tile">
            <div class="stat-value" style="font-size:1.05rem;">{a['cost']}</div>
            <div class="stat-label">Est. Repair Cost</div>
        </div>""", unsafe_allow_html=True)

        # ── High-contrast repair cost disclaimer ──────────────────
        st.info(
            "📌 **Disclaimer:** Estimated repair costs are based on average industry "
            "standards and are for **informational purposes only**. Actual costs may vary "
            "significantly based on vehicle make, model, and local labor rates."
        )

        # ── Probability Dashboard ──────────────────────────────────
        st.markdown(f"""
        <div class="glass-card" style="margin-top:20px;">
            <p class="section-title">📊 &nbsp; Probability Dashboard — Model Reasoning</p>
            {render_prob_bars(a['all_probs'])}
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Idle state ─────────────────────────────────────────────
        st.markdown("""
        <div class="glass-card" style="min-height:480px;">
            <p class="section-title">🔍 &nbsp; Prediction Result</p>
            <div class="idle-card">
                <div class="idle-icon">⏳</div>
                <div class="idle-text">Results will appear here<br>after you upload an image</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-strip">
    AUTOGUARD AI &nbsp;·&nbsp; ResNet-50 Backbone &nbsp;·&nbsp;
    FastAPI Backend &nbsp;·&nbsp; 6-Class Vehicle Damage Classifier
</div>
""", unsafe_allow_html=True)
