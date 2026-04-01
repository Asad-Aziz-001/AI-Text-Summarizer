import streamlit as st
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# ---- PAGE CONFIG ----
st.set_page_config(page_title="AI Text Summarizer", layout="centered")

# ---- CUSTOM CSS ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root & Reset ── */
:root {
    --bg:        #0a0a0f;
    --surface:   #13131a;
    --border:    #2a2a3a;
    --accent:    #7fffb2;
    --accent2:   #3af0ff;
    --text:      #e8e8f0;
    --muted:     #7070a0;
    --radius:    16px;
    --glow:      0 0 40px rgba(127,255,178,0.12);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 1.5rem 4rem !important; max-width: 760px !important; }

/* ── Animated background grain ── */
body::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(ellipse 80% 60% at 20% -10%, rgba(127,255,178,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 110%, rgba(58,240,255,0.06) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* ── Badge chip ── */
.badge {
    display: inline-block;
    background: rgba(127,255,178,0.10);
    border: 1px solid rgba(127,255,178,0.25);
    color: var(--accent);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}

/* ── Hero title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 6vw, 3.4rem);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 30%, var(--accent) 80%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.6rem;
}

.hero-sub {
    font-size: 1rem;
    color: var(--muted);
    font-weight: 300;
    margin-bottom: 2.4rem;
    line-height: 1.6;
}

/* ── Divider ── */
.divider {
    width: 48px;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 2px;
    margin: 0 0 2rem;
}

/* ── Card wrapper ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.8rem;
    margin-bottom: 1.2rem;
    position: relative;
    box-shadow: 0 4px 32px rgba(0,0,0,0.35);
    transition: border-color 0.3s;
}
.card:hover { border-color: rgba(127,255,178,0.3); }

.card-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.75rem;
}

/* ── Textarea override ── */
textarea {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    resize: vertical !important;
    transition: border-color 0.25s !important;
}
textarea:focus {
    border-color: rgba(127,255,178,0.45) !important;
    box-shadow: 0 0 0 3px rgba(127,255,178,0.07) !important;
    outline: none !important;
}
textarea::placeholder { color: var(--muted) !important; opacity: 0.6 !important; }

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
    color: #050a08 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 4px 24px rgba(127,255,178,0.22) !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(127,255,178,0.32) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Summary result box ── */
.result-box {
    background: linear-gradient(135deg, rgba(127,255,178,0.06) 0%, rgba(58,240,255,0.04) 100%);
    border: 1px solid rgba(127,255,178,0.2);
    border-radius: var(--radius);
    padding: 1.6rem 1.8rem;
    margin-top: 0.4rem;
    position: relative;
    overflow: hidden;
}
.result-box::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.8rem;
}
.result-text {
    font-size: 1.05rem;
    line-height: 1.75;
    color: var(--text);
    font-weight: 300;
}

/* ── Warning ── */
.stAlert {
    background: rgba(255,200,80,0.06) !important;
    border: 1px solid rgba(255,200,80,0.2) !important;
    border-radius: 10px !important;
    color: #ffd080 !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Char counter ── */
.char-count {
    font-size: 0.75rem;
    color: var(--muted);
    text-align: right;
    margin-top: -0.4rem;
    margin-bottom: 1rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 3rem;
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.05em;
}
.footer span { color: var(--accent); }
</style>
""", unsafe_allow_html=True)


# ---- LOAD MODEL ----
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="t5-small",
        device=-1
    )

summarizer = load_model()


# ---- HERO SECTION ----
st.markdown('<div class="badge">✦ Powered by T5 · Transformers</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">AI Text<br>Summarizer</h1>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Paste any article, paragraph, or document below.<br>Get a crisp, intelligent 2–3 line summary instantly.</p>', unsafe_allow_html=True)


# ---- INPUT CARD ----
st.markdown('<div class="card"><div class="card-label">Your Text</div>', unsafe_allow_html=True)

input_text = st.text_area(
    label="",
    height=220,
    placeholder="Paste your paragraph or article here...",
    label_visibility="collapsed"
)

char_count = len(input_text)
st.markdown(f'<div class="char-count">{char_count} characters</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ---- BUTTON ----
summarize_clicked = st.button("⚡ Generate Summary")


# ---- RESULT ----
if summarize_clicked:
    if input_text.strip():
        with st.spinner("Thinking..."):
            result = summarizer(
                input_text,
                max_length=60,
                min_length=20,
                do_sample=False
            )
        summary_text = result[0]['summary_text']
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">✦ Summary</div>
            <div class="result-text">{summary_text}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️  Please paste some text before summarizing.")


# ---- FOOTER ----
st.markdown('<div class="footer">Built by <span>ASAD AZIZ</span> · Model: T5-Small · Running on CPU</div>', unsafe_allow_html=True)
