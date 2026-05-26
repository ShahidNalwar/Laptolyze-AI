import streamlit as st
import pandas as pd
import re
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

st.set_page_config(
    page_title="Laptolyze AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap" rel="stylesheet">

<style>
/* ── THEME-AGNOSTIC DESIGN SYSTEM ── */
:root {
    --accent:       #6366f1;
    --accent-light: #818cf8;
    --accent-dim:   rgba(99,102,241,0.12);
    --green:        #10b981;
    --green-dim:    rgba(16,185,129,0.1);
    --red:          #f43f5e;
    --red-dim:      rgba(244,63,94,0.1);
    --blue:         #3b82f6;
    --blue-dim:     rgba(59,130,246,0.1);
    --amber:        #f59e0b;

    --surface:      rgba(255,255,255,0.06);
    --surface-2:    rgba(255,255,255,0.10);
    --border:       rgba(255,255,255,0.10);
    --border-2:     rgba(255,255,255,0.16);
    --text-primary: #f1f5f9;
    --text-muted:   #94a3b8;
    --text-subtle:  #64748b;

    --radius:       14px;
    --radius-sm:    8px;
    --shadow:       0 4px 24px rgba(0,0,0,0.18);
    --shadow-sm:    0 2px 8px rgba(0,0,0,0.10);
}

/* Force dark background on Streamlit app */
.stApp {
    background: #0c111d !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #080c15 !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-2) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: var(--accent-light) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 500 !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricDelta"] { font-family: 'DM Mono', monospace !important; }
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 20px !important;
}

/* Button */
.stButton > button {
    background: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 14px 28px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
}
.stButton > button:hover {
    background: var(--accent-light) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Spinner */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* Info / Warning boxes */
.stAlert {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}

/* ── CUSTOM COMPONENTS ── */
.page-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin-bottom: 4px;
    line-height: 1.2;
}
.page-subtitle {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-bottom: 0;
}

.sidebar-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.sidebar-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--accent-dim);
    color: var(--accent-light);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.spec-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 8px;
    margin-top: 12px;
}
.spec-item {
    display: flex;
    align-items: center;
    gap: 10px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 10px 14px;
}
.spec-icon { font-size: 1rem; }
.spec-text {
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.3;
}
.spec-label {
    font-size: 0.68rem;
    color: var(--text-subtle);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    display: block;
}

/* Insight cards */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label.green { color: var(--green); }
.section-label.red   { color: var(--red); }
.section-label.blue  { color: var(--blue); }

.dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    display: inline-block;
}
.dot.green { background: var(--green); }
.dot.red   { background: var(--red); }
.dot.blue  { background: var(--blue); }

.insight-card {
    padding: 14px 16px;
    border-radius: var(--radius-sm);
    margin-bottom: 9px;
    font-size: 0.875rem;
    line-height: 1.55;
    position: relative;
    transition: transform 0.15s ease;
}
.insight-card:hover { transform: translateX(3px); }

.insight-card.pro {
    background: var(--green-dim);
    border: 1px solid rgba(16,185,129,0.2);
    border-left: 3px solid var(--green);
    color: #a7f3d0;
}
.insight-card.con {
    background: var(--red-dim);
    border: 1px solid rgba(244,63,94,0.2);
    border-left: 3px solid var(--red);
    color: #fecdd3;
}
.insight-card.neu {
    background: var(--blue-dim);
    border: 1px solid rgba(59,130,246,0.2);
    border-left: 3px solid var(--blue);
    color: #bfdbfe;
}

/* Summary bar */
.summary-bar-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-top: 8px;
}
.summary-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
}
.summary-row:last-child { margin-bottom: 0; }
.summary-name {
    width: 70px;
    font-size: 0.78rem;
    color: var(--text-muted);
    text-align: right;
    flex-shrink: 0;
}
.bar-track {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}
.bar-fill.green { background: var(--green); }
.bar-fill.red   { background: var(--red); }
.bar-fill.blue  { background: var(--blue); }
.summary-pct {
    width: 44px;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-muted);
    flex-shrink: 0;
}

/* Welcome card */
.welcome-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 56px 40px;
    text-align: center;
    margin-top: 8px;
}
.welcome-icon {
    font-size: 3rem;
    margin-bottom: 16px;
    display: block;
}
.welcome-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: var(--text-primary);
    margin-bottom: 10px;
}
.welcome-desc {
    font-size: 0.9rem;
    color: var(--text-muted);
    max-width: 400px;
    margin: 0 auto 28px;
    line-height: 1.65;
}
.feature-chips {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
}
.chip {
    background: var(--surface-2);
    border: 1px solid var(--border-2);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.78rem;
    color: var(--text-muted);
}
</style>
""", unsafe_allow_html=True)


# ── MODEL & DATA ────────────────────────────────────────────────────────────

@st.cache_resource
def load_bert_model():
    tokenizer = AutoTokenizer.from_pretrained("./bert_laptop_sentiment_model")
    model = AutoModelForSequenceClassification.from_pretrained("./bert_laptop_sentiment_model")
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

@st.cache_data
def load_data():
    return pd.read_csv("data/laptop_reviews.csv")

df = load_data().dropna(subset=["review_text", "review_rating"])


# ── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-logo">Laptolyze</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tag">🤖 BERT AI Engine</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    laptop_choice = st.selectbox(
        "Select Laptop",
        sorted(df["product_name"].unique().tolist()),
        label_visibility="collapsed"
    )

    spec_info = df[df["product_name"] == laptop_choice].iloc[0]

    st.markdown("""<div style="font-size:0.7rem;color:#64748b;
        text-transform:uppercase;letter-spacing:0.08em;
        margin:16px 0 8px">Specifications</div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="spec-grid">
        <div class="spec-item">
            <span class="spec-icon">⚡</span>
            <div class="spec-text">
                <span class="spec-label">Processor</span>
                {spec_info['processor']}
            </div>
        </div>
        <div class="spec-item">
            <span class="spec-icon">🧠</span>
            <div class="spec-text">
                <span class="spec-label">Memory</span>
                {spec_info['ram']}
            </div>
        </div>
        <div class="spec-item">
            <span class="spec-icon">💾</span>
            <div class="spec-text">
                <span class="spec-label">Storage</span>
                {spec_info['storage']}
            </div>
        </div>
        <div class="spec-item">
            <span class="spec-icon">🖥️</span>
            <div class="spec-text">
                <span class="spec-label">Display</span>
                {spec_info['display(in inch)']} inch
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── MAIN ─────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="page-title">AI Review Analysis</div>
<div class="page-subtitle">{laptop_choice}</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

laptop_subset = df[df["product_name"] == laptop_choice]
avg_rating    = laptop_subset["review_rating"].mean()
total_reviews = len(laptop_subset)

# ── METRICS ──
m1, m2, m3, m4 = st.columns(4)
m1.metric("Avg Rating",   f"{avg_rating:.1f} ⭐")
m2.metric("Reviews",      total_reviews)
m3.metric("Spec Score",   f"{spec_info['spec_rating']} / 5")
m4.metric("Market Price", f"₹{int(spec_info['price(in Rs.)']):,}")

st.markdown("<br>", unsafe_allow_html=True)

# ── BUTTON ──
run = st.button("Run BERT Deep Discovery →", use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

if run:
    nlp = load_bert_model()

    with st.spinner("Analysing reviews with BERT..."):
        reviews   = laptop_subset["review_text"].astype(str).tolist()
        full_text = " ".join(reviews[:50])
        sentences = [s.strip() for s in re.split(r'[.!?]', full_text) if len(s.strip()) > 25]
        sentences = sentences[:40]
        results   = nlp(sentences)

        pros = []
        cons = []
        neut = []

        for sentence, result in zip(sentences, results):
            if result["score"] < 0.65:
                continue
            label = result["label"]
            if label == "LABEL_2":
                pros.append(sentence)
            elif label == "LABEL_0":
                cons.append(sentence)
            else:
                neut.append(sentence)

    # ── THREE COLUMNS ──
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""<div class="section-label green">
            <span class="dot green"></span> Strengths
        </div>""", unsafe_allow_html=True)
        if pros:
            for p in list(dict.fromkeys(pros))[:8]:
                st.markdown(f'<div class="insight-card pro">{p}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-card neu">No strengths detected in this sample.</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("""<div class="section-label red">
            <span class="dot red"></span> Issues
        </div>""", unsafe_allow_html=True)
        if cons:
            for c in list(dict.fromkeys(cons))[:8]:
                st.markdown(f'<div class="insight-card con">{c}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-card pro">No major issues detected.</div>', unsafe_allow_html=True)

    with c3:
        st.markdown("""<div class="section-label blue">
            <span class="dot blue"></span> Neutral
        </div>""", unsafe_allow_html=True)
        if neut:
            for n in list(dict.fromkeys(neut))[:8]:
                st.markdown(f'<div class="insight-card neu">{n}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="insight-card neu">No neutral feedback detected.</div>', unsafe_allow_html=True)

    # ── SENTIMENT SUMMARY ──
    st.markdown("<br>", unsafe_allow_html=True)
    total = len(pros) + len(cons) + len(neut)

    if total > 0:
        pp = round(len(pros) / total * 100, 1)
        np_ = round(len(cons) / total * 100, 1)
        neu_p = round(len(neut) / total * 100, 1)

        st.markdown("""<div style="font-size:0.72rem;font-weight:700;
            letter-spacing:0.1em;text-transform:uppercase;
            color:#64748b;margin-bottom:12px">
            Sentiment Breakdown
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="summary-bar-wrap">
            <div class="summary-row">
                <div class="summary-name" style="color:#10b981">Positive</div>
                <div class="bar-track">
                    <div class="bar-fill green" style="width:{pp}%"></div>
                </div>
                <div class="summary-pct">{pp}%</div>
            </div>
            <div class="summary-row">
                <div class="summary-name" style="color:#f43f5e">Negative</div>
                <div class="bar-track">
                    <div class="bar-fill red" style="width:{np_}%"></div>
                </div>
                <div class="summary-pct">{np_}%</div>
            </div>
            <div class="summary-row">
                <div class="summary-name" style="color:#3b82f6">Neutral</div>
                <div class="bar-track">
                    <div class="bar-fill blue" style="width:{neu_p}%"></div>
                </div>
                <div class="summary-pct">{neu_p}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div class="welcome-card">
        <span class="welcome-icon">🔍</span>
        <div class="welcome-title">Ready to analyse</div>
        <div class="welcome-desc">
            Select a laptop from the sidebar and click the button above to
            run BERT-powered sentiment discovery on real user reviews.
        </div>
        <div class="feature-chips">
            <span class="chip">✅ Strengths</span>
            <span class="chip">⚠️ Issues</span>
            <span class="chip">🔵 Neutral Signals</span>
            <span class="chip">📊 Sentiment Score</span>
        </div>
    </div>
    """, unsafe_allow_html=True)