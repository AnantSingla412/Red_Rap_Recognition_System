import streamlit as st

st.set_page_config(page_title="Red Cap Recognition", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.block-container { padding-top: 3.5rem; padding-bottom: 3rem; max-width: 820px; }

h1 { font-size: 2.6rem !important; font-weight: 800 !important; color: #ffffff !important; letter-spacing: -0.5px; }

.subtitle { font-size: 1.05rem; color: #c0c0c0; line-height: 1.7; margin-bottom: 1.8rem; }

.pill {
    display: inline-block;
    background: #1e1e2e;
    border: 1px solid #7c3aed44;
    color: #a78bfa;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 4px 14px;
    border-radius: 6px;
    margin: 0 5px 6px 0;
}

.card {
    border-radius: 12px;
    padding: 1.6rem 1.8rem;
    height: 100%;
}
.card-reg {
    background: linear-gradient(135deg, #1a0a2e, #2d1b69);
    border: 1px solid #7c3aed55;
}
.card-rec {
    background: linear-gradient(135deg, #0a1f12, #0d3320);
    border: 1px solid #22c55e55;
}
.card-step { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: #888; margin-bottom: 0.5rem; }
.card-title-reg { font-size: 1.15rem; font-weight: 700; color: #c4b5fd; margin-bottom: 0.5rem; }
.card-title-rec { font-size: 1.15rem; font-weight: 700; color: #4ade80; margin-bottom: 0.5rem; }
.card-desc { font-size: 0.93rem; color: #d0d0d0; line-height: 1.6; margin: 0; }
</style>
""", unsafe_allow_html=True)

st.title("Red Cap Recognition System")
st.markdown('<p class="subtitle">Detects persons wearing a red cap in real time and identifies them using face recognition.</p>', unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom: 2rem;">
    <span class="pill">YOLOv8</span>
    <span class="pill">ArcFace</span>
    <span class="pill">OpenCV</span>
    <span class="pill">YuNet</span>
    <span class="pill">Streamlit</span>
</div>
""", unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns(2, gap="medium")
with col1:
    st.markdown("""
    <div class="card card-reg">
        <div class="card-step">Step 1</div>
        <div class="card-title-reg">Register a Person</div>
        <div class="card-desc">Enroll a new person by capturing face images from the webcam across multiple angles and positions. Takes about 30 seconds.</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card card-rec">
        <div class="card-step">Step 2</div>
        <div class="card-title-rec">Start Recognition</div>
        <div class="card-desc">Launch the live camera. Anyone wearing a red cap is detected and identified in real time with a green bounding box.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Use the sidebar on the left to navigate between pages.")
