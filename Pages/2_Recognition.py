import streamlit as st
import subprocess, sys, os, pickle

st.set_page_config(page_title="Recognition", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.block-container { padding-top: 3.5rem; padding-bottom: 3rem; max-width: 820px; }

h1 { font-size: 2.2rem !important; font-weight: 800 !important; color: #ffffff !important; }
.subtitle { font-size: 1rem; color: #c0c0c0; margin-bottom: 1rem; }

.persons-bar {
    background: #0d1f12;
    border: 1px solid #22c55e33;
    border-radius: 10px;
    padding: 0.9rem 1.3rem;
    font-size: 0.95rem;
    color: #d0d0d0;
    margin-bottom: 1.5rem;
}
.persons-bar span { color: #4ade80; font-weight: 700; }

.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin: 1.2rem 0; }

.info-card {
    background: #111;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
}
.ic-label {
    font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.2px;
    color: #888; margin-bottom: 0.4rem;
}
.ic-value { font-size: 0.93rem; color: #d8d8d8; line-height: 1.5; }
.green { color: #4ade80; font-weight: 600; }
.cyan  { color: #38bdf8; font-weight: 600; }

.note {
    background: #0d1f12;
    border: 1px solid #22c55e33;
    border-left: 3px solid #22c55e;
    border-radius: 8px;
    padding: 1rem 1.3rem;
    font-size: 0.93rem;
    color: #d0d0d0;
    line-height: 1.7;
    margin: 1.2rem 0;
}
.note b { color: #fff; }
</style>
""", unsafe_allow_html=True)

st.title("Live Recognition")
st.markdown('<p class="subtitle">Start the camera to detect and identify persons wearing a red cap in real time.</p>', unsafe_allow_html=True)
st.divider()

embeddings_path = "embeddings/known_faces.pkl"
if not os.path.exists(embeddings_path):
    st.warning("No embeddings found. Please register at least one person first.")
    st.stop()

try:
    with open(embeddings_path, "rb") as f:
        known = pickle.load(f)
    names_str = ", ".join(known.keys())
    st.markdown(f'<div class="persons-bar">Registered persons: <span>{names_str}</span></div>', unsafe_allow_html=True)
except Exception:
    pass

st.markdown("""
<div class="info-grid">
    <div class="info-card">
        <div class="ic-label">Green Box</div>
        <div class="ic-value"><span class="green">Red cap detected</span> — person identified with name label</div>
    </div>
    <div class="info-card">
        <div class="ic-label">Cyan Box</div>
        <div class="ic-value"><span class="cyan">Face detected</span> — no red cap present</div>
    </div>
    <div class="info-card">
        <div class="ic-label">Not Registered</div>
        <div class="ic-value">Red cap detected but face not enrolled in the system</div>
    </div>
    <div class="info-card">
        <div class="ic-label">To Stop</div>
        <div class="ic-value">Press <b style="color:#fff">Q</b> inside the camera window</div>
    </div>
</div>

<div class="note">
The camera window may take <b>1-2 minutes</b> to open while the AI models load.<br>
Please wait and check the <b>Windows taskbar</b> if nothing appears.
</div>
""", unsafe_allow_html=True)

if st.button("Start Recognition", type="primary"):
    st.warning("Camera is opening. Models are loading — this takes 1-2 minutes. Check the taskbar.")
    with st.spinner("Recognition running — camera window is open in the taskbar..."):
        result = subprocess.run(
            [sys.executable, "run_recognition.py"],
            capture_output=True, text=True
        )
    if result.returncode == 0:
        st.success("Recognition session ended.")
    else:
        st.error("An error occurred.")
        if result.stderr:
            st.code(result.stderr[-2000:], language="text")
