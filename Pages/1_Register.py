import streamlit as st
import subprocess, os, sys, glob

st.set_page_config(page_title="Register", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.block-container { padding-top: 3.5rem; padding-bottom: 3rem; max-width: 820px; }

h1 { font-size: 2.2rem !important; font-weight: 800 !important; color: #ffffff !important; }
.subtitle { font-size: 1rem; color: #c0c0c0; margin-bottom: 1rem; }

.pill {
    display: inline-block;
    background: #1e1e2e;
    border: 1px solid #7c3aed44;
    color: #c4b5fd;
    font-size: 0.88rem;
    font-weight: 600;
    padding: 5px 14px;
    border-radius: 7px;
    margin: 0 6px 6px 0;
}
.pill small { color: #888; margin-left: 5px; font-size: 0.78rem; font-weight: 400; }

.note {
    background: #12101f;
    border: 1px solid #7c3aed33;
    border-left: 3px solid #7c3aed;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    font-size: 0.93rem;
    color: #d0d0d0;
    line-height: 1.8;
    margin: 1.3rem 0;
}
.note b { color: #fff; }

div[data-testid="stTextInput"] label {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #e0e0e0 !important;
}
div[data-testid="stTextInput"] input {
    font-size: 1rem !important;
    background: #111 !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    color: #fff !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 2px #7c3aed33 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Register New Person")
st.markdown('<p class="subtitle">Capture face images to enroll a new person into the recognition system.</p>', unsafe_allow_html=True)
st.divider()

dataset_dir = "dataset"
existing = []
if os.path.exists(dataset_dir):
    existing = [d for d in os.listdir(dataset_dir)
                if os.path.isdir(os.path.join(dataset_dir, d))]

if existing:
    st.markdown("**Registered Persons**")
    pills = " ".join([
        f'<span class="pill">{n}<small>{len(glob.glob(f"dataset/{n}/*.jpg"))} images</small></span>'
        for n in existing
    ])
    st.markdown(f'<div style="margin-bottom: 0.5rem">{pills}</div>', unsafe_allow_html=True)
    st.divider()

name = st.text_input("Full Name", placeholder="e.g. Alice")

st.markdown("""
<div class="note">
<b>Before starting</b> — make sure you are in a well-lit area.<br>
Move your head naturally: left, right, up, down, closer, farther.<br>
The camera captures automatically when it detects a new stable position.<br>
Press <b>Q</b> inside the camera window to stop at any time.<br>
The camera window may take <b>1-2 minutes</b> to open while the AI models load.<br>
Please wait and check the <b>Windows taskbar</b> if nothing appears.            
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")
with col1:
    start_capture = st.button("Start Capture", type="primary", disabled=not name.strip(), use_container_width=True)
with col2:
    build_embeddings = st.button("Build Embeddings", use_container_width=True)

if start_capture and name.strip():
    st.warning("Camera is opening. This takes 1-2 minutes while models load. Check the taskbar if nothing appears immediately.")
    with st.spinner("Capturing — camera window is open in the taskbar..."):
        result = subprocess.run(
            [sys.executable, "run_capture.py", "--name", name.strip()],
            capture_output=True, text=True
        )
    if result.returncode == 0:
        saved = result.stdout.count("[OK] Captured")
        st.success(f"Done. {saved} images saved for **{name.strip()}**. Now click Build Embeddings.")
    else:
        st.error("Capture failed or was cancelled.")
        if result.stderr:
            st.code(result.stderr, language="text")

if build_embeddings:
    with st.spinner("Building embeddings..."):
        result = subprocess.run(
            [sys.executable, "enroll.py"],
            capture_output=True, text=True
        )
    if result.returncode == 0:
        st.success("Done. This person will now be recognized during live detection.")
        st.code(result.stdout, language="text")
    else:
        st.error("Failed to build embeddings.")
        st.code(result.stderr, language="text")
