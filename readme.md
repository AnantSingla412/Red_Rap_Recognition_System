# Red Cap Recognition System

Real-time detection and identification of persons wearing a red cap using
YOLOv8, ArcFace, and OpenCV.

---

## How It Works

1. YuNet detects all faces in the webcam frame
2. YOLOv8 detects cap bounding boxes
3. HSV color check confirms the cap is red
4. ArcFace identifies the person by comparing face embeddings
5. A green bounding box with the person's name is drawn on screen

---

## Project Structure

```
red_cap_reco/
│
├── models/
│   ├── face_detection_yunet_2023mar.onnx   # included in repo
│   ├── best_model.pt                        # included in repo
│   └── w600k_r50.onnx                       # download manually (see below)
│
├── modules/
│   ├── face_detector.py
│   ├── hat_detector.py
│   ├── color_checker.py
│   ├── arcface_recognizer.py
│   ├── capture_enrollment.py
│   └── tracker.py
│
├── pages/
│   ├── 1_Register.py
│   └── 2_Recognition.py
│
├── dataset/                  # created after enrollment
├── embeddings/               # created after enrollment
│
├── main.py                   # run without UI
├── run_capture.py            # webcam enrollment capture
├── enroll.py                 # build embeddings from dataset
├── app.py                   # Streamlit home page
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/red_cap_reco.git
cd red_cap_reco
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the ArcFace model

`w600k_r50.onnx` is too large for GitHub. Download it manually:


```bash
Windows (PowerShell)
curl -L -o models/w600k_r50.onnx https://huggingface.co/public-data/insightface/resolve/33c1063c49c785b7652d3fd529f86fa4f149392b/models/buffalo_l/w600k_r50.onnx
---


Or download manually:
[https://huggingface.co/public-data/insightface/resolve/33c1063c49c785b7652d3fd529f86fa4f149392b/models/buffalo_l/w600k_r50.onnx](https://huggingface.co/public-data/insightface/resolve/33c1063c49c785b7652d3fd529f86fa4f149392b/models/buffalo_l/w600k_r50.onnx)

Place the downloaded file inside the `models/` folder.

## Running the Project

### Option A — Streamlit UI (recommended)

```bash
streamlit run app.py
```

Then use the sidebar to:
- **Register** — enroll a new person via webcam
- **Recognition** — start live red cap detection

### Option B — Terminal only (no UI)

**Step 1: Capture face images**
```bash
python run_capture.py --name "Your Name"
```
Move your head left, right, up, down, closer, and farther.
The camera captures 8 images automatically. Press `Q` to stop early.

**Step 2: Build embeddings**
```bash
python enroll.py
```

**Step 3: Run live recognition**
```bash
python main.py
```
Press `Q` inside the camera window to quit.

---

## Requirements

- Python 3.10.11 or higher
- Webcam
- Windows / macOS / Linux

See `requirements.txt` for full package list.

---

## Notes

- First launch takes **1–2 minutes** — models are loading in the background
- Enroll each person in **good lighting** for best accuracy
- Re-run `enroll.py` after adding new persons to update embeddings




