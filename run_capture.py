# run_capture.py
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import argparse
from modules.capture_enrollment import capture_face_id_style

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
args = parser.parse_args()

capture_face_id_style(args.name, num_images=8)
