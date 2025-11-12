from behavior_utils import *
import subprocess
import os
from pathlib import Path

experiment = 'clickbait-visual'

model_path = f"S:/sfn-poster/sleap-nn/clickbait-motivate-sparse-v2/models/n=1057"
bonsai_video_paths = get_file_paths(f'S:/sfn-poster/sleap-nn/7004', 'avi', '', print_paths=True, print_n=5)

def run_sleap():
    sleap_command = 'sleap-nn track'
    # Ensure CUDA environment variables are set
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'  # or whatever GPU ID you want to use
    # Run sleap inference on each video
    for video_path in bonsai_video_paths:
        output_dir = video_path.parent
        cmd = f"conda run -n sleap-25 {sleap_command} --data_path {str(video_path)} --model_paths {model_path}"
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")

run_sleap()