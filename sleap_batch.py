from requests import session
from behavior_utils import *
import subprocess
import os
from sleap.io import format
from pathlib import Path

experiment = 'clickbait-visual'

model_path = f"S:/clickbait-motivate/sleap-model/models/cb_motivate_sparse/"
bonsai_video_paths = get_file_paths(f'S:/{experiment}/bonsai', 'avi', '', print_paths=True, print_n=5)

def test_tf_gpu():
    cmd = "conda run -n sleap python -c \"import tensorflow as tf; print('TF version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU')); print('Built with CUDA:', tf.test.is_built_with_cuda())\""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Stderr: {result.stderr}")



def run_sleap():
    sleap_command = 'sleap-track'
    # Ensure CUDA environment variables are set
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'  # or whatever GPU ID you want to use
    # Run sleap inference on each video
    for video_path in bonsai_video_paths[:1]:
        output_dir = video_path.parent
        cmd = f"conda run -n sleap {sleap_command} '{video_path}' -m {model_path} -o {output_dir} --gpu 0 "
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Error: {result.stderr}")

# test_tf_gpu()
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Force GPU 0
# run_sleap()

def run_sleap_python_api():
    import sleap
    import os
    
    # Force TensorFlow to use GPU
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    try:
        # Load the model
        predictor = sleap.load_model(model_path)
        print(f"Model loaded successfully")
        
        for video_path in bonsai_video_paths[:10]:
            print(f"Processing: {video_path}")
            
            # Load video
            video = sleap.Video.from_filename(str(video_path))
            print(f"Video loaded: {len(video)} frames")
            
            # Run prediction
            predictions = predictor.predict(video, make_labels=True)
            
            # Make output directories
            mouse_id = video_path.parent.parts[-2]
            session_id = video_path.parent.parts[-1]
            experiment_dir = Path(*video_path.parent.parts[-5:-3])
            output_path = experiment_dir / "sleap" / mouse_id / session_id / f"{mouse_id}_{session_id}_predictions.slp"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Save results
            sleap.Labels.save_file(predictions, str(output_path))
            print(f"Saved to: {output_path.parent}")
            
    except Exception as e:
        print(f"Error with Python API: {e}")

run_sleap_python_api()