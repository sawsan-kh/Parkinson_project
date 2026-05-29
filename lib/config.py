# This file defines common variables that are used across python libraries
import os
my_random_state = 12883823
grid_search_scoring_metric = "matthews_corrcoef"

output_files_dir = "../output_files/"
input_files_dir = "../input_files/"
results_files_dir = "../result_files/"
log_files_dir = "../log/"

{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Live Doctor",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/live_doctor.py",
            "console": "integratedTerminal"
        }
    ]
}




# 📁 PATHS (UPDATE THESE 3 ONLY)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_files_dir = os.path.join(BASE_DIR, "../input_files/")
output_files_dir = os.path.join(BASE_DIR, "../output_files/")
results_files_dir = os.path.join(BASE_DIR, "../result_files/")
videos_dir = r"C:\Users\SAWSAN\Downloads\PD\PD\videos"

# 🎯 CONSTANTS
dataset_identifier = "fis"
max_frames_per_video = 300

# ✅ Auto-create directories
os.makedirs(input_files_dir, exist_ok=True)
os.makedirs(output_files_dir, exist_ok=True)
os.makedirs(results_files_dir, exist_ok=True)