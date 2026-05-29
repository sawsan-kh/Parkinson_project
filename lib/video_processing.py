from . import config

import os

import cv2

import math

import pandas as pd

import numpy as np

from scipy.signal import find_peaks

from tqdm import tqdm

from mediapipe.tasks.python.components.containers import NormalizedLandmark

from mediapipe import Image, ImageFormat

from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from mediapipe.tasks.python import BaseOptions

def process_video(path_videos, dataset_identifier, max_frames_per_video=300):

    """Complete working video processing """

   

    data_final = pd.DataFrame()

    frame_percentage_videos = pd.DataFrame()

    frame_ratio_videos = pd.DataFrame()

    frame_percentage_rejected_videos = pd.DataFrame()

   

    video_files = [f for f in os.listdir(path_videos) if f.lower().endswith(('.mp4','.avi','.mov','.mkv'))]

    video_files.sort()

   

    print(f"Found {len(video_files)} videos")

   

    # MediaPipe setup

    try:

        base_options = BaseOptions(model_asset_path='../utils/hand_landmarker.task')

        options = HandLandmarkerOptions(base_options=base_options, running_mode=RunningMode.IMAGE, num_hands=1)

        detector = HandLandmarker.create_from_options(options)

        print("✅ MediaPipe ready!")

    except:

        print("❌ Download hand_landmarker.task to utils/")

        return

   

    for file in tqdm(video_files, desc="Processing"):

        video_path = os.path.join(path_videos, file)

        id_user = os.path.splitext(file)[0]

       

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        frames_to_process = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames_per_video)

       

        good_frames, bad_frames = 0, 0

        landmark_frames = []

       

        # Process frames (skip every other for speed)

        for i in range(0, frames_to_process, 2):

            cap.set(cv2.CAP_PROP_POS_FRAMES, i)

            ret, frame = cap.read()

            if not ret: break

           

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

           

            results = detector.detect(mp_image)

           

            # ✅ CORRECT LANDMARK EXTRACTION

            if results.handedness and results.hand_landmarks:

                # Convert to protobuf for proper access

                hand_landmarks = results.hand_landmarks[0]

                if len(hand_landmarks) >= 9:

                    landmark_frames.append(hand_landmarks)

                    good_frames += 1

            else:

                bad_frames += 1

       

        cap.release()

       

        total = good_frames + bad_frames

        pct_good = (good_frames/total*100) if total > 0 else 0

       

        # Save FPS

        frame_ratio_videos = pd.concat([frame_ratio_videos, pd.DataFrame({

            'ID': [id_user], 'FPS': [fps]

        })], ignore_index=True)

       

        # Process good videos

        if pct_good > 85 and len(landmark_frames) > 20:

            df = process_landmarks(landmark_frames, fps)

            if len(df) > 0:

                df['ID'] = id_user

                data_final = pd.concat([data_final, df], ignore_index=True)

                frame_percentage_videos = pd.concat([frame_percentage_videos, pd.DataFrame({

                    'ID': [id_user], 'Percentage': [pct_good]

                })], ignore_index=True)

   

    # Save everything

    if len(data_final) > 0:

        data_final.to_csv(config.output_files_dir + f"{dataset_identifier}_data_time_series.csv", index=False)

   

    frame_ratio_videos.set_index('ID', inplace=True)

    frame_ratio_videos.to_csv(config.output_files_dir + f"{dataset_identifier}_fps_videos.csv")

   

    if len(frame_percentage_videos) > 0:

        frame_percentage_videos.set_index('ID', inplace=True)

        frame_percentage_videos.to_csv(config.log_files_dir + f"{dataset_identifier}_frame_rate_processed_videos.csv")

   

    print(f"✅ COMPLETE! {len(data_final)} frames from {len(frame_percentage_videos)} videos")

def process_landmarks(landmark_frames, fps):

    """Extract kinematic features"""

    data = []

   

    for landmarks in landmark_frames:

        wrist = landmarks[0]

        thumb = landmarks[4]

        index = landmarks[8]

       

        # 3D positions

        wrist_pos = np.array([wrist.x, wrist.y, wrist.z])

        thumb_pos = np.array([thumb.x, thumb.y, thumb.z])

        index_pos = np.array([index.x, index.y, index.z])

       

        # Distances

        hand_size = np.linalg.norm(index_pos - wrist_pos)

        amplitude = np.linalg.norm(thumb_pos - index_pos)

        norm_amplitude = amplitude / hand_size if hand_size > 0 else 0

       

        # Angle

        vec_wi = wrist_pos - index_pos

        vec_wt = wrist_pos - thumb_pos

        norm_wi = np.linalg.norm(vec_wi)

        norm_wt = np.linalg.norm(vec_wt)

        angle = 0

        if norm_wi > 0 and norm_wt > 0:

            cos_angle = np.dot(vec_wi, vec_wt) / (norm_wi * norm_wt)

            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

       

        data.append({

            'DISTANCE_ANG': angle / 90.0,

            'NORMALIZED_AMPLITUDE': norm_amplitude

        })

   

    if not data:

        return pd.DataFrame()

   

    df = pd.DataFrame(data)

   

    # Smoothing

    df['SMOOTHED_AMPLITUDE'] = df['NORMALIZED_AMPLITUDE'].rolling(3, center=True).mean()

    df['SMOOTHED_AMPLITUDE'] = df['SMOOTHED_AMPLITUDE'].bfill().ffill()

   

    # Derivatives

    dt = 2.0 / fps  # Frame skip compensation

    df['VELOCITY'] = df['SMOOTHED_AMPLITUDE'].diff() / dt

    df['VELOCITY'] = df['VELOCITY'].fillna(0)

    df['ACCELERATION'] = df['VELOCITY'].diff() / dt

    df['ACCELERATION'] = df['VELOCITY'].fillna(0)

   

    # Frequency

    df['FREQUENCY'] = 0.0

    if len(df) > 10:

        peaks, _ = find_peaks(df['SMOOTHED_AMPLITUDE'], distance=fps//6)

        if len(peaks) > 1:

            times = np.arange(len(df)) * dt

            periods = np.diff(times[peaks])

            freqs = 1.0 / periods

            df.iloc[peaks[1]:min(peaks[1]+len(freqs), len(df)),

                   df.columns.get_loc('FREQUENCY')] = freqs

   

    return df[['DISTANCE_ANG', 'SMOOTHED_AMPLITUDE', 'VELOCITY', 'ACCELERATION', 'FREQUENCY']]