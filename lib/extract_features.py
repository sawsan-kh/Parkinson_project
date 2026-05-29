from . import config
import os
import pandas as pd
import numpy as np

from scipy.signal import find_peaks
from scipy import signal

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif
)

np.random.seed(42)

# =========================================================
# MAIN
# =========================================================

def extract_all_features(dataset_identifier):

    extract_tsfresh_features(dataset_identifier)

    extract_classical_features(dataset_identifier)

    extract_fi_tsfresh_features(dataset_identifier)


# =========================================================
# LOAD TIME SERIES
# =========================================================

def load_time_series(dataset_identifier):

    ts_path = os.path.join(
        config.output_files_dir,
        dataset_identifier + "_data_time_series.csv"
    )

    if not os.path.exists(ts_path):
        raise FileNotFoundError(f"❌ Missing file: {ts_path}")

    data = pd.read_csv(ts_path, dtype={"ID": str})

    data = data.drop(columns=['Unnamed: 0'], errors='ignore')

    data = (
        data
        .reset_index()
        .sort_values(by=['ID', 'index'])
        .set_index('index')
    )

    return data


# =========================================================
# CLEAN FEATURES
# =========================================================

def clean_features(df):

    if df is None or df.empty:
        print("⚠️ No selected features")
        return pd.DataFrame()

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] == 0:
        print("⚠️ No numeric features")
        return pd.DataFrame(index=df.index)

    numeric_df = numeric_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    numeric_df = numeric_df.fillna(0)

    try:

        selector = VarianceThreshold(threshold=0.001)

        transformed = selector.fit_transform(numeric_df)

        cols = numeric_df.columns[
            selector.get_support()
        ]

        cleaned = pd.DataFrame(
            transformed,
            columns=cols,
            index=df.index
        )

        print(f"✅ Features after cleaning: {cleaned.shape}")

        return cleaned

    except Exception as e:

        print(f"⚠️ VarianceThreshold failed: {e}")

        return numeric_df


# =========================================================
# TSFRESH FEATURES
# =========================================================

def extract_tsfresh_features(dataset_identifier):

    print("\n🔥 TSFRESH FEATURES")

    data_final = load_time_series(dataset_identifier)

    data_final['FREQUENCY'] = data_final['FREQUENCY'].fillna(0)

    y_path = os.path.join(
        config.input_files_dir,
        dataset_identifier + "_diagnostic.csv"
    )

    y_diagnostic = pd.read_csv(
        y_path,
        dtype={'UPDRS': np.int32, "ID": str}
    )

    y_diagnostic = y_diagnostic.sort_values(by=['ID'])

    data_final = data_final[
        data_final.ID.isin(y_diagnostic.ID)
    ]

    y_diagnostic = y_diagnostic[
        y_diagnostic.ID.isin(data_final.ID)
    ]

    y_all = pd.Series(
        data=y_diagnostic["UPDRS"].values,
        index=y_diagnostic["ID"].values
    )

    # =====================================================
    # TSFRESH EXTRACTION
    # =====================================================

    extracted = extract_features(
        data_final,
        column_id="ID",
        impute_function=impute
    )

    print(f"📈 TSFresh extracted: {extracted.shape}")

    # =====================================================
    # FEATURE SELECTION
    # =====================================================

    selected = select_features(
        extracted,
        y_all,
        multiclass=True,
        n_significant=1,
        ml_task="classification"
    )

    print(f"📉 Selected features: {selected.shape}")

    if selected.empty:
        print("⚠️ Empty TSFresh features")
        return

    # =====================================================
    # CLEAN FEATURES
    # =====================================================

    cleaned = clean_features(selected)

    if cleaned.empty:
        print("⚠️ Empty cleaned features")
        return

    # =====================================================
    # SELECT TOP K FEATURES
    # =====================================================

    k_best = min(50, cleaned.shape[1])

    selector = SelectKBest(
        score_func=f_classif,
        k=k_best
    )

    X_new = selector.fit_transform(
        cleaned,
        y_all.loc[cleaned.index]
    )

    cols = cleaned.columns[
        selector.get_support()
    ]

    cleaned = pd.DataFrame(
        X_new,
        columns=cols,
        index=cleaned.index
    )

    print(f"✅ Final selected features: {cleaned.shape}")

    # =====================================================
    # SAVE
    # =====================================================

    cleaned.insert(0, "ID", cleaned.index)

    save_path = os.path.join(
        config.output_files_dir,
        dataset_identifier + "_tsfresh_features.csv"
    )

    cleaned.to_csv(save_path, index=False)

    print("✅ TSFRESH DONE")


# =========================================================
# CLASSICAL FEATURES
# =========================================================

def extract_classical_features(dataset_identifier):

    print("\n🔥 CLASSICAL FEATURES")

    data_final = load_time_series(dataset_identifier)

    fps_path = os.path.join(
        config.output_files_dir,
        dataset_identifier + "_fps_videos.csv"
    )

    fps_per_video = pd.read_csv(
        fps_path,
        dtype={"ID": str, "FPS": float}
    ).set_index("ID")

    summary_df = data_final.groupby('ID').agg({

        'SMOOTHED_AMPLITUDE': ['mean', 'std', 'max'],
        'VELOCITY': ['mean', 'std', 'max'],
        'ACCELERATION': ['mean', 'std', 'max'],
        'DISTANCE_ANG': ['mean', 'std'],
        'FREQUENCY': ['mean', 'std', 'count']

    })

    summary_df.columns = [
        '_'.join(col)
        for col in summary_df.columns
    ]

    summary_df = summary_df.reset_index()

    regularity = []

    for vid, group in data_final.groupby('ID'):

        amp = group['SMOOTHED_AMPLITUDE'].values

        fps = fps_per_video.at[vid, "FPS"]

        peaks, _ = find_peaks(
            amp,
            distance=max(1, int(fps // 3))
        )

        times = peaks / fps

        periods = np.diff(times)

        if len(periods) > 1:

            std = np.std(periods)

            cv = std / np.mean(periods)

        else:

            std, cv = np.nan, np.nan

        regularity.append({

            'ID': vid,
            'INTERVAL_STD': std,
            'INTERVAL_CV': cv

        })

    summary_df = summary_df.merge(
        pd.DataFrame(regularity),
        on='ID',
        how='left'
    )

    save_path = os.path.join(
        config.output_files_dir,
        dataset_identifier + "_classical_features.csv"
    )

    summary_df.to_csv(save_path, index=False)

    print(f"📊 Classical features: {summary_df.shape}")

    print("✅ CLASSICAL DONE")


# =========================================================
# FI + TSFRESH FEATURES
# =========================================================

def extract_fi_tsfresh_features(dataset_identifier):

    print("\n🔥 FI + TSFRESH FEATURES")

    data_final = load_time_series(dataset_identifier)

    data_final['FREQUENCY'] = data_final['FREQUENCY'].fillna(0)

    fps_path = os.path.join(
        config.output_files_dir,
        dataset_identifier + "_fps_videos.csv"
    )

    fps_per_video = pd.read_csv(
        fps_path,
        dtype={"ID": str, "FPS": float}
    ).set_index("ID")

    y_path = os.path.join(
        config.input_files_dir,
        dataset_identifier + "_diagnostic.csv"
    )

    y_diagnostic = pd.read_csv(
        y_path,
        dtype={'UPDRS': np.int32, "ID": str}
    )

    y_all = pd.Series(
        data=y_diagnostic["UPDRS"].values,
        index=y_diagnostic["ID"].values
    )

    rows = []

    for vid, group in data_final.groupby('ID'):

        amp = group['SMOOTHED_AMPLITUDE'].values

        fps = fps_per_video.at[vid, "FPS"]

        f, t, Zxx = signal.stft(amp, fs=fps)

        Zxx = np.abs(Zxx)

        max_intensity = Zxx.max(axis=0)

        max_freq = f[np.argmax(Zxx, axis=0)]

        for i in range(len(max_freq)):

            rows.append({

                'ID': vid,
                'max_freq': max_freq[i],
                'max_intensity': max_intensity[i],
                'IF_value': max_freq[i] * max_intensity[i]

            })

    df = pd.DataFrame(rows)

    extracted = extract_features(
        df,
        column_id="ID",
        impute_function=impute
    )

    common_ids = np.intersect1d(
        extracted.index.astype(str),
        y_all.index.astype(str)
    )

    extracted = extracted.loc[common_ids]

    y_filtered = y_all.loc[common_ids]

    print(f"📊 Extracted features: {extracted.shape}")

    selected = select_features(
        extracted,
        y_filtered,
        multiclass=True,
        n_significant=1,
        ml_task="classification"
    )

    selected = clean_features(selected)

    if selected.empty:
        print("⚠️ Empty FI+TSFresh features")
        return

    selected.insert(0, "ID", selected.index)

    save_path = os.path.join(
        config.output_files_dir,
        dataset_identifier + "_fi_tsfresh_features.csv"
    )

    selected.to_csv(save_path, index=False)

    print("✅ FI + TSFRESH DONE")