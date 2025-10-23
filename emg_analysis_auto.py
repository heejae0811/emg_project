import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt
import glob, os

# ===================================
# 공통 함수 정의
# ===================================
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, fs, freq=60.0, Q=30):
    w0 = freq / (fs / 2)
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, data)

def process_emg(file_path):
    # Step 1. Raw Data Import
    df = pd.read_csv(file_path, skiprows=3)
    df.columns = ["sample", "biceps", "triceps", "triceps2"]
    fs = 1000
    dt = 1 / fs
    df["time"] = df["sample"] / fs

    # Step 2. Offset Removal (Demean)
    df["biceps_demean"] = df["biceps"] - np.mean(df["biceps"])
    df["triceps_demean"] = df["triceps"] - np.mean(df["triceps"])

    # Step 3. Filtering
    df["biceps_filt"] = notch_filter(bandpass_filter(df["biceps_demean"], 20, 450, fs), fs, 60)
    df["triceps_filt"] = notch_filter(bandpass_filter(df["triceps_demean"], 20, 450, fs), fs, 60)

    # Step 4. Rectification
    df["biceps_rect"] = np.abs(df["biceps_filt"])
    df["triceps_rect"] = np.abs(df["triceps_filt"])

    # Step 5. RMS Smoothing
    window_ms = 50
    window_size = int(fs * (window_ms / 1000))
    df["biceps_rms"] = df["biceps_rect"].rolling(window=window_size, center=True).apply(lambda x: np.sqrt(np.mean(x**2)))
    df["triceps_rms"] = df["triceps_rect"].rolling(window=window_size, center=True).apply(lambda x: np.sqrt(np.mean(x**2)))

    # Step 6. Normalization
    MVC_biceps = df["biceps_rms"].max()
    MVC_triceps = df["triceps_rms"].max()
    df["biceps_norm"] = (df["biceps_rms"] / MVC_biceps) * 100
    df["triceps_norm"] = (df["triceps_rms"] / MVC_triceps) * 100

    # Step 7. Segmentation
    signal = df["biceps_norm"].fillna(0).values
    time = df["time"].values
    threshold = 10
    is_active = signal > threshold
    segments = []
    in_segment = False
    for i in range(len(signal)):
        if is_active[i] and not in_segment:
            start_idx = i
            in_segment = True
        elif not is_active[i] and in_segment:
            end_idx = i
            in_segment = False
            if (end_idx - start_idx) / fs > 0.1:
                segments.append((time[start_idx], time[end_idx]))

    # Step 8. Feature Extraction
    features = []
    for seg_i, (start, end) in enumerate(segments, start=1):
        seg_mask = (df["time"] >= start) & (df["time"] <= end)
        seg_biceps = df.loc[seg_mask, "biceps_norm"]
        seg_triceps = df.loc[seg_mask, "triceps_norm"]

        mean_biceps = seg_biceps.mean()
        peak_biceps = seg_biceps.max()
        mean_triceps = seg_triceps.mean()
        peak_triceps = seg_triceps.max()

        iEMG_biceps = np.sum(seg_biceps) * dt
        iEMG_triceps = np.sum(seg_triceps) * dt
        duration = end - start

        features.append({
            "File": os.path.basename(file_path),
            "Segment": seg_i,
            "Start (s)": round(start, 3),
            "End (s)": round(end, 3),
            "Duration (s)": round(duration, 3),
            "Mean_Biceps_%MVC": round(mean_biceps, 2),
            "Peak_Biceps_%MVC": round(peak_biceps, 2),
            "iEMG_Biceps": round(iEMG_biceps, 5),
            "Mean_Triceps_%MVC": round(mean_triceps, 2),
            "Peak_Triceps_%MVC": round(peak_triceps, 2),
            "iEMG_Triceps": round(iEMG_triceps, 5)
        })

    return pd.DataFrame(features)


# 여러 파일 일괄 분석
all_files = sorted(glob.glob("./data/*.csv"))
all_features = []

for file in all_files:
    print(f"Processing: {os.path.basename(file)}")
    feature_df = process_emg(file)
    all_features.append(feature_df)

# 결과 합치기
combined_df = pd.concat(all_features, ignore_index=True)
combined_df.to_csv("./results/emg_feature_summary.csv", index=False)
print("\n✅ 파일 저장 완료")

# 파일별 평균 비교
summary = combined_df.groupby("File")[["Mean_Biceps_%MVC", "Peak_Biceps_%MVC", "iEMG_Biceps"]].mean().reset_index()
print("\n--- Mean Feature Comparison ---")
print(summary)
