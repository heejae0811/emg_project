import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt


# ===================================
# 공통 함수 정의
# ===================================
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)

def notch_filter(data, fs, freq=60.0, Q=30):
    w0 = freq / (fs / 2)
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, data)

def compute_mvc(file_path, muscle_name, fs=1000):
    mvc_df = pd.read_csv(file_path, skiprows=3)
    mvc_df.columns = ["sample", "biceps", "triceps", "triceps2"]
    mvc_df[muscle_name] = pd.to_numeric(mvc_df[muscle_name], errors='coerce').fillna(0)

    # Offset 제거
    mvc_df[f"{muscle_name}_demean"] = mvc_df[muscle_name] - np.mean(mvc_df[muscle_name])
    filt = bandpass_filter(mvc_df[f"{muscle_name}_demean"], 20, 450, fs)
    filt = notch_filter(filt, fs, freq=60)
    rect = np.abs(filt)
    window_size = int(fs * 0.05)
    rms = pd.Series(rect).rolling(window=window_size, center=True).apply(lambda x: np.sqrt(np.mean(x**2)))
    return rms.max(skipna=True)

# MVC 값 미리 계산
fs = 1000
MVC_biceps = compute_mvc("./data/biceps_MVIC_EMG.csv", "biceps", fs)
MVC_triceps = compute_mvc("./data/triceps_MVIC_EMG.csv", "triceps", fs)
print(f"MVC_biceps = {MVC_biceps:.6f}, MVC_triceps = {MVC_triceps:.6f}")


# ===================================
# 분석 함수 정의
# ===================================
def process_emg(file_path, MVC_biceps, MVC_triceps, fs=1000):
    df = pd.read_csv(file_path, skiprows=3)
    df.columns = ["sample", "biceps", "triceps", "triceps2"]
    dt = 1 / fs
    df["time"] = df["sample"] / fs

    # Offset 제거
    df["biceps_demean"] = df["biceps"] - np.mean(df["biceps"])
    df["triceps_demean"] = df["triceps"] - np.mean(df["triceps"])

    # Filtering
    df["biceps_filt"] = notch_filter(bandpass_filter(df["biceps_demean"], 20, 450, fs), fs, freq=60)
    df["triceps_filt"] = notch_filter(bandpass_filter(df["triceps_demean"], 20, 450, fs), fs, freq=60)

    # Rectification
    df["biceps_rect"] = np.abs(df["biceps_filt"])
    df["triceps_rect"] = np.abs(df["triceps_filt"])

    # RMS
    window_size = int(fs * 0.05)
    df["biceps_rms"] = df["biceps_rect"].rolling(window=window_size, center=True).apply(lambda x: np.sqrt(np.mean(x**2)))
    df["triceps_rms"] = df["triceps_rect"].rolling(window=window_size, center=True).apply(lambda x: np.sqrt(np.mean(x**2)))

    # Normalization (%MVC)
    df["biceps_norm"] = (df["biceps_rms"] / MVC_biceps) * 100
    df["triceps_norm"] = (df["triceps_rms"] / MVC_triceps) * 100
    df.fillna(0, inplace=True)

    # Feature 추출 (전체 기간 평균 및 최대)
    mean_biceps = df["biceps_norm"].mean()
    peak_biceps = df["biceps_norm"].max()
    mean_triceps = df["triceps_norm"].mean()
    peak_triceps = df["triceps_norm"].max()

    return mean_biceps, peak_biceps, mean_triceps, peak_triceps


# ===================================
# 모든 파일 자동 처리
# ===================================
data_dir = "./data"
results = []

for filename in os.listdir(data_dir):
    if ("curl" in filename or "hammer" in filename) and filename.endswith(".csv") and "MVIC" not in filename:
        file_path = os.path.join(data_dir, filename)

        # 무게와 타입 추출
        load = filename.split("_")[0].replace("kg", "")
        exercise = "curl" if "curl" in filename else "hammer"

        mean_biceps, peak_biceps, mean_triceps, peak_triceps = process_emg(
            file_path, MVC_biceps, MVC_triceps, fs
        )

        results.append({
            "File": filename,
            "Load (kg)": int(load),
            "Exercise": exercise.capitalize(),
            "Mean_Biceps_%MVC": round(mean_biceps, 2),
            "Peak_Biceps_%MVC": round(peak_biceps, 2),
            "Mean_Triceps_%MVC": round(mean_triceps, 2),
            "Peak_Triceps_%MVC": round(peak_triceps, 2),
        })
        print(f"✅ Processed: {filename}")

# 결과 표 생성
summary_df = pd.DataFrame(results)
summary_df.sort_values(by=["Exercise", "Load (kg)"], inplace=True)
summary_df.to_csv("./results/emg_summary_all.csv", index=False)

print("\n--- Final Summary ---")
print(summary_df)

# ===================================
# 시각화 (Biceps, Triceps Mean vs Load)
# ===================================
plt.figure(figsize=(8,5))
for ex in summary_df["Exercise"].unique():
    temp = summary_df[summary_df["Exercise"] == ex]
    plt.plot(temp["Load (kg)"], temp["Mean_Biceps_%MVC"], 'o-', label=f"{ex} - Biceps")
    plt.plot(temp["Load (kg)"], temp["Mean_Triceps_%MVC"], 's--', label=f"{ex} - Triceps")

plt.xlabel("Load (kg)")
plt.ylabel("Mean EMG (%MVC)")
plt.title("Biceps & Triceps Mean %MVC by Load")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
