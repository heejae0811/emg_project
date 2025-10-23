import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt


# ===================================
# Step 1. Raw Data Import
# ===================================
df = pd.read_csv("./data/5kg curl_EMG.csv", skiprows=3)
df.columns = ["sample", "biceps", "triceps", "triceps2"]  # 컬럼명 재설정
fs = 1000  # Hz
dt = 1 / fs
df["time"] = df["sample"] / fs  # Hz를 시간으로 변환


# ===================================
# Step 2. Offset Removal: Global Demean (전체에서 평균 빼기)
# 움직이지 않아도 신호가 0이 아닌 경우가 있기 때문에 보정을 통해 0이 되도록 만든다.
# ===================================
df["biceps_demean"] = df["biceps"] - np.mean(df["biceps"])
df["triceps_demean"] = df["triceps"] - np.mean(df["triceps"])


# ===================================
# Step 3. Filtering: Band-pass (20-450Hz) + Notch (50/60Hz)
# EMG 신호 안에 노이즈가 있을 수 있기 때문에 20-450Hz만 남기고 나머지를 제거한다.
# ===================================
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band') # 필터링 함수
    return filtfilt(b, a, data)

def notch_filter(data, fs, freq=60.0, Q=30):
    w0 = freq / (fs / 2)
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, data)

biceps_filt = bandpass_filter(df["biceps_demean"], 20, 450, fs)
biceps_filt = notch_filter(biceps_filt, fs, freq=60)

triceps_filt = bandpass_filter(df["triceps_demean"], 20, 450, fs)
triceps_filt = notch_filter(triceps_filt, fs, freq=60)

df["biceps_filt"] = biceps_filt
df["triceps_filt"] = triceps_filt

# 시각화
plt.figure(figsize=(10,5))
plt.plot(df["time"], df["biceps_demean"], label="Before Filtering")
plt.plot(df["time"], df["biceps_filt"], label="After Filtering")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.title("EMG Filtering (5kg Curl Biceps)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ===================================
# Step 4. Rectification
# EMG 신호는 양음 전위가 번갈아 진동 하는데 근육의 활성화만 보면 되기 때문에 신호의 크기를 절댓값으로 바꾼다.
# ===================================
df["biceps_rect"] = np.abs(df["biceps_filt"])
df["triceps_rect"] = np.abs(df["triceps_filt"])

# 시각화
plt.figure(figsize=(10,5))
plt.plot(df["time"], df["biceps_filt"], label="Filtered EMG")
plt.plot(df["time"], df["biceps_rect"], label="Rectified EMG")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.title("EMG Rectification (5kg Curl Biceps)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ===================================
# Step 5. Smoothing: Root Mean Square (RMS)
# 절댓값으로도 바꿔도 EMG 신호가 요동치기 때문에 50-100ms 단위로 평균화해서 근활성 추세를 본다.
# ===================================
window_ms = 50  # RMS 윈도우 길이 (50ms)
window_size = int(fs * (window_ms / 1000))  # 샘플 단위로 변환 (=50ms*1000Hz=50샘플)

df["biceps_rms"] = (df["biceps_rect"].rolling(window=window_size, center=True).apply(lambda x: np.sqrt(np.mean(x**2))))
df["triceps_rms"] = df["triceps_rect"].rolling(window=window_size, center=True).apply(lambda x: np.sqrt(np.mean(x**2)))

# 시각화
plt.figure(figsize=(10,5))
plt.plot(df["time"], df["biceps_rect"], label="Rectified EMG", alpha=0.5)
plt.plot(df["time"], df["biceps_rms"], label="Smoothed EMG (RMS, 50ms window)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (V)")
plt.title("EMG Smoothing (5kg Curl Biceps)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ===================================
# Step 6. Normalization (Maximum Voluntary Contraction, MVC)
# 최대 힘 대비 근육이 얼마나 활성화되었는지 확인하기 위해 각 신호를 MVC 값으로 나눈다.
# MVC를 따로 측정해야 하는데 측정을 안했을 경우, 실험 내 최대 RMS를 대체 MVC로 사용
# ===================================
MVC_biceps = df["biceps_rms"].max()
MVC_triceps = df["triceps_rms"].max()

# %MVC
df["biceps_norm"] = (df["biceps_rms"] / MVC_biceps) * 100
df["triceps_norm"] = (df["triceps_rms"] / MVC_triceps) * 100

# 시각화
plt.figure(figsize=(10,5))
plt.plot(df["time"], df["biceps_norm"], label="Biceps (% of Max RMS)")
plt.plot(df["time"], df["triceps_norm"], label="Triceps (% of Max RMS)")
plt.xlabel("Time (s)")
plt.ylabel("%MVC (relative to max RMS)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("EMG Normalization (Relative to Max RMS)")
plt.show()


# ===================================
# Step 7. Segmentation
# EMG 데이터에는 전체 동작이 포함되어 있기 때문에 동작 수행 구간만 잘라서 분석한다.
# ===================================
signal = df["biceps_norm"].fillna(0).values
time = df["time"].values

# Threshold 설정 (예: 10%MVC 이상일 때 근활성 시작)
threshold = 10
is_active = signal > threshold

# 활성 구간의 시작과 끝 찾기
segments = []
in_segment = False
for i in range(len(signal)):
    if is_active[i] and not in_segment:
        start_idx = i
        in_segment = True
    elif not is_active[i] and in_segment:
        end_idx = i
        in_segment = False
        # 너무 짧은 노이즈 구간(0.1초 이하)은 무시
        if (end_idx - start_idx) / 1000 > 0.1:
            segments.append((time[start_idx], time[end_idx]))

# 시각화
plt.figure(figsize=(10,5))
plt.plot(time, signal, label="Biceps %MVC")
for (start, end) in segments:
    plt.axvspan(start, end, color="red", alpha=0.2)
plt.xlabel("Time (s)")
plt.ylabel("%MVC")
plt.title("Automatic Segmentation by Threshold (10%MVC)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ===================================
# Step 8. Feature Extraction: Amplitude and Timing Features
# ===================================
features = []

for i, (start, end) in enumerate(segments, start=1):
    seg_mask = (df["time"] >= start) & (df["time"] <= end)
    seg_time = df.loc[seg_mask, "time"]
    seg_biceps = df.loc[seg_mask, "biceps_norm"]
    seg_triceps = df.loc[seg_mask, "triceps_norm"]

    # 평균 및 최대
    mean_biceps = seg_biceps.mean()
    peak_biceps = seg_biceps.max()
    mean_triceps = seg_triceps.mean()
    peak_triceps = seg_triceps.max()

    # 적분(iEMG)
    iEMG_biceps = np.sum(seg_biceps) * dt
    iEMG_triceps = np.sum(seg_triceps) * dt

    # onset/duration은 자동 segmentation 결과를 그대로 사용 가능
    onset_time = start
    duration = end - start

    features.append({
        "Segment": i,
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

# 표로 정리
feature_df = pd.DataFrame(features)
print("\n--- Feature Summary ---")
print(feature_df)

# CSV로 저장
feature_df.to_csv("emg_feature_summary_5kg.csv", index=False)