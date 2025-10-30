import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch


# ===================================
# Step 0. 필터 함수
# ===================================
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)


def notch_filter(data, fs, freq=60.0, Q=30):
    w0 = freq / (fs / 2)
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, data)


# ===================================
# Step 1. MVC 계산 함수
# ===================================
def compute_mvc(file_path, muscle_name, fs=1000):
    mvc_df = pd.read_csv(file_path)
    if muscle_name not in mvc_df.columns:
        raise ValueError(f"'{muscle_name}' not found in {file_path}. Available: {mvc_df.columns.tolist()}")

    signal = pd.to_numeric(mvc_df[muscle_name], errors='coerce').fillna(0)
    demean = signal - np.mean(signal)
    filt = notch_filter(bandpass_filter(demean, 20, 450, fs), fs, 60)
    rect = np.abs(filt)
    win = int(fs * 0.05)
    rms = pd.Series(rect).rolling(window=win, center=True).apply(lambda x: np.sqrt(np.mean(x ** 2)))
    mvc_val = rms.max(skipna=True)
    print(f"MVC ({muscle_name}) from {os.path.basename(file_path)} = {mvc_val:.4f}")
    return mvc_val


# ===================================
# Step 2. EMG 파일 처리 함수
# ===================================
def process_emg(file_path, MVC_biceps, MVC_triceps, MVC_triceps2, fs=1000):
    name = os.path.basename(file_path).replace(".csv", "")
    print(f"Processing: {name}")

    df = pd.read_csv(file_path, skiprows=3)
    df.columns = ["sample", "biceps", "triceps", "triceps2"]
    df["time"] = df["sample"] / fs
    dt = 1 / fs

    # 1. Offset 제거
    for m in ["biceps", "triceps", "triceps2"]:
        df[f"{m}_demean"] = df[m] - np.mean(df[m])

    # 2. Filtering
    for m in ["biceps", "triceps", "triceps2"]:
        filt = notch_filter(bandpass_filter(df[f"{m}_demean"], 20, 450, fs), fs, 60)
        df[f"{m}_filt"] = filt

    # 3. Rectification
    for m in ["biceps", "triceps", "triceps2"]:
        df[f"{m}_rect"] = np.abs(df[f"{m}_filt"])

    # 4. RMS smoothing (50 ms)
    win = int(fs * 0.05)
    for m in ["biceps", "triceps", "triceps2"]:
        df[f"{m}_rms"] = df[f"{m}_rect"].rolling(window=win, center=True).apply(lambda x: np.sqrt(np.mean(x ** 2)))

    # 5. Normalization (%MVC)
    df["biceps_norm"] = (df["biceps_rms"] / MVC_biceps) * 100
    df["triceps_norm"] = (df["triceps_rms"] / MVC_triceps) * 100
    df["triceps2_norm"] = (df["triceps2_rms"] / MVC_triceps2) * 100

    # 6. Segmentation (10% 이상)
    signal = df["biceps_norm"].fillna(0).values
    threshold = 10
    in_segment, segments = False, []
    for i in range(len(signal)):
        if signal[i] > threshold and not in_segment:
            start_idx, in_segment = i, True
        elif signal[i] <= threshold and in_segment:
            end_idx, in_segment = i, False
            if (end_idx - start_idx) / fs > 0.1:
                segments.append((df["time"][start_idx], df["time"][end_idx]))

    # 7. Feature Extraction
    features = []
    for i, (start, end) in enumerate(segments, 1):
        seg_mask = (df["time"] >= start) & (df["time"] <= end)
        seg_biceps = df.loc[seg_mask, "biceps_norm"]
        seg_triceps = df.loc[seg_mask, "triceps_norm"]
        seg_triceps2 = df.loc[seg_mask, "triceps2_norm"]

        features.append({
            "File": name,
            "Segment": i,
            "Start (s)": round(start, 3),
            "End (s)": round(end, 3),
            "Duration (s)": round(end - start, 3),
            "Mean_Biceps_%MVC": round(seg_biceps.mean(), 2),
            "Peak_Biceps_%MVC": round(seg_biceps.max(), 2),
            "iEMG_Biceps": round(np.sum(seg_biceps) * dt, 5),
            "Mean_Triceps_%MVC": round(seg_triceps.mean(), 2),
            "Peak_Triceps_%MVC": round(seg_triceps.max(), 2),
            "iEMG_Triceps": round(np.sum(seg_triceps) * dt, 5),
            "Mean_Triceps2_%MVC": round(seg_triceps2.mean(), 2),
            "Peak_Triceps2_%MVC": round(seg_triceps2.max(), 2),
            "iEMG_Triceps2": round(np.sum(seg_triceps2) * dt, 5),
            "MeanDiff_Biceps-Triceps": round(seg_biceps.mean() - seg_triceps.mean(), 2),
            "MeanDiff_Biceps-Triceps2": round(seg_biceps.mean() - seg_triceps2.mean(), 2),
            "PeakDiff_Biceps-Triceps": round(seg_biceps.max() - seg_triceps.max(), 2),
            "PeakDiff_Biceps-Triceps2": round(seg_biceps.max() - seg_triceps2.max(), 2),
        })

    feature_df = pd.DataFrame(features)
    feature_df.to_csv(f"features_{name}.csv", index=False)
    return feature_df


# ===================================
# Step 3. 실행 파트
# ===================================
fs = 1000
MVC_biceps = compute_mvc("./data/biceps_MVIC_EMG.csv", "1. Biceps", fs)
MVC_triceps = compute_mvc("./data/triceps_MVIC_EMG.csv", "2. Triceps", fs)
MVC_triceps2 = compute_mvc("./data/triceps_MVIC_EMG.csv", "3. Tricpes2", fs)

files = sorted(glob.glob("./data/*kg_*.csv"))
all_results = []

for f in files:
    if "MVIC" not in f:  # MVC 파일 제외
        res = process_emg(f, MVC_biceps, MVC_triceps, MVC_triceps2, fs)
        all_results.append(res)

final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv("./results/all_emg_features.csv", index=False)
print("\n✅ 결과는 all_emg_features.csv 에 저장되었습니다.")


# ===================================
# Step 4. 요약 테이블 생성
# ===================================
print("\n📊 Generating summary table...")

# 하중(5,10,14) 및 운동 종류(Arm Curl / Hammer Curl) 추출
final_df["Load"] = final_df["File"].str.extract(r"(\d+)kg")[0].astype(int)
final_df["Exercise"] = final_df["File"].apply(lambda x: "Arm Curl" if "curl" in x.lower() else "Hammer Curl")

# 하중 & 운동별 평균 계산
summary = final_df.groupby(["Load", "Exercise"]).agg({
    "Mean_Biceps_%MVC": "mean",
    "Peak_Biceps_%MVC": "mean",
    "Mean_Triceps_%MVC": "mean",
    "Peak_Triceps_%MVC": "mean",
    "Mean_Triceps2_%MVC": "mean",
    "Peak_Triceps2_%MVC": "mean"
}).reset_index()

# 보기 좋은 컬럼명으로 정리
summary = summary.rename(columns={
    "Mean_Biceps_%MVC": "Biceps_Mean_RMS(%MVC)",
    "Peak_Biceps_%MVC": "Biceps_Peak_RMS(%MVC)",
    "Mean_Triceps_%MVC": "Triceps_Mean_RMS(%MVC)",
    "Peak_Triceps_%MVC": "Triceps_Peak_RMS(%MVC)",
    "Mean_Triceps2_%MVC": "Triceps2_Mean_RMS(%MVC)",
    "Peak_Triceps2_%MVC": "Triceps2_Peak_RMS(%MVC)"
})

# 결과 출력
print("\n============================")
print("Mean / Peak RMS (%MVC) Summary by Load & Exercise")
print("============================")
print(summary.to_string(index=False))

# csv 저장
output_path = "./results/all_emg_summary.csv"
summary.to_csv(output_path, index=False)
print(f"\n✅ 요약 테이블 저장 완료: {output_path}")


# ===================================
# Step 5. 시각화
# ===================================
sns.set(style="whitegrid")

# Mean RMS (%MVC) vs Load
plt.figure(figsize=(8,6))
sns.lineplot(
    data=summary,
    x="Load",
    y="Biceps_Mean_RMS(%MVC)",
    hue="Exercise",
    style="Exercise",
    markers=True,
    dashes=False,
    linewidth=2.5
)
plt.title("Biceps Mean RMS (%MVC) vs Load (Arm Curl vs Hammer Curl)")
plt.xlabel("Load (kg)")
plt.ylabel("Mean RMS (%MVC)")
plt.legend(title="Exercise")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Peak RMS (%MVC) vs Load
plt.figure(figsize=(8,6))
sns.lineplot(
    data=summary,
    x="Load",
    y="Biceps_Peak_RMS(%MVC)",
    hue="Exercise",
    style="Exercise",
    markers=True,
    dashes=False,
    linewidth=2.5
)
plt.title("Biceps Peak RMS (%MVC) vs Load (Arm Curl vs Hammer Curl)")
plt.xlabel("Load (kg)")
plt.ylabel("Peak RMS (%MVC)")
plt.legend(title="Exercise")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("✅ 시각화 완료!")



# # -------------------------------
# # (2) 동일 하중에서 Arm Curl vs Hammer Curl 비교 (Bar plot)
# # -------------------------------
# plt.figure(figsize=(8,6))
# sns.barplot(
#     data=summary,
#     x="Load",
#     y="Biceps_Mean_RMS(%MVC)",
#     hue="Exercise",
#     palette="Blues"
# )
# plt.title("Comparison of Biceps EMG Amplitude (%MVC)\nArm Curl vs Hammer Curl at Same Load")
# plt.xlabel("Load (kg)")
# plt.ylabel("Mean RMS (%MVC)")
# plt.legend(title="Exercise")
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.savefig("./data/biceps_emg_comparison_bar.png", dpi=300)
# plt.show()
#
# print("\n✅ 시각화 완료! 그래프가 './data/' 폴더에 저장되었습니다.")
