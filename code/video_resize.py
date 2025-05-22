import os
import pandas as pd
import subprocess

# === 設定路徑 ===
video_dir = r"E:\FOLDER\computer_vision\data\mp4_old"
output_dir = r"E:\FOLDER\computer_vision\data\mp4"
csv_path = r"E:\FOLDER\computer_vision\data\Metadata\LIVE-Meta-MobileCloudGaming_metadata.csv"
ffmpeg_path = r"C:\Users\binbin\Downloads\ffmpeg-2025-05-15-git-12b853530a-full_build\ffmpeg-2025-05-15-git-12b853530a-full_build\bin\ffmpeg.exe"  # ✅ 明確指定 ffmpeg 路徑

# 建立輸出資料夾（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# === 讀取 CSV ===
df = pd.read_csv(csv_path)
df = df.iloc[0:601]  # 取第2~601列

# === 處理每個影片 ===
for idx, row in df.iterrows():
    video_name = str(row.iloc[0])  # 第一欄應為影片檔名
    height = int(row['DisplayHeight'])
    width = int(row['DisplayWidth'])

    input_path = os.path.join(video_dir, video_name)
    output_path = os.path.join(output_dir, video_name)

    if not os.path.isfile(input_path):
        print(f"⚠️ 找不到影片：{input_path}")
        continue

    # === FFmpeg 指令（使用 NVIDIA GPU）===
    command = [
        ffmpeg_path,  # ✅ 使用明確的 ffmpeg 執行檔路徑
        '-hwaccel', 'cuda',
        '-i', input_path,
        '-vf', f'scale={width}:{height}',
        '-c:v', 'h264_nvenc',
        '-preset', 'fast',
        '-c:a', 'copy',
        '-y',
        output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ 成功處理：{video_name} → {width}x{height}")
    except subprocess.CalledProcessError:
        print(f"❌ 處理失敗：{video_name}")

print("\n🎬 所有影片 GPU 轉檔處理完成！")
